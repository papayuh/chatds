#!/usr/bin/env python3
"""R4 curriculum v2 framework: pools, wrappers, engine, contamination
filter, writer, manifest, suite-coverage check. stdlib only, no network,
nothing generated is ever executed (ast.parse only).

R3 (corpus/python_curriculum.py) failed because ~25 identifiers per pool x
64 parameter tuples produced 3,593 distinct answers in 44k rows: the model
learned a lookup table instead of learning to COPY identifiers and literals
out of the prompt. v2 fixes that with big pools and one distinct parameter
tuple per training row.

Recipe modules (corpus/curriculum_v2_python.py, corpus/curriculum_v2_bugfix.py)
do `from corpus.curriculum_v2 import R`; R and POOLS are therefore defined
before any recipe module is imported, and the recipe modules are imported
lazily inside load_recipes() so the cycle never bites.

Usage:
    python3 corpus/curriculum_v2.py --out-dir /tmp/cv2 [--seed 42]
    python3 corpus/curriculum_v2.py --check-suite
"""
import argparse
import ast
import builtins
import hashlib
import json
import keyword
import os
import random
import re
import string as _string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, HERE, os.path.join(REPO_ROOT, "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import postprocess as pp   # noqa: E402  (sha-pinned suite + normalize_key)
import offline_python as op  # noqa: E402  (trusted filter_pairs contract)

SOURCE = "curriculum-v2"
DEFAULT_SEED = 42
TRAIN_PER_FAMILY = 4150      # ceiling of kept train rows per family. R8 used
                             # 4700 for the 175-family corpus; R9's bigger
                             # (thousands-of-entries) identifier pools raise
                             # per-family pool CAPACITY past 4700 for many
                             # families that used to be capacity-limited
                             # below it, which alone would have pushed the
                             # same 175 families past 735k rows -- lowered to
                             # land back in the ~600-700k target range.
MIN_FAMILY_ROWS = 1500       # floor for families with a tiny parameter space
                             # (single-slot families like len(xs) are the
                             # tier-1 suite items; 200 rows starved them)
ROW_CAP = None               # hard per-family cap (tests only); overrides rows=
MAX_ATTEMPTS = 20000
DEV_TUPLES = 8               # dev parameter tuples per (wrapper, core)
# R9: fraction of python-family train rows deliberately forced onto the bare
# "{core}" wrapper (no wrapper text, no "Python" mention) with an
# imperative/noun-phrase core -- uniform random wrapper choice picked one of
# these among ~44 per family only ~1-2% of the time (measured on R8's
# build), well under the >=10% overall / >=5% per-family floor the R9 spec
# requires. vp3 (third-person) cores have no bare wrapper by design (R8),
# so python families still clear the floor with margin; bugfix keeps its
# existing uniform choice (BUG_WRAPPERS has no bare "{core}" entry).
BARE_FRACTION = 0.16
PROMPT_MAX_BYTES = 160
TRAIN_FILE = "python-curriculum-v2-train.jsonl"
DEV_FILE = "python-curriculum-v2-dev.jsonl"
MANIFEST_FILE = "manifest.json"
FORBIDDEN_DIR = os.path.realpath(os.path.join(HERE, "results"))

# R11: second prompt source -- LLM-written casual templates per family, see
# corpus/llm_templates/FAMILIES.md. A family with no template file behaves
# exactly as before (P_LLM never fires because the template list is empty).
P_LLM = 0.5
LLM_TEMPLATES_DIR = os.path.join(HERE, "llm_templates")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Recipe contract (fixed; both recipe modules use exactly this)
# ---------------------------------------------------------------------------

def R(family, category, tier, suite_ids, cores, dev_cores, slots, answer,
      bug=None, guard=None, parse=True, suite_bindings=None, extra=None,
      rows=None, answer_only_slots=(), literal_slot=None, literal_kind="str",
      literal_share=0.30):
    """`extra` is callable(raw_combo) -> dict of display-only names usable in
    a core (e.g. the inclusive end "0 to 4" for the answer range(5)). Slots
    still render normally; extra names never reach the answer.

    `rows` overrides TRAIN_PER_FAMILY for this family (still clamped by the
    pool product): R5 scored bugfix 44/50 because ~39 bugfix templates per
    family were memorised as shapes, so the context/semantic families need a
    bigger share of the corpus than the tier-1 syntax ones.

    `answer_only_slots`: bugfix slot names the copy/presence invariant in
    _render should skip -- for a slot the model must GENERATE (any valid
    value accepted) rather than copy out of the prompt, e.g. the arbitrary
    digit in `int('1')` where the suite accepts any digit string.

    `literal_slot`: R9 fix for string/list families that "mostly use
    variable operands" -- with probability `literal_share`, `_render`
    overrides this slot's rendered prompt AND code with a LITERAL value
    (a quoted word for literal_kind="str", a small list literal for
    literal_kind="list") instead of the sampled identifier, so the model
    also trains on prompts like "the string 'walrus'" whose answer is
    `'walrus'.upper()`, not a variable. The same literal text is used for
    both the prompt and the answer, so the prompt's quote style (single or
    double) is always the answer's quote style too."""
    return {"family": family, "category": category, "tier": tier,
            "suite_ids": list(suite_ids), "cores": list(cores),
            "dev_cores": list(dev_cores), "slots": dict(slots),
            "answer": answer, "bug": bug, "guard": guard, "parse": parse,
            "suite_bindings": suite_bindings or {}, "extra": extra,
            "rows": rows, "answer_only_slots": tuple(answer_only_slots),
            "literal_slot": literal_slot, "literal_kind": literal_kind,
            "literal_share": literal_share}


# A python core carries its GRAMMATICAL FORM as a prefix, because the suite
# asks the same question three ways and R4 only ever trained the first:
#   "get the largest value in {xs}"          imperative  (plain, no prefix)
#   "np:the largest value in the list {xs}"  noun phrase
#   "vp3:reverses the list {xs}"             third-person verb phrase
# The framework picks the wrapper set from the form, so a family's cores can
# mix forms freely.
def core_form(core):
    for tag in ("np", "vp3"):
        if core.startswith(tag + ":"):
            return tag, core[len(tag) + 1:]
    return "imp", core


# ---------------------------------------------------------------------------
# Pools. kind decides rendering:
#   name  -> bare identifier in prompt and in code
#   quote -> "v" in the prompt, repr(v) in the code
#   bare  -> unquoted words in the prompt, repr(v) in the code
#   num   -> repr(v) in both
# Train and dev slices are disjoint per pool: that is half the split (the
# other half is dev-only wrappers x dev-only cores).
# ---------------------------------------------------------------------------

def _p(kind, train, dev):
    train, dev = list(dict.fromkeys(train)), list(dict.fromkeys(dev))
    assert not (set(train) & set(dev)), (kind, set(train) & set(dev))
    return {"kind": kind, "train": train, "dev": dev}


def _mix(prefixes, suffixes, sep="_"):
    return [p + sep + s for p in prefixes for s in suffixes]


_LIST_ROOTS = [
    "xs", "ys", "lst", "nums", "items", "values", "data", "scores", "ages",
    "names", "words", "rows", "cols", "temps", "prices", "heights", "weights",
    "speeds", "times", "rates", "runs", "hits", "counts", "points", "marks",
    "entries", "records", "results", "samples", "tokens", "lines", "paths",
    "keys", "vals", "codes", "ids", "tags", "cells", "bins", "buckets",
    "amounts", "totals", "deltas", "labels", "colors", "sizes", "angles",
    "volts", "amps", "watts"]
_LIST_ROOTS_DEV = [
    "pips", "coins", "votes", "slots", "crates", "barrels", "pebbles",
    "tiles", "grains", "shards", "beads", "planks", "reels", "spools",
    "flasks", "lanterns", "anchors", "kites"]
_LIST_B_ROOTS = [
    "ys", "others", "extra", "rest", "more", "spares", "backups", "tails",
    "heads", "mids", "evens", "odds", "alts", "clones", "mirrors",
    "second", "third", "partners", "mates", "peers"]
_LIST_B_ROOTS_DEV = ["twins", "duals", "echoes", "shadows", "spouts", "prongs"]

_STR_ROOTS = [
    "s", "text", "word", "name", "line", "msg", "title", "sentence", "phrase",
    "label", "note", "body", "caption", "header", "footer", "greeting",
    "message", "subject", "author", "city", "country", "street", "email",
    "phone", "code", "token", "slug", "path", "filename", "stem", "value",
    "desc", "summary", "comment", "query", "reply", "topic", "blurb",
    "quote_text", "heading"]
_STR_ROOTS_DEV = [
    "chant", "motto", "banner", "ditty", "riddle", "quip", "verse",
    "psalm", "hymn", "rune", "glyph", "scrawl"]
_STR_B_ROOTS = [
    "other", "suffix", "prefix", "ending", "piece", "chunk", "intro",
    "outro", "front", "back", "middle", "core", "rim", "wrap", "tailtext",
    "headtext", "second_str", "extra_str", "next_str", "alt_text"]
_STR_B_ROOTS_DEV = ["closer", "opener", "coda", "preface"]

_DICT_ROOTS = [
    "d", "info", "config", "user", "counts", "lookup", "table", "record",
    "props", "fields", "cache", "stats", "meta", "opts", "params", "settings",
    "registry", "mapping", "inventory", "catalog", "profile", "session",
    "prefs", "totals", "weights_by", "bag", "sheet", "roster", "ledger",
    "manifest"]
_DICT_ROOTS_DEV = [
    "specs", "menu2", "cards", "shelves", "vaults", "crates_by",
    "atlas", "almanac"]

_NUM_ROOTS = [
    "x", "n", "total", "count", "value", "age", "price", "a", "num", "amount",
    "size", "width", "height", "depth", "limit", "base", "rate", "level",
    "score", "index", "offset", "delta", "step", "gap", "span", "area",
    "speed", "mass", "cost", "temp", "angle", "radius", "length", "volume",
    "power", "energy", "force", "seconds", "year", "budget"]
_NUM_ROOTS_DEV = [
    "q", "g", "vv", "mag", "yield_", "tally", "quota", "margin", "pulse",
    "cadence"]
_NUM_B_ROOTS = [
    "y", "b", "m", "k", "z", "w", "r", "t", "u", "c", "factor", "divisor",
    "addend", "scale", "shift", "bias", "second_num", "other_num",
    "extra_num", "next_num"]
_NUM_B_ROOTS_DEV = ["h", "p", "j", "e2"]

_ITEM_ROOTS = [
    "x", "item", "elem", "v", "val", "e", "n", "c", "ch", "w", "tok",
    "node", "obj", "thing", "piece", "unit", "entry", "cell", "bit", "part"]
_ITEM_ROOTS_DEV = ["nib", "cog", "dot", "pip"]

_IDX_ROOTS = ["i", "j", "idx", "pos", "k", "n", "index", "at", "slot", "cursor"]
_IDX_ROOTS_DEV = ["m", "spot", "place"]

_SET_ROOTS = [
    "seen", "tags_set", "flags", "kinds", "types", "group", "chosen",
    "picked", "unique", "visited", "members", "allowed", "blocked",
    "active", "pending", "known"]
_SET_ROOTS_DEV = ["roles", "modes", "realms", "guilds"]
_SET_B_ROOTS = [
    "past", "others_set", "extras", "spares_set", "wanted", "banned",
    "queued", "done_set"]
_SET_B_ROOTS_DEV = ["sides", "zones"]

_TUPLE_ROOTS = [
    "t", "pair", "trio", "quad", "row_t", "node_t", "edge", "link",
    "stepper", "jump", "move", "turn", "fold", "wrapper", "bind", "knot",
    "coord", "shape_t", "range_t", "span_t"]
_TUPLE_ROOTS_DEV = ["duo", "boxt", "packt"]

_WORDS = """hello world abc apple banana cherry grape lemon melon peach berry
mango olive pear lime corn rice oats salt sugar bread cheese butter honey
water juice coffee tea milk cake pie soup toast jam bean nut seed leaf root
stem bark wood stone sand clay glass metal iron gold silver copper paper cloth
silk wool cotton rope chain hook nail screw bolt wheel gear spring lever pipe
wire cable plug lamp bulb torch flame smoke ash dust mist rain snow hail wind
storm cloud sky star moon sun comet planet ocean river lake pond creek shore
beach cliff hill mount valley field farm barn fence gate road path bridge
tunnel town city village market shop stall bench chair table shelf drawer box
crate basket bucket jar bottle cup plate bowl spoon fork knife dog cat bird
fish frog snake mouse horse sheep goat cow duck goose swan owl hawk crow robin
finch bee ant moth wasp beetle spider worm snail crab clam a b c d e f g h ok
hi yes no red blue green black white gray pink teal navy
brick tile beam plank rivet hinge latch clasp buckle ribbon thread needle
button pocket collar sleeve cuff boot glove scarf helmet shield sword arrow
bow spear armour banner crest crown throne castle tower moat wall arch dome
vault attic cellar porch patio garden hedge lawn bush vine fern moss reed
willow birch cedar maple aspen alder rowan hazel walnut almond cashew raisin
apricot plumcake muffin scone waffle pancake omelet noodle pasta gravy
broth stew roast chowder cider nectar syrup vinegar mustard pepper ginger
clove nutmeg cocoa vanilla caramel toffee nougat sorbet gelato yogurt
salmon trout perch bream tuna squid prawn oyster mussel lobster otter
badger weasel marten stoat hare rabbit squirrel beaver rat vole shrew
mole hedgehog lynx puma jaguar cheetah leopard panther bison moose elk
deer boar goatling calf foal piglet lambkin kitten puppy cygnet""".split()

_CAP_WORDS = ([
    "Hello", "Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace",
    "Heidi", "Ivan", "Judy", "Ok", "Hi", "Welcome", "Goodbye", "Morning",
    "Evening", "Sunday", "Monday", "Tuesday", "Friday", "January", "March",
    "April", "June", "Paris", "Tokyo", "Berlin", "Madrid", "Lisbon", "Oslo",
    "Dublin", "Cairo", "Lima", "Quito", "Yes", "No", "Ready", "Done", "Start",
    "Stop", "Left", "Right", "North"]
    + ["Sofia", "Milan", "Prague", "Vienna", "Athens", "Boston", "Denver",
       "Austin", "Dallas", "Seattle", "Toronto", "Ottawa", "Havana", "Bogota",
       "Nairobi", "Accra", "Dakar", "Rabat", "Tunis", "Amman", "Doha",
       "Kyoto", "Osaka", "Seoul", "Hanoi", "Manila", "Perth", "Hobart",
       "Naomi", "Oscar", "Peter", "Quinn", "Rosa", "Simon", "Tina", "Ursula",
       "Victor", "Wendy", "Xavier", "Yolanda", "Zack", "Nina", "Omar",
       "Priya", "Rafael", "Sasha", "Tomas", "Uma", "Vera", "Wesley",
       "Later", "Sooner", "Always", "Never", "Maybe", "Perhaps", "Indeed",
       "Almost", "Nearly", "Onward", "Upward", "Homeward"])

# Combinatorial: 1600 plain two/three-word messages, so no single message can
# be memorised as a constant -- the model has to copy it out of the prompt.
_MSG_A = ["hello", "good", "great", "nice", "warm", "cold", "bright", "quiet",
          "loud", "new", "old", "fresh", "clear", "dark", "light", "soft",
          "quick", "slow", "wide", "calm"]
_MSG_B = ["world", "morning", "evening", "day", "night", "start", "finish",
          "news", "work", "plan", "idea", "move", "step", "turn", "room",
          "road", "sky", "sea", "song", "game"]
_MESSAGES = ([a + " " + b for a in _MSG_A for b in _MSG_B]
             + ["the " + a + " " + b for a in _MSG_A for b in _MSG_B]
             + [a + " " + b + " today" for a in _MSG_A for b in _MSG_B]
             + [a + " " + b + " again" for a in _MSG_A for b in _MSG_B])
_MSG_A_DEV = ["kind", "silent", "golden", "lucky", "brave", "eager"]
_MSG_B_DEV = ["harbour", "meadow", "lantern", "compass", "orchard", "beacon"]
_MESSAGES_DEV = ([a + " " + b for a in _MSG_A_DEV for b in _MSG_B_DEV]
                 + ["the " + a + " " + b
                    for a in _MSG_A_DEV for b in _MSG_B_DEV])

_KEYS = ([
    "k", "a", "b", "name", "age", "id", "city", "color", "price", "score",
    "email", "phone", "title", "year", "size", "count", "total", "level",
    "kind", "type", "state", "status", "owner", "label", "code", "note",
    "date", "time", "rank", "tier", "group", "team", "unit", "zone", "mode",
    "step", "part", "slot", "seat", "room", "floor", "block", "wing", "gate",
    "lane", "track", "stage", "phase", "round", "shift"]
    + _mix(("user", "item", "row", "cell", "job"),
           ("id", "name", "key", "code", "tag"), sep="_")
    + _mix(("first", "last", "next", "prev"), ("name", "id", "step"), sep="_")
    + ["k1", "k2", "k3", "kk", "aa", "bb", "cc", "dd", "ee", "ff",
       "q1", "q2", "q3", "q4", "x1", "x2", "y1", "y2", "z1", "z2",
       "alpha", "beta", "gamma", "delta", "theta", "sigma", "omega",
       "north", "south", "east", "west", "up", "down", "left", "right",
       "small", "large", "medium", "wide", "tall", "deep", "flat", "round2",
       "start_at", "end_at", "made_on", "seen_on", "paid_on", "due_on",
       "host", "port", "user2", "pass2", "token2", "scope", "realm", "node",
       "edge", "leaf", "depth2", "width2", "height2", "weight2", "colour",
       "shape", "sound", "smell", "taste", "touch", "speed2", "power2"])

_FILE_STEMS = [
    "data", "notes", "log", "out", "input", "report", "scores", "users",
    "items", "config", "readme", "dump", "raw", "clean", "final", "draft",
    "index", "listing", "table", "sheet", "record", "entry", "sample",
    "batch", "chunk", "part", "page", "book", "story", "letter", "memo",
    "plan", "budget", "sales", "orders", "stock", "prices", "counts",
    "totals", "results", "output", "errors", "debug", "trace", "audit",
    "history", "export", "summary", "detail", "roster", "ledger", "invoice",
    "receipt", "manifest", "schedule", "roadmap", "backlog", "survey"]
_FILE_EXTS = ("txt", "csv", "json", "log", "md", "dat")

# Short (1-2 char) identifier candidates, shared by every _gen_ident_pool()
# category below: real users write these regardless of variable type, and
# because each category's generated pool has thousands of entries, a name
# shared across categories still lands far under the per-identifier cap
# (see _gen_ident_pool and TestIdentifierPools).
_SHORT_NAMES_TRAIN = ["a", "b", "c", "d", "n", "q", "r", "s", "x", "y",
                      "xs", "ys", "k", "v", "i", "j"]
_SHORT_NAMES_DEV = ["m", "w", "z", "xy", "ab"]

# Generic mid/suffix vocabulary every _gen_ident_pool() category combines
# with its own roots to build 2/3-part names.
_LONG_MIDS = ["step", "user", "order", "item", "score", "session", "batch",
              "record", "sensor", "report"]
_LONG_SUFFIXES = ["count", "counts", "value", "values", "total", "totals",
                  "list", "rate", "score", "log"]
_LONG_MIDS_DEV = ["client", "device", "queue"]
_LONG_SUFFIXES_DEV = ["totals", "rates", "log"]

_RESERVED_IDENTS = set(keyword.kwlist) | set(dir(builtins))

# R10 fix for the identifier/literal-copy failure measured on R9: identifier
# roots came from ~20-30 hand-picked words per category and string
# keys/literals from the ~300-word _WORDS/_KEYS lists, so the model only ever
# learned to copy THOSE subword sequences -- a novel two-part identifier it
# hasn't seen as a root comes back with a piece repeated or dropped
# ("xxx_levels" -> "xxx_leveve"), and a novel quoted key gets replaced by
# some other word from the prompt. Load the system dictionary once (falls
# back to the existing curated lists if it is missing) and split it
# deterministically by word -- not by build -- so dev words never leak into
# train regardless of how many times a pool draws from it.
_DICT_WORDS_PATH = "/usr/share/dict/words"


def _load_dict_words():
    try:
        with open(_DICT_WORDS_PATH, encoding="utf-8", errors="ignore") as f:
            raw = f.read().split()
    except OSError:
        raw = []
    train, dev, seen = [], [], set()
    for w in raw:
        if not (3 <= len(w) <= 12) or not w.isascii() or not w.isalpha() \
                or not w.islower():
            continue
        if w in seen or w in _RESERVED_IDENTS:
            continue
        seen.add(w)
        bucket = hashlib.sha1(w.encode()).digest()[0] % 10
        (dev if bucket == 0 else train).append(w)
    if not train:   # dictionary file missing -- reuse the curated word list
        train, dev = list(_WORDS), ["kiwi", "plum", "fig", "date", "yam"]
    return train, dev


_DICT_WORDS_TRAIN, _DICT_WORDS_DEV = _load_dict_words()
# Every dictionary-built identifier _gen_ident_pool() has ever produced (both
# train and dev, every category) -- lets tests measure the dictionary share
# of a sample of drawn identifiers without threading a name through every
# _gen_ident_pool() call site.
_DICT_BUILT_IDENT_POOL = set()


def _dict_word_combo(words, count, seed, sep="_", digit_chance=0.0):
    """`count` deterministic 1-3-word combinations of `words` joined by
    `sep`, optionally with a trailing digit -- a fresh, effectively
    never-repeating draw instead of a small fixed list, so copying is forced
    across thousands of subword sequences rather than ~20-300 memorized
    ones. Seeded so the whole build stays reproducible."""
    if not words or count <= 0:
        return []
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        combo = sep.join(rng.choice(words) for _ in range(rng.randint(1, 3)))
        if digit_chance and rng.random() < digit_chance:
            combo += str(rng.randint(2, 9))
        out.append(combo)
    return out


def _dict_words_sample(words, count, seed):
    """`count` distinct dictionary words (falls back to sampling with
    replacement if `count` exceeds the pool)."""
    if not words:
        return []
    rng = random.Random(seed)
    if count <= len(words):
        return rng.sample(words, count)
    return [rng.choice(words) for _ in range(count)]


def _dedup_dev(train, dev):
    """Drop any dev candidate a train candidate already claims -- needed
    once both sides can draw from the same dictionary word source."""
    train_have = set(train)
    return [w for w in dev if w not in train_have]


def _clean_idents(words):
    """Dedup (order-preserving) and drop anything that collides with a
    Python keyword or builtin -- a generated identifier must never shadow
    one, since it can land in a generated ANSWER, not just a prompt."""
    out, seen = [], set()
    for w in words:
        if w in seen or w in _RESERVED_IDENTS or not w.isidentifier():
            continue
        seen.add(w)
        out.append(w)
    return out


def _gen_ident_pool(kind, roots_train, roots_dev, short_train, short_dev,
                     mids_train=None, suffixes_train=None,
                     mids_dev=None, suffixes_dev=None, digit_span=4):
    """R9 fix for the identifier-copy failure: R8's variable pools were flat
    curated lists (e.g. dict_var's 30-ish words including "inventory"), so
    one training build put 725 rows under the bare name "inventory" and
    nothing ever trained the model to keep copying once it recognised that
    prefix -- "inventory_count" in a prompt came back as just "inventory".
    Generate 1/2/3-part snake_case names at pool-build time instead: every
    root is used BARE (x) and as the first component of a 2-part (x_y) and
    3-part (x_y_z) extension, so those forms all occur together and copying
    PAST a familiar first word is forced, not incidental. Pool sizes land in
    the thousands, so no single string can reach anywhere near R8's 725-row
    share of a ~650k-row corpus (<=0.05% is ~325 rows) -- see
    corpus/test_curriculum_v2.py::TestIdentifierPools.
    """
    mids_train = mids_train if mids_train is not None else _LONG_MIDS
    suffixes_train = suffixes_train if suffixes_train is not None else _LONG_SUFFIXES
    mids_dev = mids_dev if mids_dev is not None else _LONG_MIDS_DEV
    suffixes_dev = suffixes_dev if suffixes_dev is not None else _LONG_SUFFIXES_DEV

    def build(roots, short, mids, suffixes, dict_words):
        one = list(roots) + list(short)
        two = [f"{r}_{s}" for r in roots for s in suffixes]
        three = [f"{r}_{m}_{s}" for r in roots for m in mids for s in suffixes]
        digits = [f"{r}{n}" for r in roots for n in range(2, digit_span)]
        curated = _clean_idents(one + two + three + digits)
        # R10: curated roots stay ~40% of draws (they carry type meaning,
        # e.g. "scores"/"names"); the rest are fresh dictionary-built 1-3
        # word names -- 1.5x the curated count makes
        # dict/(curated+dict) == 0.6.
        seed = "dictident:" + ",".join(sorted(roots)) + "|" + ",".join(sorted(short))
        dict_built = _dict_word_combo(dict_words, round(len(curated) * 1.5),
                                      seed, sep="_", digit_chance=0.10)
        _DICT_BUILT_IDENT_POOL.update(dict_built)
        return _clean_idents(curated + dict_built)

    train = build(roots_train, short_train, mids_train, suffixes_train,
                 _DICT_WORDS_TRAIN)
    dev = build(roots_dev, short_dev, mids_dev, suffixes_dev, _DICT_WORDS_DEV)
    # Single-letter roots/short-names are deliberately shared across
    # categories and sides (real users reuse "x"/"m"/"w" regardless of
    # type); drop any dev candidate a train candidate already claims rather
    # than hand-auditing every category's letter choices for overlap.
    train_have = set(train)
    dev = [w for w in dev if w not in train_have]
    return _p(kind, train, dev)


# R10: string keys/literals/messages get the same dictionary mix as
# identifiers (~50% of draws) -- the ~300-word _WORDS/_KEYS lists had the
# same copy-only-these-N-words failure mode as the identifier pools.
_WORD_LIT_DEV_BASE = [
    "kiwi", "plum", "fig", "date", "yam", "leek", "kale", "chard",
    "quince", "medlar", "sloe", "damson", "citron", "pomelo", "loquat"]
_WORD_LIT_TRAIN = _WORDS + _dict_words_sample(
    _DICT_WORDS_TRAIN, len(_WORDS), "word_lit:train")
_WORD_LIT_DEV = _dedup_dev(_WORD_LIT_TRAIN, _WORD_LIT_DEV_BASE
                          + _dict_words_sample(_DICT_WORDS_DEV,
                                              len(_WORD_LIT_DEV_BASE),
                                              "word_lit:dev"))

_KEY_LIT_DEV_BASE = [
    "region", "sector", "batch", "lot", "shelf", "aisle", "bay", "dock",
    "berth", "quay", "hold", "crate_id"]
_KEY_LIT_TRAIN = _KEYS + _dict_words_sample(
    _DICT_WORDS_TRAIN, len(_KEYS), "key_lit:train")
_KEY_LIT_DEV = _dedup_dev(_KEY_LIT_TRAIN, _KEY_LIT_DEV_BASE
                         + _dict_words_sample(_DICT_WORDS_DEV,
                                             len(_KEY_LIT_DEV_BASE),
                                             "key_lit:dev"))

# Messages: "1-3 dictionary words joined by spaces" per the R10 spec, rather
# than single words -- keeps the multi-word "bare" rendering realistic.
_MESSAGE_LIT_TRAIN = _MESSAGES + _dict_word_combo(
    _DICT_WORDS_TRAIN, len(_MESSAGES), "message_lit:train", sep=" ")
_MESSAGE_LIT_DEV = _dedup_dev(_MESSAGE_LIT_TRAIN, _MESSAGES_DEV
                             + _dict_word_combo(_DICT_WORDS_DEV,
                                               len(_MESSAGES_DEV),
                                               "message_lit:dev", sep=" "))


POOLS = {
    "list_var": _gen_ident_pool("name", _LIST_ROOTS, _LIST_ROOTS_DEV,
                                _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "list_var_b": _gen_ident_pool("name", _LIST_B_ROOTS, _LIST_B_ROOTS_DEV,
                                  _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "str_var": _gen_ident_pool("name", _STR_ROOTS, _STR_ROOTS_DEV,
                               _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "str_var_b": _gen_ident_pool("name", _STR_B_ROOTS, _STR_B_ROOTS_DEV,
                                 _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "dict_var": _gen_ident_pool("name", _DICT_ROOTS, _DICT_ROOTS_DEV,
                                _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "num_var": _gen_ident_pool("name", _NUM_ROOTS, _NUM_ROOTS_DEV,
                               _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "num_var_b": _gen_ident_pool("name", _NUM_B_ROOTS, _NUM_B_ROOTS_DEV,
                                 _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "item_var": _gen_ident_pool("name", _ITEM_ROOTS, _ITEM_ROOTS_DEV,
                                _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "idx_var": _gen_ident_pool("name", _IDX_ROOTS, _IDX_ROOTS_DEV,
                               _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    # Function names occur in definitions, calls and bugfixes. Give them
    # the same compositional copy practice as variable names rather than a
    # small fixed list of verbs (while keeping the short suite bindings).
    "fn_name": _gen_ident_pool(
        "name", ["add", "f", "g", "total", "combine", "area", "greet",
                 "compute", "calc", "apply", "run", "make", "build",
                 "handle", "process", "update", "merge", "render",
                 "check", "verify", "load", "save", "send", "fetch",
                 "parse", "convert", "scale", "shift", "rotate",
                 "normalize", "summarize", "report", "notify", "init",
                 "reset", "finish"],
        ["combine_all", "sum_n", "tally_up", "pick_one", "zip_up",
         "fold_in", "name_it", "cap_it"],
        ["f", "g", "get_value", "do_work", "split_up", "format_it"],
        ["make_it", "do_next"]),
    "cls_name": _p("name", [
        "Dog", "Cat", "Car", "Point", "User", "Node", "Shape", "Robot",
        "Book", "Tree", "Stack", "Queue", "Timer", "Board", "Piece", "Token",
        "Player", "Level", "Sprite", "Buffer", "Sensor", "Motor", "Screen",
        "Button", "Window", "Ticket", "Order", "Basket", "Wallet", "Engine"],
        ["Kite", "Anvil", "Lamp", "Crate", "Beacon"]),
    # Capitalized words render bare so they can sit inside an f-string body.
    "Word_lit": _p("name", _CAP_WORDS, [
        "Zara", "Pluto", "Vega", "Rigel", "Altair", "Mira", "Lyra", "Orion"]),
    "word_lit": _p("quote", _WORD_LIT_TRAIN, _WORD_LIT_DEV),
    "message_lit": _p("bare", _MESSAGE_LIT_TRAIN, _MESSAGE_LIT_DEV),
    "key_lit": _p("quote", _KEY_LIT_TRAIN, _KEY_LIT_DEV),
    "sep_lit": _p("quote", [",", ", ", "-", " ", "_", ";", "/", ":"],
                  ["|", " - "]),
    # "pair" kind: the prompt shows a plain English name for the separator,
    # the code gets the character. py-020 asks for "with commas", not '","'.
    "sep_named": _p("pair", [
        ("commas", ","), ("semicolons", ";"), ("spaces", " "),
        ("dashes", "-"), ("slashes", "/"), ("colons", ":"),
        ("underscores", "_"), ("plus signs", "+"), ("dots", "."),
        ("pipes", "|"), ("stars", "*"), ("equals signs", "="),
        ("hashes", "#"), ("at signs", "@"), ("question marks", "?"),
        ("exclamation marks", "!")], [
        ("tildes", "~"), ("percent signs", "%"), ("carets", "^"),
        ("ampersands", "&")]),
    "small_int": _p("num", list(range(-20, 151)), list(range(151, 171))),
    "pos_int": _p("num", list(range(1, 81)), list(range(81, 101))),
    "float_val": _p("num", [round(0.1 * i, 1) for i in range(1, 61)],
                    [round(0.1 * i, 1) for i in range(61, 71)]),
    "numeric_text": _p("quote", [str(i) for i in range(1, 61)],
                       [str(i) for i in range(61, 71)]),
    # "bare": the suite writes py-045 as `opens data.txt for reading`, no
    # quotes -- the model has to add them itself in the answer.
    "file_name": _p("bare",
                    [b + "." + e for b in _FILE_STEMS for e in _FILE_EXTS],
                    [b + "." + e
                     for b in ("archive", "backup", "temp", "scratch",
                               "spool", "inbox")
                     for e in _FILE_EXTS]),
    "file_var": _p("name", [
        "f", "fh", "fp", "handle", "infile", "outfile", "stream", "src",
        "sink", "reader", "writer", "log_file", "data_file", "text_file"],
        ["fd", "chan", "spout", "pipe_"]),
    "err_var": _p("name", [
        "e", "err", "exc", "ex", "error", "problem", "issue", "bug", "oops",
        "trouble", "mishap", "fault2", "snag", "blip"],
        ["exn", "fault", "glitch", "hiccup"]),
    "type_name": _p("name", [
        "int", "str", "float", "list", "dict", "bool", "tuple", "set",
        "bytes", "complex"], ["frozenset", "bytearray"]),
    "set_var": _gen_ident_pool("name", _SET_ROOTS, _SET_ROOTS_DEV,
                               _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "set_var_b": _gen_ident_pool("name", _SET_B_ROOTS, _SET_B_ROOTS_DEV,
                                 _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "tuple_var": _gen_ident_pool("name", _TUPLE_ROOTS, _TUPLE_ROOTS_DEV,
                                 _SHORT_NAMES_TRAIN, _SHORT_NAMES_DEV),
    "module_name": _p("name", ["math", "random", "os", "sys", "json", "time",
                               "re"], ["string", "csv"]),
}


def _render_prompt(kind, value):
    if kind == "pair":
        return str(value[0])
    if kind in ("name", "bare"):
        return str(value)
    if kind == "quote":
        return '"%s"' % value
    return repr(value)


def _render_code(kind, value):
    if kind == "pair":
        return repr(value[1])
    if kind == "name":
        return str(value)
    return repr(value)


def answer_identifiers(code):
    """Every identifier an answer binds or uses -- lets tests prove the
    prompt's named variables actually reach the answer."""
    names = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[0])
    return names


# ---------------------------------------------------------------------------
# Wrappers. Train and dev wrappers are disjoint surface forms.
# ---------------------------------------------------------------------------

PY_WRAPPERS = [
    "Write a Python expression to {core}.",
    "Write a Python statement to {core}.",
    "Write Python to {core}.",
    "Write a Python one-liner to {core}.",
    "Give me a Python expression to {core}.",
    "Python expression to {core}:",
    "How do I {core} in Python?",
    "In Python, how do I {core}?",
    "What is the Python expression to {core}?",
    "What Python code would {core}?",
    "Show me a Python expression that will {core}.",
    "I need Python code to {core}.",
    "{core} in Python",
    "Python: {core}",
    "Write the Python that would {core}.",
    "Which Python expression would {core}?",
    "Python code to {core}",
    "{core}",
    "{core}?",
    "how do i {core}",
    "how do i {core}?",
    "how to {core}",
    "how to {core}?",
    "quick q, how do i {core}",
    "quick q: how do i {core}?",
    "python: {core}",
    "python - {core}",
    "python, {core}",
    "py: {core}",
    "can you {core}",
    "can you {core}?",
    "whats the code to {core}",
    "what's the code to {core}?",
    "need to {core}",
    "need help to {core}",
    "help me {core}",
    "help {core}",
    "trying to {core}",
    "{core} pls",
    "{core} please",
    "{core} thanks",
    "anyone know how to {core}",
    "so how do i {core}",
    "code to {core}?",
    "one liner to {core}?",
    "whats a quick way to {core}",
]

PY_DEV_WRAPPERS = [
    "In Python, {core}?",
    "{core}, in Python?",
    "Any Python expression to {core}?",
    "Remind me how to {core} in Python.",
    "quick, {core}?",
    "whats the way to {core}",
    "yo how do i {core}",
]

# Noun-phrase cores. Two sub-styles, picked from the core's first word:
# "the ..." takes "expression FOR {core}"; "a ..."/"an ..." takes the bare
# "Write {core}." shape, because "an expression for a Python function" is
# not English.
NP_THE_WRAPPERS = [
    "Write a Python expression for {core}.",
    "Python expression for {core}",
    "Give me the Python expression for {core}.",
    "What is the Python expression for {core}?",
    "Python: {core}",
    "{core} in Python",
    "I need the Python for {core}.",
    "Write Python for {core}.",
    "Show me the Python expression for {core}.",
    "Python code for {core}",
    "Which Python expression gives {core}?",
    "{core}",
    "{core}?",
    "whats {core}",
    "what's {core}?",
    "whats {core} in python",
    "quick q, whats {core}",
    "python: {core}",
    "py: {core}",
    "need {core}",
    "need {core} pls",
    "how do i get {core}",
    "how do i get {core}?",
    "code for {core}",
    "code for {core}?",
    "whats the code for {core}",
    "can someone give me {core}",
    "anyone have {core}",
    "{core}, python",
    "{core} - python",
    "{core} pls",
    "{core} please",
    "{core} thanks",
    "trying to find {core}",
    "looking for {core}",
    "whats {core} look like in python",
]

NP_THE_DEV_WRAPPERS = [
    "In Python, what is {core}?",
    "{core}, as a Python expression",
    "How would I write {core} in Python?",
    "whats {core} again",
    "{core}?? need it now",
]

NP_A_WRAPPERS = [
    "Write {core}.",
    "Give me {core}.",
    "Show me {core}.",
    "I need {core}.",
    "{core}, in Python",
    "Write {core} in Python.",
    "Give me {core} in Python.",
    "Python: {core}",
    "I need {core} in Python.",
    "Show me {core} in Python.",
    "{core}",
    "{core}?",
    "need {core}",
    "need {core} pls",
    "gimme {core}",
    "gimme {core}?",
    "whats {core}",
    "whats {core} look like",
    "python: {core}",
    "py: {core}",
    "quick q, {core}?",
    "can you give me {core}",
    "can someone write {core}",
    "{core} pls",
    "{core} please",
    "{core} thanks",
    "looking for {core}",
    "trying to get {core}",
    "how do i write {core}",
    "how do i write {core}?",
    "{core}, python",
    "{core} - python",
    "just need {core}",
    "{core} real quick",
    "anyone got {core}",
]

NP_A_DEV_WRAPPERS = [
    "How do I write {core} in Python?",
    "{core}, please.",
    "Can you give me {core}?",
    "hey can i get {core}",
    "{core} real fast",
]

VP3_WRAPPERS = [
    "Write a Python expression that {core}.",
    "Write a Python statement that {core}.",
    "Write Python that {core}.",
    "Give me a Python expression that {core}.",
    "Python expression that {core}",
    "I need a Python expression that {core}.",
    "Which Python expression {core}?",
    "What Python code {core}?",
    "Show me Python that {core}.",
    "Write a Python one-liner that {core}.",
    "Write the Python that {core}.",
    "whats the code that {core}",
    "what's the code that {core}?",
    "python: something that {core}",
    "py: something that {core}",
    "need something that {core}",
    "need code that {core}",
    "quick q, whats the code that {core}",
    "can you write something that {core}",
    "can someone write something that {core}",
    "looking for something that {core}",
    "trying to find something that {core}",
    "how do i get something that {core}",
    "how do i get something that {core}?",
    "something that {core}?",
    "something that {core}",
    "gimme something that {core}",
    "whats a one liner that {core}",
    "one liner that {core}?",
    "code that {core}, pls",
    "code that {core} please",
    "code that {core}",
    "want something that {core}",
    "anyone have something that {core}",
]

VP3_DEV_WRAPPERS = [
    "A Python expression that {core}, please.",
    "Python one-liner that {core}?",
    "Give me Python which {core}.",
    "something that {core}, quick?",
    "whats code that {core}",
]

# form key -> (train wrappers, dev wrappers). The key also goes into the
# template id, so ids stay unique across forms.
_FORM_WRAPPERS = {
    "i": (PY_WRAPPERS, PY_DEV_WRAPPERS),
    "n": (NP_THE_WRAPPERS, NP_THE_DEV_WRAPPERS),
    "a": (NP_A_WRAPPERS, NP_A_DEV_WRAPPERS),
    "v": (VP3_WRAPPERS, VP3_DEV_WRAPPERS),
}

BUG_WRAPPERS = [
    "Fix this Python line: {core}",
    "Fix the bug in this Python line: {core}",
    "Correct this Python line: {core}",
    "This Python line is broken, fix it: {core}",
    "Repair this line of Python: {core}",
    "Fix this line: {core}",
    "Fix: {core}",
    "Debug this Python line: {core}",
    "Rewrite this Python line so it works: {core}",
    "What is the fixed version of this Python line? {core}",
    "This line has a bug, give the corrected line: {core}",
    "Fix the error in this Python code: {core}",
    "Fix this line of Python code: {core}",
    "{core} <- syntax error",
    "{core} <- error",
    "{core} <- broken",
    "{core} <- whats wrong",
    "{core} <- fix?",
    "why does this crash: {core}",
    "why doesnt this work: {core}",
    "fix pls: {core}",
    "fix please: {core}",
    "{core} gives an error",
    "{core} throws an error",
    "{core} - error",
    "{core} is broken",
    "{core} doesnt work",
    "{core} wont run",
    "help, this line fails: {core}",
    "help this fails: {core}",
    "help pls: {core}",
    "my code has {core} in it and it breaks",
    "my code has a bug: {core}",
    "this line is broken: {core}",
    "whats wrong with this: {core}",
    "whats wrong with {core}",
    "quick q whats wrong with {core}",
    "can you fix {core}",
    "can you fix this: {core}",
    "python: {core}",
    "py: {core}",
    "{core} pls fix",
    "{core}?",
]

BUG_DEV_WRAPPERS = [
    "Broken Python line, fix it: {core}",
    "Please repair this Python line: {core}",
    "Corrected version of: {core}",
    "{core} <- can you fix",
    "broken: {core}",
    "{core} - please fix",
]


def form_key(core):
    """'i' | 'n' | 'a' | 'v' -- which wrapper family this core needs."""
    form, text = core_form(core)
    if form == "vp3":
        return "v"
    if form == "np":
        return "a" if text.split()[0].lower() in ("a", "an") else "n"
    return "i"


def _wrappers(recipe, split, core):
    if recipe["category"] == "bugfix":
        return BUG_DEV_WRAPPERS if split == "dev" else BUG_WRAPPERS
    return _FORM_WRAPPERS[form_key(core)][1 if split == "dev" else 0]


# Casual-noise transform: applied to the WRAPPER and CORE TEMPLATE TEXT only
# (before {slot} substitution), so it can never touch an identifier, literal,
# or bugfix {bug} line -- it only ever sees hand-authored English prose. This
# is what makes formal-looking training prompts also cover the messy real
# surface forms (lowercase, no "Python" mention, no trailing punctuation,
# apostrophe-less contractions) real users actually type.
_PYTHON_DROP = [
    "a Python expression to ", "a Python statement to ", "a Python one-liner to ",
    "the Python expression to ", "a Python expression for ", "the Python expression for ",
    "a Python expression that ", "a Python statement that ",
    "Python expression to ", "Python statement to ", "Python expression for ",
    "Python expression that ", "Python code to ", "Python code for ",
    "Python one-liner to ", "Python one-liner that ",
    "In Python, ", " in Python", " using Python", " via Python",
    "Python: ", "Python code ", "Python ",
]

_CONTRACTIONS = {
    "isn't": "isnt", "aren't": "arent", "wasn't": "wasnt", "weren't": "werent",
    "don't": "dont", "doesn't": "doesnt", "didn't": "didnt", "won't": "wont",
    "wouldn't": "wouldnt", "can't": "cant", "couldn't": "couldnt",
    "shouldn't": "shouldnt", "what's": "whats", "it's": "its", "that's": "thats",
    "there's": "theres", "let's": "lets",
}
_CONTRACTION_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)),
    re.I)


def _decontract(text):
    def repl(m):
        w = m.group(0)
        out = _CONTRACTIONS[w.lower()]
        return out.capitalize() if w[0].isupper() else out
    return _CONTRACTION_RE.sub(repl, text)


# R9 prose-robustness noise, additional to and independent of the R8 40%
# "noisy" gate above -- each op is its own bernoulli draw against the WHOLE
# corpus (not nested inside the 40%), so the realized overall share matches
# the ~15%/~5%/~15% the R9 spec asks for. Every op is guarded to skip any
# template token containing "{" or "}" -- a `{slot}` placeholder -- so word
# dropout, the typo and the lead-in/tail can never touch an identifier, a
# literal or a bugfix `{bug}` line; only hand-authored English prose is
# ever a candidate (see TestNoiseAndCasual.test_r9_noise_never_touches_code).
WORD_DROPOUT_FRACTION = 0.15
TYPO_FRACTION = 0.05
# _casual_wrap only counts as applied when it actually adds a lead-in or a
# tail (each drawn at 0.55 once this outer gate fires: P(at least one) =
# 1-0.45**2 = 0.7975), so the outer gate is raised above 0.15 to land the
# REALIZED share (rows that actually got a lead-in/tail) near the spec's
# ~15%: 0.19 * 0.7975 = ~15.1% (measured on the R9 build).
CASUAL_WRAP_FRACTION = 0.19

_DROP_WORDS = {"the", "a", "of", "to", "in", "from", "my", "this", "is"}
_LEAD_INS = ["hey", "ok so", "quick one:", "so", "yo"]
_TAILS = [" pls", " thx", "?", "??"]


def _word_dropout(text, rng, max_drops=2):
    """Drop 1-2 function words from template PROSE text. A token is only a
    candidate if it carries no "{"/"}" -- so a `{slot}` placeholder (even
    one whose slot name IS a drop word, e.g. `abs_diff`'s {a}) is never a
    candidate regardless of what it strips down to."""
    tokens = text.split(" ")
    idxs = [i for i, t in enumerate(tokens)
            if "{" not in t and "}" not in t
            and re.sub(r"[^\w']", "", t).lower() in _DROP_WORDS]
    if not idxs:
        return text
    rng.shuffle(idxs)
    drop = set(idxs[:rng.randint(1, min(max_drops, len(idxs)))])
    return " ".join(t for i, t in enumerate(tokens) if i not in drop)


def _typo_word(text, rng):
    """One adjacent-char swap or dropped letter in one prose word of
    length>=5 (checked on its letters only, so trailing punctuation like
    "again." still counts). Same "{"/"}" guard as _word_dropout."""
    tokens = text.split(" ")
    cand = [i for i, t in enumerate(tokens)
            if "{" not in t and "}" not in t
            and len(re.sub(r"[^A-Za-z]", "", t)) >= 5]
    if not cand:
        return text
    i = rng.choice(cand)
    chars = list(tokens[i])
    alpha_idxs = [j for j, ch in enumerate(chars) if ch.isalpha()]
    j = rng.choice(alpha_idxs[:-1])
    if chars[j + 1].isalpha() and rng.random() < 0.5:
        chars[j], chars[j + 1] = chars[j + 1], chars[j]
    else:
        del chars[j]
    tokens[i] = "".join(chars)
    return " ".join(tokens)


def _casual_wrap(prompt, rng, want_lead, want_tail):
    """Glue a casual lead-in and/or tail onto the fully rendered PROMPT
    (after slot substitution) -- pure string concatenation outside the
    substituted content, so it structurally cannot touch an identifier or
    literal."""
    if want_lead:
        lead = rng.choice(_LEAD_INS)
        prompt = lead + " " + prompt[:1].lower() + prompt[1:]
    if want_tail:
        prompt = prompt + rng.choice(_TAILS)
    return prompt


def _pick_noise_ops(rng):
    """Independent bernoulli draws for each op; called only for the ~40% of
    rows that get any noise at all (see `_render`)."""
    ops = set()
    if rng.random() < 0.6:
        ops.add("lower_all" if rng.random() < 0.6 else "lower_first")
    if rng.random() < 0.5:
        ops.add("no_python")
    if rng.random() < 0.35:
        ops.add("contractions")
    if rng.random() < 0.7:
        ops.add("strip_punct")
    return ops


def _noise_template(text, ops):
    """text is a wrapper or core TEMPLATE string, {slot} placeholders intact
    and un-substituted -- only literal English characters are touched."""
    if "lower_all" in ops:
        text = text.lower()
    elif "lower_first" in ops:
        text = text[:1].lower() + text[1:]
    if "no_python" in ops:
        for old in _PYTHON_DROP:
            text = text.replace(old, "")
    if "contractions" in ops:
        text = _decontract(text)
    return text


# ---------------------------------------------------------------------------
# LLM templates (R11): a second prompt source per family, hand-written by a
# different agent against corpus/llm_templates/FAMILIES.md. See _render's
# `llm_template` argument -- an LLM template is treated exactly like a core
# with an identity ("{core}") wrapper, so it rides the same noise/casual-wrap
# pipeline every other row does.
# ---------------------------------------------------------------------------

def _family_slot_names(recipe):
    """(required, allowed) exact {slot} names an LLM template for `recipe`
    may reference. `required` must each appear exactly once; `allowed` is
    the full set a template may draw from.

    bugfix: the only slot every existing core ever uses is {bug} -- the
    rendered broken line already embeds every other slot's value (see
    curriculum_v2_bugfix._wrap_ctx), so {bug} alone is required; the
    family's other slots are allowed but optional flavor.

    python with recipe["extra"]: mirrors `_validate` -- a family that
    supplies `extra` shows its DISPLAY names (e.g. range_from_zero's {k1} =
    k - 1) instead of the raw slot, so required comes from calling `extra`
    once, not from `slots`.

    python without extra: every slot is required, same as `_validate`
    enforces on hand-authored cores."""
    slots = set(recipe["slots"])
    if recipe["category"] == "bugfix":
        return {"bug"}, slots | {"bug"}
    if recipe["extra"] is None:
        return slots, slots
    sample = {s: POOLS[p]["train"][0] for s, p in recipe["slots"].items()}
    extra_keys = set(recipe["extra"](sample))
    return extra_keys, slots | extra_keys


def _template_fields(template):
    """Field names {like_this} referenced by `template`, or None if the
    braces are unbalanced/malformed. Rejects a template that contains a
    literal "{{" or "}}" too (FAMILIES.md's "no other braces" rule) -- those
    parse fine as escaped literals but are never what a template author
    means to write here."""
    if "{{" in template or "}}" in template:
        return None
    try:
        return [name for _, name, _, _ in _string.Formatter().parse(template)
                if name is not None]
    except ValueError:
        return None


def llm_template_ok(recipe, template):
    """True if `template` is safe to render for `recipe`: balanced braces,
    every required slot present exactly once, and no unknown slot."""
    fields = _template_fields(template)
    if fields is None:
        return False
    required, allowed = _family_slot_names(recipe)
    counts = {}
    for f in fields:
        if f not in allowed:
            return False
        counts[f] = counts.get(f, 0) + 1
    return all(counts.get(f) == 1 for f in required)


def _llm_split_bucket(template):
    """Deterministic train/dev split for one template line: a template's
    sha1 never changes, so it always lands in the same bucket across
    rebuilds and across the process re-reading its own file."""
    return ("dev" if hashlib.sha1(template.encode()).digest()[0] % 5 == 0
            else "train")


def _load_family_llm_templates(recipe):
    """(train_list, dev_list, rejected_count) for one family, read from
    corpus/llm_templates/<family>.txt. Missing file -> all empty, unchanged
    behavior. A rejected line is a warning, not a crash -- one bad line from
    another agent should not sink the whole build."""
    path = os.path.join(LLM_TEMPLATES_DIR, recipe["family"] + ".txt")
    train, dev, rejected = [], [], 0
    if not os.path.isfile(path):
        return train, dev, rejected
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not llm_template_ok(recipe, line):
                print(f"WARNING: corpus/llm_templates/{recipe['family']}.txt:"
                      f"{lineno}: rejected template (bad/missing/unknown "
                      f"slot): {line!r}", file=sys.stderr)
                rejected += 1
                continue
            (dev if _llm_split_bucket(line) == "dev" else train).append(line)
    return train, dev, rejected


# family -> {"train": [...], "dev": [...]}; populated by load_recipes().
LLM_TEMPLATES = {}
LLM_TEMPLATES_REJECTED = 0


def _load_llm_templates(recipes):
    global LLM_TEMPLATES, LLM_TEMPLATES_REJECTED
    LLM_TEMPLATES, LLM_TEMPLATES_REJECTED = {}, 0
    for recipe in recipes:
        train, dev, rejected = _load_family_llm_templates(recipe)
        LLM_TEMPLATES_REJECTED += rejected
        if train or dev:
            LLM_TEMPLATES[recipe["family"]] = {"train": train, "dev": dev}


# ---------------------------------------------------------------------------
# Recipe loading (lazy, so recipe modules can import R from here)
# ---------------------------------------------------------------------------

_RECIPES = None


def load_recipes():
    """Import the recipe modules, merge their EXTRA_POOLS into POOLS, and
    validate the whole set. Cached. A missing/empty recipe module is not an
    error -- the two modules land independently."""
    global _RECIPES
    if _RECIPES is not None:
        return _RECIPES
    recipes = []
    for mod_name in ("corpus.curriculum_v2_python", "corpus.curriculum_v2_bugfix"):
        try:
            mod = __import__(mod_name, fromlist=["RECIPES"])
        except ImportError:
            continue
        for name, pool in getattr(mod, "EXTRA_POOLS", {}).items():
            if name in POOLS:
                raise SystemExit(f"{mod_name}: duplicate pool name {name!r}")
            POOLS[name] = pool
        recipes.extend(getattr(mod, "RECIPES", []))
    _validate(recipes)
    _load_llm_templates(recipes)
    _RECIPES = recipes
    return recipes


def _validate(recipes):
    seen = set()
    for r in recipes:
        if r["family"] in seen:
            raise SystemExit(f"duplicate family {r['family']!r}")
        seen.add(r["family"])
        assert r["category"] in ("python", "bugfix"), r["family"]
        assert r["tier"] in (1, 2, 3), r["family"]
        need = 3 if r["category"] == "bugfix" else 5
        assert len(r["cores"]) >= need, (r["family"], "too few cores")
        assert len(r["dev_cores"]) >= 1, (r["family"], "no dev core")
        assert not (set(r["cores"]) & set(r["dev_cores"])), r["family"]
        for slot, pool in r["slots"].items():
            if pool not in POOLS:
                raise SystemExit(f"{r['family']}: unknown pool {pool!r}")
        if r["category"] == "python":
            # R4 lost python because the model only ever saw imperative
            # prompts; the suite asks two thirds of its items as noun
            # phrases and third-person verb phrases.
            for split_cores, floor in ((r["cores"], 4), (r["dev_cores"], 1)):
                counts = {}
                for core in split_cores:
                    f = core_form(core)[0]
                    counts[f] = counts.get(f, 0) + 1
                for f in ("imp", "np", "vp3"):
                    assert counts.get(f, 0) >= floor, (
                        r["family"], f, counts, "too few cores of this form")
        for core in r["cores"] + r["dev_cores"]:
            if r["category"] == "bugfix":
                assert "{bug}" in core, (r["family"], core)
            elif r["extra"] is None:
                for slot in r["slots"]:
                    assert ("{%s}" % slot) in core, (r["family"], slot, core)
        if r["category"] == "bugfix":
            assert r["bug"] is not None, (r["family"], "bugfix needs bug()")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _capacity(recipe, split):
    n = 1
    for slot, pool in recipe["slots"].items():
        n *= len(POOLS[pool][split])
    return n


def _distinct_names_ok(recipe, combo):
    """Two name-kind slots in one record must never draw the same identifier
    (`def f(x, x)`, `[x for x in x]`). No recipe wants them equal."""
    vals = [v for slot, v in combo.items()
            if POOLS[recipe["slots"][slot]]["kind"] == "name"]
    return len(vals) == len(set(vals))


def _sample_combos(recipe, split, want, rng):
    """`want` parameter tuples, each a distinct point of the pool product
    while the product lasts (that is the anti-lookup-table property: one
    training row per binding, not 27 rephrasings of one binding)."""
    keys = sorted(recipe["slots"])
    pools = [POOLS[recipe["slots"][k]][split] for k in keys]
    cap = 1
    for pool in pools:
        cap *= len(pool)
    guard = recipe["guard"]
    out, used = [], set()
    for _pass in range(64):
        if len(out) >= want:
            break
        n = min(cap, max(2 * (want - len(out)), 64))
        for idx in rng.sample(range(cap), n):
            if idx in used:
                continue
            used.add(idx)
            combo, rest = {}, idx
            for k, pool in zip(keys, pools):
                combo[k] = pool[rest % len(pool)]
                rest //= len(pool)
            if not _distinct_names_ok(recipe, combo):
                continue
            if guard and not guard(combo):
                continue
            out.append(combo)
            if len(out) >= want:
                break
        if len(used) >= cap:      # product exhausted; start a repeat pass
            used.clear()
            if not out:
                break
    return out[:want]


def _pick_str_literal(rng, split):
    """A quoted string literal, single or double quote chosen at random --
    the SAME text is used for the prompt and the code, so whichever quote
    style lands in the prompt is guaranteed to be the answer's quote style
    too. Reuses word_lit's plain (unquoted) word list; none of those words
    contain a quote character, so no escaping is ever needed."""
    w = rng.choice(POOLS["word_lit"][split])
    q = "'" if rng.random() < 0.5 else '"'
    return q + w + q


def _pick_list_literal(rng, split):
    """A small (3-5 item) integer list literal, e.g. "[3, 1, 4]" -- again the
    same text for prompt and code."""
    pool = POOLS["small_int"][split]
    vals = [rng.choice(pool) for _ in range(rng.randint(3, 5))]
    return "[" + ", ".join(str(v) for v in vals) + "]"


def _render(recipe, combo, wrapper, core, template_id, split, seq, stats,
            rng=None, llm_template=None):
    """`llm_template`, when given, replaces the (wrapper, core) pair with a
    single flat casual template string (see corpus/llm_templates/) rendered
    against the SAME `combo`/`shown`/answer as any other row -- `wrapper`
    and `core` are ignored in that case. It rides the wrapper/core pipeline
    below unchanged by pretending to be a core under an identity wrapper, so
    it gets the exact same noise/casual-wrap treatment as every other row."""
    kinds = {s: POOLS[p]["kind"] for s, p in recipe["slots"].items()}
    code = {s: _render_code(kinds[s], v) for s, v in combo.items()}
    shown = {s: _render_prompt(kinds[s], v) for s, v in combo.items()}
    lit_slot = recipe.get("literal_slot")
    if (lit_slot and rng is not None
            and rng.random() < recipe.get("literal_share", 0.30)):
        lit_text = (_pick_list_literal(rng, split) if recipe["literal_kind"] == "list"
                    else _pick_str_literal(rng, split))
        code[lit_slot] = lit_text
        shown[lit_slot] = lit_text
        stats["literal_operand_rows"] += 1
    answer = recipe["answer"](code)
    quote_style = None
    if recipe["category"] == "bugfix":
        bug_line = recipe["bug"](code)
        quote_style = "single"
        # Every training bug line is rendered with repr() so every quote in
        # it is single -- the model never saw the double-quoted shapes the
        # frozen suite actually uses (`d = {"a": 1`, `int("abc")`). Flip
        # these to double quotes, consistently across bug line and answer,
        # skipping the rare case where a value's repr already needed a
        # double quote (an apostrophe inside the string). Only ~49% of
        # bugfix rows carry a quoted literal at all (most bug classes --
        # missing colon, = vs ==, xrange, ...  -- have none to flip), so a
        # flat 0.5 draw on that subset lands near 25% double overall; use
        # 0.65 so the realized double-quote share on the ~49%-eligible rows
        # clears the 30%-of-all-bugfix-rows floor with margin.
        if (rng is not None and '"' not in bug_line and '"' not in answer
                and "'" in bug_line and "'" in answer
                and rng.random() < 0.65):
            bug_line = bug_line.replace("'", '"')
            answer = answer.replace("'", '"')
            quote_style = "double"
        shown = dict(shown, bug=bug_line)
        carrier = bug_line
    else:
        bug_line = None
        carrier = answer
    if llm_template is not None:
        core_text, wrapper_text = llm_template, "{core}"
    else:
        core_text = core_form(core)[1]
        wrapper_text = wrapper
    noisy = rng is not None and rng.random() < 0.40
    strip_punct = False
    if noisy:
        ops = _pick_noise_ops(rng)
        wrapper_text = _noise_template(wrapper_text, ops)
        core_text = _noise_template(core_text, ops)
        strip_punct = "strip_punct" in ops
        stats["noisy_rows"] += 1
    # R9 prose-robustness noise -- independent bernoulli draws against the
    # WHOLE corpus (not nested inside the 40% `noisy` gate above), each
    # guarded to only ever touch hand-authored English tokens (see
    # _word_dropout/_typo_word docstrings).
    if rng is not None and rng.random() < WORD_DROPOUT_FRACTION:
        if rng.random() < 0.5:
            wrapper_text = _word_dropout(wrapper_text, rng)
        else:
            core_text = _word_dropout(core_text, rng)
        stats["word_dropout_rows"] += 1
    if rng is not None and rng.random() < TYPO_FRACTION:
        if rng.random() < 0.5:
            wrapper_text = _typo_word(wrapper_text, rng)
        else:
            core_text = _typo_word(core_text, rng)
        stats["typo_rows"] += 1
    if recipe["extra"] is not None:
        shown = dict(shown, **recipe["extra"](combo))
    prompt = wrapper_text.format(core=core_text.format(**shown))
    if strip_punct and prompt[-1:] in ".?":
        prompt = prompt[:-1]
    if rng is not None and rng.random() < CASUAL_WRAP_FRACTION:
        want_lead = rng.random() < 0.55
        want_tail = rng.random() < 0.55
        if want_lead or want_tail:
            prompt = _casual_wrap(prompt, rng, want_lead, want_tail)
            stats["casual_wrap_rows"] += 1
    # The entire point of v2: every identifier/literal named in the prompt
    # must be copied verbatim into the answer. A recipe that violates this
    # is a bug, not a droppable row. Bugfix families are exempt: their
    # slots may live in the surrounding problem text rather than the fixed
    # line, and their buggy line may deliberately mangle quoting -- there
    # the invariant is only that the slot reached the prompt at all.
    strip = str.maketrans("", "", "\"'")
    hay = (carrier if bug_line is None else prompt).translate(strip)
    for slot, value in code.items():
        if slot in recipe["answer_only_slots"]:
            continue
        assert value.translate(strip) in hay, (
            recipe["family"], slot, value, carrier)
    if not _prompt_ok(prompt):
        stats["drop_prompt_shape"] += 1
        return None
    if "\n" in answer:
        stats["drop_multiline_answer"] += 1
        return None
    if recipe["parse"] and not _parses(answer):
        stats["syntax_fail"] += 1
        return None
    tag = "tr" if split == "train" else "dev"
    rec = {"id": f"cv2-{tag}-{recipe['family']}-{seq:05d}",
           "prompt": prompt, "answer": answer,
           "category": recipe["category"], "tier": recipe["tier"],
           "source": SOURCE, "task_family": recipe["family"],
           "template_id": template_id, "_parse": recipe["parse"]}
    if bug_line is not None:
        rec["_bug"] = bug_line
        rec["_quote_style"] = quote_style
    return rec


def _prompt_ok(prompt):
    return (prompt.isascii() and "\n" not in prompt and "\\" not in prompt
            and all(32 <= ord(c) < 127 for c in prompt)
            and len(prompt.encode("utf-8")) <= PROMPT_MAX_BYTES)


def _parses(answer):
    """score.py:py_parses -- a header line ending in ':' gets a synthesized
    `pass` body, nothing else is synthesized."""
    for cand in (answer, answer + " pass"):
        try:
            ast.parse(cand)
            return True
        except (SyntaxError, ValueError):
            pass
    return False


def build_split(recipes, split, stats):
    records = []
    for recipe in recipes:
        cores = recipe["dev_cores"] if split == "dev" else recipe["cores"]
        wsets = [_wrappers(recipe, split, c) for c in cores]
        fam_stats = stats["families"].setdefault(
            recipe["family"], {"train_candidates": 0, "dev_candidates": 0})
        llm_list = LLM_TEMPLATES.get(recipe["family"], {}).get(split, [])
        seen, seq = set(), 0
        if split == "train":
            rng = random.Random(f"{recipe['family']}:train:v2")
            # explicit rows= is a target, not a ceiling: repeat bindings with
            # new templates past pool capacity (the sampler already does)
            want = recipe.get("rows") or min(
                TRAIN_PER_FAMILY, max(MIN_FAMILY_ROWS, _capacity(recipe, "train")))
            if ROW_CAP:              # tests shrink the corpus uniformly
                want = min(want, ROW_CAP)
            combos = _sample_combos(recipe, "train", min(want, MAX_ATTEMPTS), rng)
            bare_cis = ([i for i, w in enumerate(wsets) if "{core}" in w]
                        if recipe["category"] == "python" else [])
            plan = []
            for c in combos:
                if llm_list and rng.random() < P_LLM:
                    plan.append((None, None, c, rng.randrange(len(llm_list))))
                    continue
                if bare_cis and rng.random() < BARE_FRACTION:
                    ci = rng.choice(bare_cis)
                    wi = wsets[ci].index("{core}")
                elif recipe["category"] == "bugfix" and rng.random() < 0.30:
                    # Keep the broken line prominent without discarding the
                    # contextual hints: copy/preservation must also work when
                    # the user supplies only code and a brief fix request.
                    ci = cores.index("{bug}") if "{bug}" in cores else rng.randrange(len(cores))
                    wi = rng.randrange(len(wsets[ci]))
                else:
                    ci = rng.randrange(len(cores))
                    wi = rng.randrange(len(wsets[ci]))
                plan.append((wi, ci, c, None))
        else:
            rng = random.Random(f"{recipe['family']}:dev:v2")
            plan = []
            for ci in range(len(cores)):
                for wi in range(len(wsets[ci])):
                    for combo in _sample_combos(recipe, "dev", DEV_TUPLES, rng):
                        if llm_list and rng.random() < P_LLM:
                            plan.append((None, None, combo,
                                        rng.randrange(len(llm_list))))
                        else:
                            plan.append((wi, ci, combo, None))
        for wi, ci, combo, llm_idx in plan:
            if llm_idx is not None:
                tid = (f"{recipe['family']}/llm{llm_idx}" if split == "train"
                       else f"{recipe['family']}/dev-llm{llm_idx}")
                rec = _render(recipe, combo, None, None, tid, split, seq,
                              stats, rng, llm_template=llm_list[llm_idx])
                stats["llm_template_rows"] += 1
            else:
                fk = (form_key(cores[ci]) if recipe["category"] == "python"
                      else "b")
                tid = (f"{recipe['family']}/{fk}w{wi}c{ci}" if split == "train"
                       else f"{recipe['family']}/dev-{fk}w{wi}c{ci}")
                rec = _render(recipe, combo, wsets[ci][wi], cores[ci], tid,
                              split, seq, stats, rng)
            if rec is None:
                continue
            key = (rec["prompt"], rec["answer"])
            if key in seen:
                stats["drop_duplicate_pair"] += 1
                continue
            seen.add(key)
            seq += 1
            records.append(rec)
        fam_stats[f"{split}_candidates"] = seq
    return records


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def assert_writable(out_dir):
    real = os.path.realpath(out_dir)
    if real == FORBIDDEN_DIR or real.startswith(FORBIDDEN_DIR + os.sep):
        raise SystemExit(f"refusing to write into {FORBIDDEN_DIR}")
    if os.path.realpath(HERE) == real:
        raise SystemExit("refusing to write into corpus/ itself")
    existing = [n for n in (TRAIN_FILE, DEV_FILE, MANIFEST_FILE)
                if os.path.exists(os.path.join(real, n))]
    if existing:
        raise SystemExit(f"refusing to overwrite existing outputs in "
                         f"{out_dir}: {', '.join(existing)}")


def _new_stats():
    return {"drop_prompt_shape": 0, "syntax_fail": 0,
            "drop_multiline_answer": 0, "drop_duplicate_pair": 0,
            "drop_train_dev_prompt": 0, "drop_train_dev_pair": 0,
            "noisy_rows": 0, "word_dropout_rows": 0, "typo_rows": 0,
            "casual_wrap_rows": 0, "literal_operand_rows": 0,
            "llm_template_rows": 0,
            "families": {}}


def build(recipes=None, extra_eval=None):
    """Both filtered splits + stats, without touching the filesystem."""
    recipes = recipes if recipes is not None else load_recipes()
    stats = _new_stats()
    train_raw = build_split(recipes, "train", stats)
    dev_raw = build_split(recipes, "dev", stats)

    tr_ids = {r["template_id"] for r in train_raw}
    dv_ids = {r["template_id"] for r in dev_raw}
    if tr_ids & dv_ids:
        raise SystemExit(f"template id leak: {sorted(tr_ids & dv_ids)[:5]}")

    dev_kept, dev_filter = op.filter_pairs(dev_raw, extra_eval)
    train_kept, train_filter = op.filter_pairs(train_raw, extra_eval)
    stats["dev_filter"] = dev_filter
    stats["train_filter"] = train_filter

    dev_prompts = {pp.normalize_key(r["prompt"]) for r in dev_kept}
    dev_pairs = {(pp.normalize_key(r["prompt"]), pp.normalize_key(r["answer"]))
                 for r in dev_kept}
    kept = []
    for rec in train_kept:
        pn = pp.normalize_key(rec["prompt"])
        if pn in dev_prompts:
            stats["drop_train_dev_prompt"] += 1
            continue
        if (pn, pp.normalize_key(rec["answer"])) in dev_pairs:
            stats["drop_train_dev_pair"] += 1
            continue
        kept.append(rec)
    return kept, dev_kept, stats


def run(out_dir, seed=DEFAULT_SEED, recipes=None, extra_eval=None):
    assert_writable(out_dir)
    recipes = recipes if recipes is not None else load_recipes()
    train, dev, stats = build(recipes, extra_eval)
    os.makedirs(out_dir, exist_ok=True)

    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(dev)

    train_path = os.path.join(out_dir, TRAIN_FILE)
    dev_path = os.path.join(out_dir, DEV_FILE)
    with open(train_path, "w") as f:
        for rec in train:
            f.write(json.dumps({k: v for k, v in rec.items()
                                if not k.startswith("_")},
                               sort_keys=True) + "\n")
    with open(dev_path, "w") as f:
        for rec in dev:
            first = " ".join(rec["answer"].splitlines()[0].split())
            f.write(json.dumps({
                "id": rec["id"], "category": rec["category"],
                "prompt": rec["prompt"], "answer": rec["answer"],
                "accept": ["^" + re.escape(first) + "$"],
                "scoring": "python-ast" if rec["_parse"] else "regex",
                "max_new_tokens": 64, "tier": rec["tier"],
                "source": rec["source"], "task_family": rec["task_family"],
                "template_id": rec["template_id"],
            }, sort_keys=True) + "\n")

    fams = stats["families"]
    for rec in train:
        f_ = fams.setdefault(rec["task_family"], {})
        f_["train_count"] = f_.get("train_count", 0) + 1
        f_.setdefault("train_answers", set()).add(rec["answer"])
        if rec["template_id"].rsplit("/", 1)[-1].startswith("llm"):
            f_["llm_train_rows"] = f_.get("llm_train_rows", 0) + 1
    for rec in dev:
        f_ = fams.setdefault(rec["task_family"], {})
        f_["dev_count"] = f_.get("dev_count", 0) + 1
        if rec["template_id"].rsplit("/", 1)[-1].startswith("dev-llm"):
            f_["llm_dev_rows"] = f_.get("llm_dev_rows", 0) + 1
    for recipe in recipes:
        f_ = fams.setdefault(recipe["family"], {})
        f_.setdefault("train_count", 0)
        f_.setdefault("dev_count", 0)
        f_.setdefault("llm_train_rows", 0)
        f_.setdefault("llm_dev_rows", 0)
        llm = LLM_TEMPLATES.get(recipe["family"], {})
        f_.update(category=recipe["category"], tier=recipe["tier"],
                  suite_ids=recipe["suite_ids"],
                  cores=len(recipe["cores"]),
                  core_forms={f: sum(core_form(c)[0] == f
                                     for c in recipe["cores"])
                              for f in ("imp", "np", "vp3")},
                  dev_cores=len(recipe["dev_cores"]),
                  rows_cap=recipe["rows"] or TRAIN_PER_FAMILY,
                  slots=recipe["slots"],
                  llm_templates_loaded={"train": len(llm.get("train", [])),
                                        "dev": len(llm.get("dev", []))})
    for name, f_ in fams.items():
        answers = f_.pop("train_answers", set())
        f_["distinct_train_answers"] = len(answers)
        cand = f_.get("train_candidates", 0)
        f_["filter_loss"] = round(1 - f_["train_count"] / cand, 4) if cand else 0.0

    manifest = {
        "label": "EXPERIMENTAL R4 curriculum v2 -- not a capability claim",
        "source": SOURCE, "seed": seed,
        "train_file": TRAIN_FILE, "dev_file": DEV_FILE,
        "train_count": len(train), "dev_count": len(dev),
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "distinct_train_answers": len({r["answer"] for r in train}),
        "distinct_train_prompts": len({r["prompt"] for r in train}),
        "distinct_dev_answers": len({r["answer"] for r in dev}),
        "family_count": len(recipes),
        "llm_templates_rejected": LLM_TEMPLATES_REJECTED,
        "families": {k: {kk: (sorted(vv) if isinstance(vv, set) else vv)
                         for kk, vv in v.items()} for k, v in sorted(fams.items())},
        "extra_eval": extra_eval or [],
        "train_per_family": TRAIN_PER_FAMILY,
        "min_family_rows": MIN_FAMILY_ROWS,
        "dev_tuples_per_template": DEV_TUPLES,
        "wrappers": {"imperative_train": PY_WRAPPERS,
                     "imperative_dev": PY_DEV_WRAPPERS,
                     "np_the_train": NP_THE_WRAPPERS,
                     "np_the_dev": NP_THE_DEV_WRAPPERS,
                     "np_a_train": NP_A_WRAPPERS, "np_a_dev": NP_A_DEV_WRAPPERS,
                     "vp3_train": VP3_WRAPPERS, "vp3_dev": VP3_DEV_WRAPPERS,
                     "bugfix_train": BUG_WRAPPERS, "bugfix_dev": BUG_DEV_WRAPPERS},
        "drop_stats": {k: v for k, v in stats.items() if k != "families"},
        "notes": [
            "contamination filtered through offline_python.filter_pairs -> "
            "postprocess.load_eval_overlap_sets (sha256-pinned suite-v1.1); "
            "thresholds unchanged",
            "one distinct parameter tuple per train row while the pool "
            "product lasts -- the R3 lookup-table failure mode is a "
            "distinct-answer/row ratio, and it is in this manifest",
            "answers are parse-validated only (ast.parse, plus score.py's "
            "'header + pass' allowance); nothing is ever executed",
        ],
    }
    with open(os.path.join(out_dir, MANIFEST_FILE), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return manifest


# ---------------------------------------------------------------------------
# --check-suite: does every targeted suite item actually pass, when the
# family's answer is instantiated with the suite item's own bindings?
# ---------------------------------------------------------------------------

def check_suite(categories=("python", "bugfix"), verbose=True):
    import score
    actual = sha256_file(pp.EVAL_SUITE)
    if actual != pp.EVAL_SUITE_SHA256:
        raise SystemExit(f"suite sha256 mismatch: {actual}")
    items = score.load_suite(pp.EVAL_SUITE)
    recipes = load_recipes()
    by_id = {}
    for r in recipes:
        for sid in r["suite_ids"]:
            by_id.setdefault(sid, []).append(r)

    rows, fails = [], 0
    for item in items:
        if item["category"] not in categories:
            continue
        fams = by_id.get(item["id"], [])
        if not fams:
            rows.append((item["id"], "-", "", "FAIL uncovered"))
            fails += 1
            continue
        for r in fams:
            binding = r["suite_bindings"].get(item["id"])
            if binding is None or set(binding) != set(r["slots"]):
                rows.append((item["id"], r["family"], "",
                             "FAIL missing/partial suite_bindings"))
                fails += 1
                continue
            code = {s: _render_code(POOLS[r["slots"][s]]["kind"], v)
                    for s, v in binding.items()}
            answer = r["answer"](code)
            line = score.first_line(answer, item["prompt"])
            ok = bool(score.correct(item, line))
            note = "OK" if ok else "FAIL not accepted"
            if ok and len(line) > score.char_budget(item):
                ok, note = False, "FAIL over char budget"
            if ok and r["category"] == "bugfix":
                bug_line = score.first_line(r["bug"](code), item["prompt"])
                if score.correct(item, bug_line):
                    ok, note = False, "FAIL buggy line already accepted"
            fails += not ok
            rows.append((item["id"], r["family"], line, note))

    if verbose:
        print("%-8s %-28s %-46s %s" % ("id", "family", "answer", "result"))
        for r in rows:
            print("%-8s %-28s %-46s %s" % r)
        print("%d row(s), %d failure(s)" % (len(rows), fails))
    return fails, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--check-suite", action="store_true")
    ap.add_argument("--extra-eval", action="append", default=None,
                    help="additional eval-schema jsonl (id/prompt/answer per "
                         "line) to contamination-filter against on top of "
                         "the frozen sha-pinned suite; repeatable")
    args = ap.parse_args()

    if args.check_suite:
        fails, _ = check_suite()
        return 1 if fails else 0
    if not args.out_dir:
        ap.error("--out-dir required unless --check-suite")

    m = run(args.out_dir, args.seed, extra_eval=args.extra_eval)
    print(f"{m['train_count']} train + {m['dev_count']} dev rows, "
          f"{m['family_count']} families -> {args.out_dir}")
    print(f"distinct train answers: {m['distinct_train_answers']} "
          f"({100.0 * m['distinct_train_answers'] / max(1, m['train_count']):.1f}%)")
    lossy = sorted((f["filter_loss"], name) for name, f in m["families"].items()
                   if f.get("filter_loss", 0) > 0.30)
    if lossy:
        print("families losing >30%% to the contamination filter: "
              + ", ".join(f"{n} ({100 * l:.0f}%)" for l, n in lossy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
