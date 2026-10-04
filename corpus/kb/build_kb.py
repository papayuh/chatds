#!/usr/bin/env python3
"""Build the KB3 knowledge index from the Simple English Wikipedia dump. See FORMAT.md."""
import bz2, collections, hashlib, html, json, os, re, struct, urllib.request, xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
MAX_TEXT, MAX_KEY, MIN_TEXT = 240, 63, 40
CTX_MAX, CTX_MIN_CLAUSE = 100, 30  # KB3 record = one context sentence of <= CTX_MAX chars
FZ_MIN_WORD, FZ_MAX_WORD = 5, 20  # words that get a fuzzy (delete-1) index entry
SECTOR = 512

from kb import ascii_only, normalize, fnv1a, variants

# ---- wikitext -> lead text ----
_TPL = re.compile(r"\{\{[^{}]*\}\}")
_FILE = re.compile(r"\[\[\s*(?:File|Image)\s*:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]", re.I)
_LINK2 = re.compile(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]")
_LINK1 = re.compile(r"\[\[([^\[\]]*)\]\]")
_PAREN = re.compile(r"\s*\([^()]*\)")  # every parenthetical: dates and pronunciations eat the 80-char context budget
_SKIP_START = "=*#:;|!{"
_ABBR = {"mrs", "prof", "dr", "st", "jr", "sr", "vs", "inc", "ltd", "etc"}

def strip_markup(w):
    w = re.sub(r"<!--.*?-->", "", w, flags=re.S)
    w = re.sub(r"<ref[^>/]*/>|<ref\b[^>]*>.*?</ref>", "", w, flags=re.S | re.I)
    w = re.sub(r"<(math|gallery|timeline|imagemap)\b.*?</\1>", "", w, flags=re.S | re.I)
    prev = None
    while prev != w:                       # nested templates, innermost first
        prev, w = w, _TPL.sub("", w)
    w = re.sub(r"\{\|.*?\|\}", "", w, flags=re.S)
    prev = None
    while prev != w:
        prev, w = w, _FILE.sub("", w)
    return w

def clean_line(l):
    prev = None
    while prev != l:
        prev = l
        l = _LINK2.sub(r"\1", l)
        l = _LINK1.sub(r"\1", l)
    l = re.sub(r"\[https?://[^\s\]]+\s+([^\]]*)\]|\[https?://[^\s\]]*\]", r"\1", l)
    l = re.sub(r"'{2,}", "", l)
    l = re.sub(r"<[^>]*>", "", l)
    l = ascii_only(html.unescape(l))
    l = re.sub(r"\(\s*(?:(?:or|and)\s*)?[;,:]+\s*", "(", l)  # template residue "(or; x)" -> "(x)"
    prev = None
    while prev != l:
        prev, l = l, _PAREN.sub("", l)
    l = re.sub(r"\s+", " ", l).strip()
    l = re.sub(r"\(\s+", "(", l)
    l = re.sub(r"\s+([,.;:)])", r"\1", l)
    return l

def sentences(t):
    out, start = [], 0
    for m in re.finditer(r"[.!?]\s+(?=[A-Z\"0-9])", t):
        w = re.search(r"(\w*)$", t[start:m.start()]).group(1)
        if len(w) < 3 or w.lower() in _ABBR:
            continue
        out.append(t[start:m.start() + 1])
        start = m.end()
    out.append(t[start:])
    return [s for s in out if s]

def cut_words(t, n):
    if len(t) <= n:
        return t
    t = t[:n + 1]
    return t[:t.rfind(" ")].rstrip(" ,;:") if " " in t else t[:n]

def extract_lead(wikitext):
    for line in strip_markup(wikitext).split("\n"):
        line = line.strip()
        if not line or line[0] in _SKIP_START or line.lower().startswith(("[[category", "__")):
            continue
        t = clean_line(line)
        if not t:
            continue
        ss = sentences(t)
        out = ss[0]
        if len(out) > MAX_TEXT:
            return cut_words(out, MAX_TEXT)
        if len(ss) > 1 and len(out) + 1 + len(ss[1]) <= MAX_TEXT:
            out += " " + ss[1]
        return out
    return ""

def extract_text(wikitext):
    """Full article text: every prose line through the same cleaner, one paragraph per line."""
    out = []
    for line in strip_markup(wikitext).split("\n"):
        line = line.strip()
        if not line or line[0] in _SKIP_START or line.lower().startswith(("[[category", "__")):
            continue
        t = clean_line(line)
        if t:
            out.append(t)
    return "\n\n".join(out)

# function words and prepositions: a context must not end on one
_FUNC = set("""a an the of in on at by for from to with as into onto over under between among about after before during
through since until upon within without against toward towards and or but nor so yet that which who whom whose where when while
than is are was were be been being has have had its his her their this these those not also both such other any some many
most more much very can could may might will would shall should do does did""".split())

def _tidy(s):
    """drop trailing function words and punctuation, end with a period"""
    w = s.rstrip(" ,;:-").split(" ")
    while len(w) > 1 and w[-1].lower().rstrip(",;:") in _FUNC:
        w.pop()
    s = " ".join(w).rstrip(" ,;:-")
    s = re.sub(r" to [a-z]+$", "", s)  # "established in April 2010 to produce" -> a dangling infinitive
    return s if s.endswith(".") else s + "."

def _ok(s):
    return len(s) >= CTX_MIN_CLAUSE and s.count('"') % 2 == 0 and s.count("(") == s.count(")")

def context_sentence(text):
    """KB3 context: the first sentence if it fits CTX_MAX; else the longest prefix ending at a clause boundary
    (`, ; :` or before ' and / which / who / that ') of >= CTX_MIN_CLAUSE chars; else the words back to the last
    non-function word. Always ends with a period, never in the middle of a phrase."""
    s = sentences(text)[0].strip()
    if len(s) <= CTX_MAX:
        return s
    head = s[:CTX_MAX - 1]  # room for the period
    if s[CTX_MAX - 1] != " ":  # the cut falls inside a word: drop it
        head = head[:head.rfind(" ")] if " " in head else head
    cuts = [m.start() for m in re.finditer(r"[,;:](?!\d)", head)] + \
           [m.start() for m in re.finditer(r" (?:and|which|who|that) ", s[:CTX_MAX + 7]) if m.start() <= CTX_MAX - 1 and not re.search(r"\b(?:between|both)\b", s[:m.start()])]
    for c in sorted(cuts, reverse=True):
        t = _tidy(s[:c])
        if _ok(t):
            return t
    return _tidy(head)

_DAB = re.compile(r"\{\{\s*(disambig|disambiguation|dab|geodis|hndis|surname|given name)", re.I)

# ---- dump parsing ----
def open_dump(p):
    return bz2.open(p, "rb") if p.endswith(".bz2") else open(p, "rb")

def parse(path, extract=extract_lead, min_text=MIN_TEXT):
    """returns (articles {title: text}, redirects {title: target})"""
    arts, reds = {}, {}
    with open_dump(path) as f:
        for _, e in ET.iterparse(f):
            if not e.tag.endswith("}page") and e.tag != "page":
                continue
            g = {c.tag.rsplit("}", 1)[-1]: c for c in e}
            if (g["ns"].text or "") == "0":
                title = g["title"].text or ""
                if "redirect" in g:
                    tgt = g["redirect"].get("title", "").split("#")[0]
                    if tgt:
                        reds[title] = tgt
                else:
                    rev = g.get("revision")
                    tx = rev.find("{*}text") if rev is not None else None
                    wt = (tx.text or "") if tx is not None else ""
                    if not (title.lower().startswith("list of ") or title.endswith("(disambiguation)")
                            or _DAB.search(wt)):
                        t = extract(wt)
                        if len(t) >= min_text:
                            arts[title] = t
            e.clear()
    return arts, reds

# ---- writer ----
def pad(n, a=SECTOR):
    return -n % a

def build(dump, out, source, lic="CC BY-SA 4.0"):
    arts, reds = parse(dump)
    keymap, prio = {}, {}  # key -> text, key -> priority of the entry that supplied it
    def add(key, text, p):
        # collisions ("George Washington" vs the city "George, Washington"): plain titles beat titles with
        # punctuation, which beat redirects; ties keep the longer text
        if 0 < len(key) <= MAX_KEY and (key not in keymap or (p, -len(text)) < (prio[key], -len(keymap[key]))):
            keymap[key], prio[key] = text, p
    for t, x in arts.items():
        add(normalize(t), x, 0 if not re.search(r"[,(]", t) else 1)
    for t, tgt in reds.items():
        if tgt in arts:
            add(normalize(t), arts[tgt], 2)
    # records: one context sentence per distinct article text, stored once
    ctx = {x: context_sentence(x) for x in set(keymap.values())}
    keymap = {k: ctx[x] for k, x in keymap.items()}
    toff, tarea = {}, bytearray()
    for x in keymap.values():
        if x not in toff:
            toff[x] = len(tarea)
            tarea += bytes([len(x)]) + x.encode("ascii")
    nk = len(keymap)
    nb = 1
    while nb * 4 < nk:
        nb *= 2
    buckets = [[] for _ in range(nb)]
    for k, x in keymap.items():
        kb = k.encode("ascii")
        h = fnv1a(kb)
        buckets[h & (nb - 1)].append((h, kb, toff[x]))
    karea, offs = bytearray(), []
    for b in buckets:
        offs.append(len(karea))
        for h, kb, to in sorted(b):
            karea += struct.pack("<IB", h, len(kb)) + kb + struct.pack("<I", to)
    offs.append(len(karea))
    btab = struct.pack("<%dI" % len(offs), *offs)
    bt_off = SECTOR
    k_off = bt_off + len(btab) + pad(len(btab))
    t_off = k_off + len(karea) + pad(len(karea))
    size = t_off + len(tarea)
    # ---- fuzzy section: word-level SymSpell delete index, see FORMAT.md ----
    freq = collections.Counter(w for k in keymap for w in set(k.split())
                               if FZ_MIN_WORD <= len(w) <= FZ_MAX_WORD and not w.isdigit())
    fwords = sorted(freq, key=lambda w: (-freq[w], w))  # word offset order == frequency rank
    fw_area, fw_off = bytearray(), {}
    for w in fwords:
        fw_off[w] = len(fw_area)
        fw_area += bytes([len(w)]) + w.encode("ascii")
    ents = [(fnv1a(v.encode("ascii")), fw_off[w]) for w in fwords for v in variants(w)]
    fnb = 1
    while fnb * 4 < len(ents):
        fnb *= 2
    fb = [[] for _ in range(fnb)]
    for h, wo in ents:
        fb[h & (fnb - 1)].append((h, wo))
    fent, foffs = bytearray(), []
    for b in fb:
        foffs.append(len(fent))
        for h, wo in sorted(b):
            fent += struct.pack("<II", h, wo)
    foffs.append(len(fent))
    fbtab = struct.pack("<%dI" % len(foffs), *foffs)
    t_end = t_off + len(tarea)
    fw_off_abs = t_end + pad(t_end)
    fbt_abs = fw_off_abs + len(fw_area) + pad(len(fw_area))
    fent_abs = fbt_abs + len(fbtab) + pad(len(fbtab))
    size = fent_abs + len(fent)
    src = ("%s | %s | %s" % (source, lic, "one context sentence per article (<= 100 chars), ASCII-transliterated")).encode("ascii")
    hdr = b"KB3\0" + struct.pack("<8I", 3, nb, nk, len(toff), bt_off, k_off, t_off, size) + \
        struct.pack("<6I", fnb, fbt_abs, fent_abs, fw_off_abs, len(ents), len(fwords)) + src
    assert len(hdr) <= SECTOR
    with open(out, "wb") as f:
        f.write(hdr + bytes(SECTOR - len(hdr)))
        f.write(btab + bytes(pad(len(btab))))
        f.write(karea + bytes(pad(len(karea))))
        f.write(tarea + bytes(pad(t_end)))
        f.write(fw_area + bytes(pad(len(fw_area))))
        f.write(fbtab + bytes(pad(len(fbtab))))
        f.write(fent)
    assert os.path.getsize(out) == size
    sizes = [offs[i + 1] - offs[i] for i in range(nb)]
    one = sum(1 for i in range(nb) if offs[i] // SECTOR == max(offs[i + 1] - 1, offs[i]) // SECTOR)
    return dict(n_records=len(toff), n_keys=nk, n_articles=len(arts), n_redirects=len(reds), n_buckets=nb,
                file_size=size, fuzzy_bytes=size - fw_off_abs, fuzzy_words=len(fwords), fuzzy_entries=len(ents),
                avg_fuzzy_bucket_entries=len(ents) / fnb, max_bucket_bytes=max(sizes), avg_bucket_bytes=sum(sizes) / nb,
                pct_buckets_le_512=100 * sum(s <= SECTOR for s in sizes) / nb,
                pct_buckets_one_sector=100 * one / nb)

# ---- download ----
def download():
    cache = os.path.join(HERE, "cache")
    os.makedirs(cache, exist_ok=True)
    path, meta_p = os.path.join(cache, "simplewiki-latest-pages-articles.xml.bz2"), os.path.join(cache, "meta.json")
    if not os.path.exists(meta_p):
        h = hashlib.sha1()
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": os.environ.get("CHATDS_USER_AGENT", "ChatDS-KB/1.0")})) as r, open(path + ".part", "wb") as f:
            date = r.headers.get("Last-Modified", "")
            while chunk := r.read(1 << 20):
                f.write(chunk); h.update(chunk)
        os.replace(path + ".part", path)
        json.dump(dict(url=URL, last_modified=date, sha1=h.hexdigest()), open(meta_p, "w"))
    return path, json.load(open(meta_p))

def main():
    path, m = download()
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    date = m["last_modified"]
    stats = build(path, os.path.join(out, "simplewiki.kb"),
                  "Simple English Wikipedia dump %s (%s) sha1 %s" % (date, m["url"], m["sha1"]))
    open(os.path.join(out, "ATTRIBUTION.txt"), "w").write(
        "Text in simplewiki.kb is derived from Simple English Wikipedia articles by Simple English Wikipedia\n"
        "contributors, licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).\n"
        "Source: %s\nDump date (Last-Modified): %s\nDump sha1: %s\n"
        "The text was modified: only one short context sentence (at most 100 characters, cut at a clause\n"
        "boundary when the first sentence is longer) of each article lead is kept, markup removed,\n"
        "transliterated to ASCII.\n" % (m["url"], date, m["sha1"]))
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
