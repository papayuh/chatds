#!/usr/bin/env python3
"""R4 curriculum v2 -- bugfix recipe families (data only).

One family per bug CLASS in eval/suite-v1.1.jsonl (bug-001..bug-050), so
every suite bugfix item is covered by a family that renders the same shape
of broken line from pools instead of from one memorised string.

Contract (see corpus/curriculum_v2.py): each family renders `bug(code)` into
the prompt core and `answer(code)` as the target. Every identifier and
literal the buggy line shows is copied verbatim into the answer -- the whole
point of R4 is that the model learns to COPY the binding, not to recall a
lookup table. Answers are complete single statements so a strict
ast.parse() accepts them (the one exception is bf_except_comma, whose fixed
line is an `except` header that cannot parse standalone: parse=False).

R6 change (R5 scored python 50/50 but bugfix 44/50): a bugfix family had 3
cores x 13 wrappers = 39 prompt shapes, so the model memorised the SHAPE and
fell over on the suite's own bare "Fix this Python line: <code>" form.
Every family now carries >= 8 train cores (the bare "{bug}" among them,
which is exactly the suite's shape) and >= 2 held-out dev cores, the
context-carrying families spell their context several plain ways, and the
13 context/semantic families plus the six that failed get rows=5000 instead
of the default 4000 so the shapes that matter are not outnumbered by tier-1
syntax rows.

Nothing here is ever executed; bug lines are deliberately unparseable.
"""
from corpus.curriculum_v2 import R

# Pools beyond the framework's shared list. Prefixed `bug_` so they can
# never collide with the python recipe module's EXTRA_POOLS.
_FILE_STEMS = [
    "f", "data", "notes", "log", "out", "input", "report", "scores", "users",
    "items", "config", "readme", "dump", "raw", "clean", "final", "draft",
    "index", "table", "sheet", "record", "entry", "sample", "batch", "chunk",
    "part", "page", "book", "story", "letter", "memo", "plan", "budget",
    "sales", "orders", "stock", "prices", "counts", "totals", "results",
    "output", "errors", "debug", "trace", "audit", "history", "export",
    "summary", "detail", "roster", "ledger", "invoice", "receipt", "schedule",
    "roadmap", "backlog", "survey", "list", "words", "notes2"]
_FILE_PREFIXES = ["", "my_", "old_", "new_", "tmp_", "raw_"]
_FILE_EXTS = ["txt", "csv", "json", "log", "md", "dat"]

EXTRA_POOLS = {
    # function parameters / loop targets: needs "a", "x", "k" for suite bindings
    "bug_param": {"kind": "name", "train": [
        "a", "x", "n", "i", "v", "k", "s", "t", "p", "q", "u", "w", "m",
        "val", "num", "arg", "item", "key", "obj", "size", "flag", "temp",
        "count", "index", "first", "left", "right", "start", "end", "low",
        "high", "node", "word", "text", "line", "label", "price", "score",
        "total", "depth", "width", "limit", "base", "step", "rate", "mode",
        "seed", "spot", "head", "tail"],
        "dev": ["alpha", "beta", "gamma", "delta", "zeta", "omega",
                "cursor", "marker"]},
    # the second parameter: needs "b" (bug-037) and "v" (bug-019)
    "bug_param_b": {"kind": "name", "train": [
        "b", "y", "j", "z", "v", "c", "g", "h", "r", "o", "d2", "other",
        "second", "extra", "more", "rest", "back", "off", "gap", "pad",
        "mate", "aux", "peer", "next_one"],
        "dev": ["theta", "kappa", "sigma", "tau", "rho", "iota"]},
    "bug_exc": {"kind": "name", "train": [
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "RuntimeError", "OSError", "IOError", "AttributeError", "NameError",
        "ZeroDivisionError", "ImportError", "StopIteration",
        "FileNotFoundError", "ArithmeticError", "LookupError",
        "OverflowError", "AssertionError"],
        "dev": ["RecursionError", "UnicodeError", "EOFError", "BufferError"]},
    "bug_err_var": {"kind": "name", "train": [
        "e", "err", "ex", "exc", "error", "ee", "e1", "bad", "why", "cause",
        "oops", "problem"],
        "dev": ["fault", "issue", "trouble", "snag"]},
    # "total: " style prefixes for string + number concatenation. R5 output
    # `print('total: ' + str('5))` on bug-020: 24 labels, all ending ": ",
    # was a shape to memorise. 72 labels x three trailing punctuations.
    "bug_label": {"kind": "quote", "train": [
        stem + punct
        for stem in ("total", "count", "sum", "value", "score", "age",
                     "result", "size", "price", "answer", "length", "index",
                     "level", "speed", "width", "height", "depth", "rate",
                     "step", "limit", "amount", "cost", "weight", "time")
        for punct in (": ", "=", " -> ")],
        "dev": [stem + punct
                for stem in ("grade", "volume", "stock", "mass")
                for punct in (": ", "=", " -> ")]},
    # zero-argument str methods; needs "strip" for bug-039
    "bug_method": {"kind": "name", "train": [
        "strip", "upper", "lower", "title", "split", "lstrip", "rstrip",
        "capitalize", "swapcase", "isdigit", "isalpha", "islower",
        "isupper", "isspace", "casefold", "splitlines"],
        "dev": ["encode", "isnumeric", "istitle", "isascii"]},
    # quoted digit strings that int() accepts; needs "3" for bug-014
    "bug_digits": {"kind": "quote",
                   "train": [str(i) for i in range(0, 46)],
                   "dev": [str(i) for i in range(46, 60)]},
    # quoted words int() cannot parse; needs "abc" for bug-034. R5 echoed the
    # prompt back unchanged on that item, so this pool is wide enough (100)
    # that "the quoted thing in the prompt" cannot be a memorised constant.
    "bug_word_text": {"kind": "quote", "train": [
        "abc", "xyz", "hello", "five", "ten", "none", "nan", "foo", "bar",
        "oops", "word", "text", "many", "few", "blank", "empty", "unknown",
        "zero", "one", "two", "three", "four", "later", "soon", "yes", "no",
        "maybe", "extra", "plenty", "some", "lots", "half", "twice", "double",
        "single", "pair", "dozen", "score", "count", "total", "sum", "value",
        "number", "digit", "figure", "amount", "size", "age", "price", "cost",
        "name", "title", "label", "tag", "code", "key", "id", "kind", "type",
        "shape", "color", "red", "blue", "green", "cat", "dog", "bird", "fish",
        "tree", "leaf", "stone", "river", "cloud", "north", "south", "east",
        "west", "up", "down", "left", "right", "first", "last", "next", "prev",
        "start", "stop", "open", "close", "true", "false", "null", "nil",
        "todo", "wip", "draft", "final", "old", "new", "spam", "eggs"],
        "dev": ["mystery", "unset", "junk", "misc", "other", "rhubarb",
                "quince", "medlar"]},
    # Literal texts for a dict value: rendered verbatim in prompt and code
    # (kind "name"), so one pool covers ints, strings, floats, bools and
    # containers. bug-010 needs "1".
    "bug_dict_val": {"kind": "name",
                     "train": [str(i) for i in range(0, 40)] + [
                         "'red'", "'blue'", "'bob'", "'ok'", "'yes'", "'no'",
                         "'a'", "'x'", "'one'", "'two'", "'hello'", "'cat'",
                         "'dog'", "'left'", "'up'", "'new'", "'done'",
                         "True", "False", "None", "0.5", "1.5", "2.25",
                         "3.14", "[]", "[1]", "[1, 2]", "{}"],
                     "dev": ["77", "88", "99", "9.75", "'kite'", "'mira'",
                             "[9]", "'beacon'"]},
    # Optional second entry of a dict literal (""  = the one-entry form the
    # suite shows). Rendered verbatim, so it needs no slot of its own.
    # "" repeated so roughly a third of the rows keep the suite's one-entry
    # shape: a single "" among 20 tails made the two-entry form the norm,
    # which is the same "one memorised shape" trap in a new costume.
    "bug_dict_tail": {"kind": "name", "train": [""] * 9 + [
        ", 'b': 2", ", 'name': 'bob'", ", 'age': 30", ", 'y': 5",
        ", 'city': 'rome'", ", 'ok': True", ", 'n': 7", ", 'size': 12",
        ", 'color': 'red'", ", 'id': 3", ", 'k2': 9", ", 'left': 1",
        ", 'top': 0", ", 'w': 4", ", 'flag': False", ", 'cost': 25",
        ", 'tag': 'new'", ", 'z': 8", ", 'count': 2"],
        "dev": [", 'q': 11", ", 'zone': 'north'", ", 'mm': 6"]},
    # Optional body after an `if ...:` header ("" = the header-only form the
    # suite shows for bug-018). No `return`: the answer must parse at module
    # level.
    "bug_if_tail": {"kind": "name", "train": [""] * 7 + [
        " pass", " print(1)", " print(0)", " print('yes')",
        " print('found')", " x = 1", " total = 1", " ok = True",
        " count = 0", " print('hit')", " n = 2", " flag = True",
        " print(2)", " y = 3", " print('here')"],
        "dev": [" print(9)", " done = True", " print('later')"]},
    # File names for the unclosed-file family: 2160 of them, because R5
    # answered bug-045 with `f.read('f')` -- it had learnt the file name as
    # part of the shape.
    "bug_file": {"kind": "quote",
                 "train": [p + s + "." + e for p in _FILE_PREFIXES
                           for s in _FILE_STEMS for e in _FILE_EXTS],
                 "dev": [s + "." + e
                         for s in ("archive", "backup", "temp", "scratch",
                                   "spool", "inbox")
                         for e in _FILE_EXTS]},
    # ponytail: zero-argument read methods only. `.write(x)` would need a
    # payload slot that the bare "{bug}" core cannot show, and the suite's
    # accept regex (`with\s+open\(|\.close\(\s*\)`) does not care which
    # method the fixed line calls.
    "bug_file_method": {"kind": "name",
                        "train": ["read", "readlines", "readline"],
                        "dev": ["truncate", "flush"]},
    # The fixed digit in `int('1')` for bug-034 -- the suite accepts any
    # digit string, so this slot is answer-only (see bf_int_of_word) and the
    # model must learn to produce a digit, not just recall the constant '1'.
    "bug_digit": {"kind": "quote",
                  "train": [str(i) for i in range(1, 10)]
                           + ["10", "12", "42", "100"],
                  "dev": ["7", "11"]},

    # ---- R9: bug-line CONTEXT variation -----------------------------------
    # R8 rendered a bug family's defect inside exactly one surrounding shape
    # (has_key only ever inside "if ...:", iteritems only in a for header,
    # def-missing-colon only ever a 1-param def). Real pastes show the same
    # broken construct bare, assigned, printed, or in an if/while test. `ctx`
    # is an ordinary slot like any other -- always answer_only (it picks a
    # STRUCTURE, not a value the prompt names, so the copy-invariant does not
    # apply to it) -- and the family's bug()/answer() branch on it so the
    # exact same context wraps both lines.
    "bug_ctx": {"kind": "name", "train": ["bare", "assign", "print", "if"],
                "dev": ["while"]},
    # iteritems' original for-header context is unique to it (a/b unpacking
    # only means anything in a for-loop), so it gets its own set.
    "bug_ctx_for": {"kind": "name", "train": ["bare", "assign", "for"],
                    "dev": ["print"]},
    # =/== and is/== confusion only ever happens inside a conditional test.
    "bug_ctx_cond": {"kind": "name", "train": ["if"], "dev": ["while"]},
    # Second-parameter names for the def-colon param-count family (bug_param
    # already supplies the first, bug_param_b the second).
    "bug_param_c": {"kind": "name", "train": [
        "c", "m", "q", "l", "u", "w", "g", "r", "arg2", "arg3", "third",
        "extra2", "val2", "num2", "item2"],
        "dev": ["mu", "nu", "xi"]},
    "bug_nparams": {"kind": "name", "train": [0, 1, 2], "dev": [3]},
    "bug_def_body": {"kind": "name", "train": ["return", "print"],
                     "dev": ["pass"]},
    "bug_assign_var": {"kind": "name", "train": [
        "found", "ok", "result", "flag", "hit", "pairs", "n", "added",
        "done", "value"], "dev": ["outcome", "seen"]},
}


def _raw(v):
    """Bare text inside a code-rendered literal ("'hi'" -> "hi"); a bare
    identifier passes through unchanged. Used where the buggy line needs the
    unquoted form of a slot the answer quotes."""
    return v[1:-1] if len(v) > 1 and v[0] == v[-1] and v[0] in "'\"" else v


def _quoted(v):
    """Code-rendered literal form of a slot ("Hello" -> "'Hello'"). Pools of
    capitalized words render bare (kind "name"), so quote them here."""
    return v if len(v) > 1 and v[0] == v[-1] and v[0] in "'\"" else "'%s'" % v


# Plain, short trailing contexts. The suite says nothing but the broken line
# ("Fix this Python line: d = {"a": 1"), so "{bug}" alone is always a train
# core; the rest are the everyday ways someone tacks on what they noticed.
_SYNTAX_HINTS = ("-- it does not run", "(syntax error)", ", which crashes",
                 "-- what is wrong?", "-- one character is missing",
                 "raises an error")
_SYNTAX_DEV = ("-- python will not parse it", "(it fails to compile)")
# For code that parses but is wrong at run time: nothing is "missing a
# character" there.
_SEM_HINTS = ("-- it does not run", ", which crashes", "-- what is wrong?",
              "raises an error", "-- the result is wrong", "-- this is a bug")
_SEM_DEV = ("-- it breaks at run time", "(the behaviour is wrong)")
# For code python 3 simply no longer accepts.
_API_HINTS = ("-- it does not run", ", which crashes", "-- what is wrong?",
              "raises an error", "-- this is python 2 code",
              "-- python 3 rejects it")
_API_DEV = ("-- it worked on python 2 only", "(the name is gone in python 3)")

_HINTS = {"syn": (_SYNTAX_HINTS, _SYNTAX_DEV),
          "sem": (_SEM_HINTS, _SEM_DEV),
          "api": (_API_HINTS, _API_DEV)}


def _tack(hint):
    return "{bug}" + ("" if hint[0] in ",;" else " ") + hint


def _c(kind, *specific):
    """Train cores: the bare line (exactly the suite's own shape), six plain
    short contexts, plus family-specific descriptions of the defect."""
    return ["{bug}"] + [_tack(h) for h in _HINTS[kind][0] + specific]


def _d(kind, *specific):
    """Held-out dev cores; never a phrasing that appears in train."""
    return [_tack(h) for h in _HINTS[kind][1] + specific]


def _distinct(*slots):
    return lambda c: len({c[s] for s in slots}) == len(slots)


def _wrap_ctx(frag, ctx, av, tail=" pass"):
    """R9: wrap the SAME defect/fixed fragment in a chosen surrounding
    context so a family's bug()/answer() render identically-shaped lines
    that differ only in the fragment itself. `av` is the assign-context
    variable name; `tail` is what follows an if/while header's colon (a
    bugfix family with its own tail pool, e.g. has_key's bug_if_tail, can
    pass that through instead of the default)."""
    if ctx == "bare":
        return frag
    if ctx == "assign":
        return "%s = %s" % (av, frag)
    if ctx == "print":
        return "print(%s)" % frag
    if ctx == "if":
        return "if %s:%s" % (frag, tail)
    if ctx == "while":
        return "while %s:%s" % (frag, tail)
    raise ValueError(ctx)


def _def_params_distinct(c):
    """Only the params bf_colon_def actually USES for this row's nparams
    need to be distinct -- an unused v2/v3 collision is harmless."""
    n = c["nparams"]
    names = [c["v"], c["v2"], c["v3"]][:n]
    return len(set(names)) == len(names)


def _def_line(c, buggy):
    """bf_colon_def: 0-3 params, a plain-return/print/pass body -- the bug
    line is missing only the header's closing colon."""
    n = int(c["nparams"])
    params = [c["v"], c["v2"], c["v3"]][:n]
    plist = ", ".join(params)
    if n == 0:
        body = "pass"
    elif c["body_kind"] == "print":
        body = "print(%s)" % params[0]
    else:
        body = "return %s" % params[0]
    colon = "" if buggy else ":"
    return "def %s(%s)%s %s" % (c["fn"], plist, colon, body)


def _iteritems_line(c, buggy):
    """bf_iteritems: bare/assigned/printed call, or the original for-header
    unpacking a/b."""
    call = "%s.iteritems()" % c["d"] if buggy else "%s.items()" % c["d"]
    if c["ctx"] == "for":
        if buggy:
            return "for %s, %s in %s:" % (c["a"], c["b"], call)
        return "for %s, %s in %s: print(%s, %s)" % (
            c["a"], c["b"], call, c["a"], c["b"])
    return _wrap_ctx(call, c["ctx"], c["av"])


RECIPES = [

    # ---------------------------------------------------------- unclosed
    R("bf_unclosed_print", "bugfix", 1, ["bug-001"],
      _c("syn", "-- the call is never closed", "(a bracket is still open)"),
      _d("syn", "with one paren too few"),
      {"w": "word_lit"},
      lambda c: "print(%s)" % c["w"],
      bug=lambda c: "print(%s" % c["w"],
      suite_bindings={"bug-001": {"w": "hi"}}),

    R("bf_unclosed_len", "bugfix", 2, ["bug-024"],
      _c("syn", "-- one call is left open",
         "(the outer call has no closing paren)"),
      _d("syn", "where a paren is still hanging"),
      {"xs": "list_var"},
      lambda c: "print(len(%s))" % c["xs"],
      bug=lambda c: "print(len(%s)" % c["xs"],
      suite_bindings={"bug-024": {"xs": "xs"}}),

    R("bf_unclosed_group", "bugfix", 1, ["bug-007"],
      _c("syn", "-- the grouping is unbalanced", "(the sum stays open)"),
      _d("syn", "missing its final paren"),
      {"n": "num_var", "a": "small_int", "b": "small_int"},
      lambda c: "%s = (%s + %s)" % (c["n"], c["a"], c["b"]),
      bug=lambda c: "%s = (%s + %s" % (c["n"], c["a"], c["b"]),
      suite_bindings={"bug-007": {"n": "x", "a": 1, "b": 2}}),

    R("bf_unclosed_bracket", "bugfix", 1, ["bug-009"],
      _c("syn", "-- the list never ends", "(square bracket left open)"),
      _d("syn", "lacking the closing square bracket"),
      {"xs": "list_var", "a": "small_int", "b": "small_int", "c": "small_int"},
      lambda c: "%s = [%s, %s, %s]" % (c["xs"], c["a"], c["b"], c["c"]),
      bug=lambda c: "%s = [%s, %s, %s" % (c["xs"], c["a"], c["b"], c["c"]),
      guard=_distinct("a", "b", "c"),
      suite_bindings={"bug-009": {"xs": "xs", "a": 1, "b": 2, "c": 3}}),

    # R5 answered bug-010 with `d = 'a'`: it had seen one dict shape
    # ({name: small int}) and dropped the braces entirely. The value is now
    # any literal and the mapping sometimes carries a second entry.
    R("bf_unclosed_brace", "bugfix", 1, ["bug-010"],
      _c("syn", "-- the mapping is left open", "(no closing curly brace)"),
      _d("syn", "that forgets to shut the brace"),
      {"d": "dict_var", "k": "key_lit", "v": "bug_dict_val",
       "t": "bug_dict_tail"},
      lambda c: "%s = {%s: %s%s}" % (c["d"], c["k"], c["v"], c["t"]),
      bug=lambda c: "%s = {%s: %s%s" % (c["d"], c["k"], c["v"], c["t"]),
      rows=5000,
      suite_bindings={"bug-010": {"d": "d", "k": "a", "v": "1", "t": ""}}),

    R("bf_unclosed_string", "bugfix", 1, ["bug-008"],
      _c("syn", "-- the text runs past the end", "(no second quote mark)"),
      _d("syn", "whose string is never terminated"),
      {"w": "word_lit"},
      lambda c: "print(%s)" % c["w"],
      bug=lambda c: 'print("%s)' % _raw(c["w"]),
      suite_bindings={"bug-008": {"w": "hi"}}),

    R("bf_quote_mismatch", "bugfix", 2, ["bug-025"],
      _c("syn", "-- it opens and shuts with different quote marks",
         "(the two quotes disagree)"),
      _d("syn", "whose quote characters do not match"),
      {"s": "str_var", "w": "word_lit"},
      lambda c: "%s = %s" % (c["s"], c["w"]),
      bug=lambda c: "%s = '%s" % (c["s"], _raw(c["w"])) + '"',
      suite_bindings={"bug-025": {"s": "s", "w": "hello"}}),

    # ------------------------------------------------------ missing colon
    # R9: R8 only ever rendered a fixed 1-param "def f(x): return x" shape.
    # Real defs show up with 0-3 params and a plain/print/pass body; every
    # row still differs from its bug line by exactly the missing colon.
    R("bf_colon_def", "bugfix", 1, ["bug-002"],
      _c("syn", "-- the header stops short of its punctuation",
         "(a function header needs one more character)"),
      _d("syn", "where the definition line is not terminated"),
      {"fn": "fn_name", "v": "bug_param", "v2": "bug_param_b",
       "v3": "bug_param_c", "nparams": "bug_nparams",
       "body_kind": "bug_def_body"},
      lambda c: _def_line(c, buggy=False),
      bug=lambda c: _def_line(c, buggy=True),
      guard=_def_params_distinct,
      answer_only_slots=("v", "v2", "v3", "nparams", "body_kind"),
      suite_bindings={"bug-002": {"fn": "f", "v": "x", "v2": "y", "v3": "z",
                                  "nparams": 1, "body_kind": "return"}}),

    R("bf_colon_for", "bugfix", 1, ["bug-003"],
      _c("syn", "-- the loop header is unfinished",
         "(nothing separates the header from the body)"),
      _d("syn", "where the loop line lacks its punctuation"),
      {"i": "idx_var", "k": "small_int"},
      lambda c: "for %s in range(%s): print(%s)" % (c["i"], c["k"], c["i"]),
      bug=lambda c: "for %s in range(%s) print(%s)" % (c["i"], c["k"], c["i"]),
      guard=lambda c: c["k"] > 0,
      suite_bindings={"bug-003": {"i": "i", "k": 5}}),

    R("bf_colon_if", "bugfix", 1, ["bug-004"],
      _c("syn", "-- the test and the body run together",
         "(the branch header is incomplete)"),
      _d("syn", "where the condition is never closed off"),
      {"n": "num_var", "k": "small_int"},
      lambda c: "if %s > %s: print(%s)" % (c["n"], c["k"], c["n"]),
      bug=lambda c: "if %s > %s print(%s)" % (c["n"], c["k"], c["n"]),
      suite_bindings={"bug-004": {"n": "x", "k": 0}}),

    R("bf_colon_while", "bugfix", 1, ["bug-005"],
      _c("syn", "-- the repeat header is unfinished",
         "(the test and the update are not separated)"),
      _d("syn", "where the loop test is not closed off"),
      {"n": "num_var", "k": "small_int"},
      lambda c: "while %s < %s: %s = %s + 1" % (c["n"], c["k"], c["n"], c["n"]),
      bug=lambda c: "while %s < %s %s = %s + 1" % (c["n"], c["k"], c["n"], c["n"]),
      suite_bindings={"bug-005": {"n": "x", "k": 5}}),

    R("bf_colon_while_true", "bugfix", 2, ["bug-035"],
      _c("syn", "-- the endless loop header is incomplete",
         "(the body is glued onto the header)"),
      _d("syn", "where the forever loop needs its punctuation"),
      {"k": "small_int"},
      lambda c: "while True: print(%s)" % c["k"],
      bug=lambda c: "while True print(%s)" % c["k"],
      suite_bindings={"bug-035": {"k": 1}}),

    R("bf_colon_class", "bugfix", 2, ["bug-036"],
      _c("syn", "-- the type header is unfinished",
         "(a class header needs one more character)"),
      _d("syn", "where the class line is not terminated"),
      {"C": "cls_name", "k": "small_int"},
      lambda c: "class %s: print(%s)" % (c["C"], c["k"]),
      bug=lambda c: "class %s print(%s)" % (c["C"], c["k"]),
      suite_bindings={"bug-036": {"C": "Dog", "k": 1}}),

    # ------------------------------------------------------------ python 2
    R("bf_py2_print", "bugfix", 1, ["bug-006"],
      _c("api", "-- written the old python 2 way",
         "(printing is a call these days)"),
      _d("api", "using the outdated print form"),
      {"w": "word_lit"},
      lambda c: "print(%s)" % c["w"],
      bug=lambda c: "print %s" % c["w"],
      suite_bindings={"bug-006": {"w": "hello"}}),

    R("bf_xrange", "bugfix", 2, ["bug-015"],
      _c("api", "-- that counting helper is gone in python 3",
         "(the old iteration builtin no longer exists)"),
      _d("api", "calling a builtin python 3 dropped"),
      {"i": "idx_var", "k": "small_int"},
      lambda c: "for %s in range(%s): print(%s)" % (c["i"], c["k"], c["i"]),
      bug=lambda c: "for %s in xrange(%s): print(%s)" % (c["i"], c["k"], c["i"]),
      guard=lambda c: c["k"] > 0,
      suite_bindings={"bug-015": {"i": "i", "k": 5}}),

    R("bf_raw_input", "bugfix", 2, ["bug-016"],
      _c("api", "-- the reading builtin was renamed in python 3",
         "(python 3 spells that prompt helper differently)"),
      _d("api", "using the python 2 name for reading a line"),
      {"s": "str_var"},
      lambda c: "%s = input()" % c["s"],
      bug=lambda c: "%s = raw_input()" % c["s"],
      suite_bindings={"bug-016": {"s": "name"}}),

    R("bf_except_comma", "bugfix", 2, ["bug-017"],
      _c("api", "-- python 2 comma form of catching an error",
         "(the handler binds its variable the old way)"),
      _d("api", "written with the retired handler syntax"),
      {"exc": "bug_exc", "ev": "bug_err_var"},
      lambda c: "except %s as %s:" % (c["exc"], c["ev"]),
      bug=lambda c: "except %s, %s:" % (c["exc"], c["ev"]),
      parse=False,
      suite_bindings={"bug-017": {"exc": "Exception", "ev": "e"}}),

    # R5 answered bug-018 with `d['a']`: the training answer always carried a
    # `print(d[k])` body, so the model reproduced the body and lost the test.
    # The body is now a slot that is empty as often as not -- and when it is
    # empty the answer is the suite's own `if "a" in d:`. R9: has_key showed
    # up ONLY inside an if-header (the R8 spec's own named example); it is
    # a plain boolean expression, so it now also renders bare/assigned/
    # printed/while, keeping the if-header (with its tail pool) as one ctx
    # among five rather than the only one.
    R("bf_has_key", "bugfix", 2, ["bug-018"],
      _c("api", "-- that mapping method vanished in python 3",
         "(membership is tested with an operator now)"),
      _d("api", "using a dict method python 3 removed"),
      {"d": "dict_var", "k": "key_lit", "t": "bug_if_tail", "ctx": "bug_ctx",
       "av": "bug_assign_var"},
      lambda c: _wrap_ctx("%s in %s" % (c["k"], c["d"]), c["ctx"], c["av"],
                          tail=c["t"]),
      bug=lambda c: _wrap_ctx("%s.has_key(%s)" % (c["d"], c["k"]), c["ctx"],
                              c["av"], tail=c["t"]),
      rows=5000, answer_only_slots=("t", "ctx", "av"),
      suite_bindings={"bug-018": {"d": "d", "k": "a", "t": "", "ctx": "if",
                                  "av": "found"}}),

    # R9: iteritems showed up ONLY in a for-header (the R8 spec's other
    # named example). The call itself is a plain expression too, so it now
    # also renders bare/assigned/printed, keeping the for-header (with its
    # a/b unpacking) as one ctx among four.
    R("bf_iteritems", "bugfix", 2, ["bug-019"],
      _c("api", "-- the pair walker lost its python 2 name",
         "(that view method is spelled shorter now)"),
      _d("api", "walking pairs the python 2 way"),
      {"d": "dict_var", "a": "bug_param", "b": "bug_param_b",
       "ctx": "bug_ctx_for", "av": "bug_assign_var"},
      lambda c: _iteritems_line(c, buggy=False),
      bug=lambda c: _iteritems_line(c, buggy=True),
      guard=_distinct("a", "b"),
      answer_only_slots=("ctx", "a", "b", "av"),
      suite_bindings={"bug-019": {"d": "d", "a": "k", "b": "v", "ctx": "for",
                                  "av": "pairs"}}),

    # ------------------------------------------------- = / == confusion
    # R9: this only ever happens inside a conditional test, so its context
    # variation is just if <-> while rather than the fuller bare/assign/
    # print/if/while set _wrap_ctx offers.
    R("bf_assign_in_if", "bugfix", 2, ["bug-011"],
      _c("syn", "-- a branch cannot assign like that",
         "(the test uses the wrong operator)"),
      _d("syn", "comparing with a single equals sign"),
      {"n": "num_var", "k": "small_int", "ctx": "bug_ctx_cond"},
      lambda c: "%s %s == %s: print(%s)" % (c["ctx"], c["n"], c["k"], c["n"]),
      bug=lambda c: "%s %s = %s: print(%s)" % (c["ctx"], c["n"], c["k"], c["n"]),
      answer_only_slots=("ctx",),
      suite_bindings={"bug-011": {"n": "x", "k": 5, "ctx": "if"}}),

    R("bf_assign_in_not", "bugfix", 3, ["bug-041"],
      _c("syn", "-- a negated test cannot store a value",
         "(the operator inside the negation is wrong)"),
      _d("syn", "negating an assignment instead of a comparison"),
      {"n": "num_var", "k": "small_int"},
      lambda c: "if not %s == %s: print(%s)" % (c["n"], c["k"], c["n"]),
      bug=lambda c: "if not %s = %s:" % (c["n"], c["k"]),
      suite_bindings={"bug-041": {"n": "x", "k": 5}}),

    R("bf_compare_as_stmt", "bugfix", 2, ["bug-022"],
      _c("sem", "-- this was meant to store the value, not test it",
         "(comparing on its own line does nothing)"),
      _d("sem", "which compares where it should assign"),
      {"n": "num_var", "k": "small_int"},
      lambda c: "%s = %s" % (c["n"], c["k"]),
      bug=lambda c: "%s == %s" % (c["n"], c["k"]),
      suite_bindings={"bug-022": {"n": "x", "k": 5}}),

    R("bf_is_literal", "bugfix", 3, ["bug-044"],
      _c("sem", "-- identity is not the right check for a number",
         "(comparing objects where values were meant)"),
      _d("sem", "testing identity instead of equality"),
      {"n": "num_var", "k": "small_int", "ctx": "bug_ctx_cond"},
      lambda c: "%s %s == %s: print(%s)" % (c["ctx"], c["n"], c["k"], c["n"]),
      # R9: the bug line now carries the same print(n) body as the fix (it
      # used to be header-only while the answer added a body from nowhere).
      bug=lambda c: "%s %s is %s: print(%s)" % (c["ctx"], c["n"], c["k"], c["n"]),
      answer_only_slots=("ctx",),
      suite_bindings={"bug-044": {"n": "x", "k": 5, "ctx": "if"}}),

    # ------------------------------------------------ typos / wrong names
    # R9: this typo'd attribute access was ONLY ever print-wrapped; it is a
    # plain expression, so it now also renders bare/assigned/if/while.
    R("bf_len_typo", "bugfix", 2, ["bug-012"],
      _c("syn", "-- the size is misspelled and is not an attribute either",
         "(length comes from a builtin, spelled right)"),
      _d("syn", "asking for a misspelled attribute"),
      {"xs": "list_var", "ctx": "bug_ctx", "av": "bug_assign_var"},
      lambda c: _wrap_ctx("len(%s)" % c["xs"], c["ctx"], c["av"]),
      bug=lambda c: _wrap_ctx("%s.lenght" % c["xs"], c["ctx"], c["av"]),
      answer_only_slots=("ctx", "av"),
      suite_bindings={"bug-012": {"xs": "xs", "ctx": "print", "av": "found"}}),

    R("bf_return_typo", "bugfix", 2, ["bug-027"],
      _c("syn", "-- the keyword letters are swapped",
         "(that word is misspelled)"),
      _d("syn", "with a scrambled keyword"),
      {"fn": "fn_name", "v": "bug_param"},
      lambda c: "def %s(%s): return %s" % (c["fn"], c["v"], c["v"]),
      bug=lambda c: "def %s(%s): retrun %s" % (c["fn"], c["v"], c["v"]),
      suite_bindings={"bug-027": {"fn": "f", "v": "x"}}),

    R("bf_module_typo", "bugfix", 2, ["bug-023"],
      _c("sem", "-- no module goes by that spelling",
         "(an extra letter crept into the module name)"),
      _d("sem", "importing a name that does not exist"),
      {"mod": "module_name"},
      lambda c: "import %s" % _raw(c["mod"]),
      bug=lambda c: "import %ss" % _raw(c["mod"]),
      suite_bindings={"bug-023": {"mod": "math"}}),

    # R9: bare-statement only in R8; now also assigned/printed/if/while.
    R("bf_push_append", "bugfix", 2, ["bug-013"],
      _c("sem", "-- lists do not answer to that method",
         "(that is the javascript name for adding)"),
      _d("sem", "calling a list method python does not have"),
      {"xs": "list_var", "k": "small_int", "ctx": "bug_ctx",
       "av": "bug_assign_var"},
      lambda c: _wrap_ctx("%s.append(%s)" % (c["xs"], c["k"]), c["ctx"], c["av"]),
      bug=lambda c: _wrap_ctx("%s.push(%s)" % (c["xs"], c["k"]), c["ctx"], c["av"]),
      answer_only_slots=("ctx", "av"),
      suite_bindings={"bug-013": {"xs": "xs", "k": 4, "ctx": "bare",
                                  "av": "found"}}),

    R("bf_sort_arg", "bugfix", 2, ["bug-038"],
      _c("sem", "-- sorting in place takes nothing",
         "(the list is passed to its own method)"),
      _d("sem", "handing the list to its own sort"),
      {"xs": "list_var", "ctx": "bug_ctx", "av": "bug_assign_var"},
      lambda c: _wrap_ctx("%s.sort()" % c["xs"], c["ctx"], c["av"]),
      bug=lambda c: _wrap_ctx("%s.sort(%s)" % (c["xs"], c["xs"]), c["ctx"], c["av"]),
      answer_only_slots=("ctx", "av"),
      suite_bindings={"bug-038": {"xs": "xs", "ctx": "bare", "av": "found"}}),

    R("bf_missing_parens", "bugfix", 2, ["bug-039"],
      _c("sem", "-- naming a method does not run it",
         "(nothing here actually invokes anything)"),
      _d("sem", "that never calls the method it names"),
      {"s": "str_var", "m": "bug_method", "ctx": "bug_ctx",
       "av": "bug_assign_var"},
      lambda c: _wrap_ctx("%s.%s()" % (c["s"], c["m"]), c["ctx"], c["av"]),
      bug=lambda c: _wrap_ctx("%s.%s" % (c["s"], c["m"]), c["ctx"], c["av"]),
      answer_only_slots=("ctx", "av"),
      suite_bindings={"bug-039": {"s": "s", "m": "strip", "ctx": "bare",
                                  "av": "found"}}),

    # ------------------------------------------------------- type mixing
    R("bf_num_plus_str", "bugfix", 2, ["bug-014"],
      _c("sem", "-- a number and some text cannot be added",
         "(the digits on the right are text)"),
      _d("sem", "adding a quoted number to a real one"),
      {"n": "num_var", "k": "small_int", "t": "bug_digits"},
      lambda c: "%s = %s + int(%s)" % (c["n"], c["k"], c["t"]),
      bug=lambda c: "%s = %s + %s" % (c["n"], c["k"], c["t"]),
      suite_bindings={"bug-014": {"n": "x", "k": 5, "t": "3"}}),

    # R5 answered bug-020 with `print('total: ' + str('5))` -- it produced the
    # str() shape but quoted the number, having only ever seen 24 labels that
    # all ended ": ". 72 labels with three different trailing punctuations.
    R("bf_str_plus_num", "bugfix", 2, ["bug-020"],
      _c("sem", "-- text will not join a bare number",
         "(the number has to become text first)"),
      _d("sem", "gluing a number onto a piece of text"),
      {"lab": "bug_label", "k": "small_int"},
      lambda c: "print(%s + str(%s))" % (c["lab"], c["k"]),
      bug=lambda c: "print(%s + %s)" % (c["lab"], c["k"]),
      rows=5000,
      suite_bindings={"bug-020": {"lab": "total: ", "k": 5}}),

    R("bf_input_arith", "bugfix", 3, ["bug-049"],
      _c("sem", "-- what was read is text, so the arithmetic breaks",
         "(the typed value never becomes a number)"),
      _d("sem", "doing maths on text that was read in"),
      {"n": "num_var", "m": "num_var_b"},
      lambda c: "%s = int(input()); %s = %s + 1" % (c["n"], c["m"], c["n"]),
      bug=lambda c: "%s = input() ; %s = %s + 1" % (c["n"], c["m"], c["n"]),
      suite_bindings={"bug-049": {"n": "x", "m": "y"}}),

    # --------------------------------------------------- dangling tokens
    R("bf_dangling_plus", "bugfix", 2, ["bug-021"],
      _c("syn", "-- the expression stops mid-operation",
         "(there is an operator with nothing after it)"),
      _d("syn", "ending on a stray operator"),
      {"fn": "fn_name", "v": "bug_param"},
      lambda c: "def %s(): return %s" % (c["fn"], c["v"]),
      bug=lambda c: "def %s(): return %s +" % (c["fn"], c["v"]),
      suite_bindings={"bug-021": {"fn": "f", "v": "x"}}),

    R("bf_dangling_and", "bugfix", 2, ["bug-028"],
      _c("syn", "-- the condition trails off after a connector",
         "(a joining word sits there with no second test)"),
      _d("syn", "whose condition ends on a leftover keyword"),
      {"n": "num_var", "k": "small_int"},
      lambda c: "if %s > %s: print(%s)" % (c["n"], c["k"], c["n"]),
      bug=lambda c: "if %s > %s and: print(%s)" % (c["n"], c["k"], c["n"]),
      suite_bindings={"bug-028": {"n": "x", "k": 0}}),

    R("bf_double_comma", "bugfix", 2, ["bug-026"],
      _c("syn", "-- two separators in a row",
         "(an empty slot between the items)"),
      _d("syn", "with a doubled separator inside the list"),
      {"xs": "list_var", "a": "small_int", "b": "small_int",
       "c": "small_int", "e": "small_int"},
      lambda c: "%s = [%s, %s, %s, %s]" % (c["xs"], c["a"], c["b"], c["c"], c["e"]),
      bug=lambda c: "%s = [%s, %s, %s,,%s]" % (c["xs"], c["a"], c["b"], c["c"], c["e"]),
      guard=_distinct("a", "b", "c", "e"),
      suite_bindings={"bug-026": {"xs": "xs", "a": 1, "b": 2, "c": 3, "e": 4}}),

    R("bf_param_comma", "bugfix", 2, ["bug-037"],
      _c("syn", "-- the two parameters are not separated",
         "(a comma is missing between the arguments)"),
      _d("syn", "whose parameter list has no separator"),
      {"fn": "fn_name", "a": "bug_param", "b": "bug_param_b"},
      lambda c: "def %s(%s, %s): return %s" % (c["fn"], c["a"], c["b"], c["a"]),
      bug=lambda c: "def %s(%s %s): return %s" % (c["fn"], c["a"], c["b"], c["a"]),
      guard=_distinct("a", "b"),
      suite_bindings={"bug-037": {"fn": "f", "a": "a", "b": "b"}}),

    R("bf_adjacent_str", "bugfix", 2, ["bug-040"],
      _c("sem", "-- the two pieces of text are only sitting side by side",
         "(nothing joins the two literals)"),
      _d("sem", "where the two literals are never combined"),
      {"u": "key_lit", "v": "key_lit"},
      lambda c: "print(%s + %s)" % (c["u"], c["v"]),
      bug=lambda c: "print(%s %s)" % (c["u"], c["v"]),
      guard=_distinct("u", "v"),
      suite_bindings={"bug-040": {"u": "a", "v": "b"}}),

    # -------------------------------------------------- runtime semantics
    R("bf_div_zero", "bugfix", 2, ["bug-029"],
      _c("sem", "-- dividing by nothing blows up at runtime",
         "(the divisor makes this fail)"),
      _d("sem", "whose divisor cannot be used"),
      {"n": "num_var", "k": "small_int"},
      lambda c: "%s = %s / 1" % (c["n"], c["k"]),
      bug=lambda c: "%s = %s / 0" % (c["n"], c["k"]),
      rows=5000,
      suite_bindings={"bug-029": {"n": "x", "k": 10}}),

    # R5 answered bug-030 with `xs = [1, 2, 3]` -- it copied the context back
    # instead of the fixed index. Every core here shows the list (the family's
    # slots live in the context, not in the broken line), so the model has to
    # learn that the answer comes from the code before the "when".
    R("bf_index_oob", "bugfix", 2, ["bug-030"],
      ["{bug} when {xs} = [{a}, {b}, {c}]",
       "{bug} given {xs} = [{a}, {b}, {c}]",
       "{bug} where {xs} = [{a}, {b}, {c}]",
       "{bug} if {xs} = [{a}, {b}, {c}]",
       "{bug} with {xs} = [{a}, {b}, {c}]",
       "{bug} and {xs} is [{a}, {b}, {c}]",
       "{bug} but {xs} = [{a}, {b}, {c}] only has three items",
       "{bug} -- {xs} = [{a}, {b}, {c}] is too short",
       "{bug} if {xs} holds [{a}, {b}, {c}]",
       "{bug} where {xs} contains [{a}, {b}, {c}]"],
      ["{bug} for {xs} = [{a}, {b}, {c}]",
       "{bug} while {xs} stores [{a}, {b}, {c}]"],
      {"xs": "list_var", "a": "small_int", "b": "small_int", "c": "small_int"},
      lambda c: "%s[2]" % c["xs"],
      bug=lambda c: "%s[3]" % c["xs"],
      guard=_distinct("a", "b", "c"),
      rows=5000,
      suite_bindings={"bug-030": {"xs": "xs", "a": 1, "b": 2, "c": 3}}),

    R("bf_missing_arg", "bugfix", 2, ["bug-031"],
      ["{bug}",
       "{bug} called as {fn}()",
       "{bug} invoked as {fn}()",
       "{bug} but used as {fn}()",
       "{bug} but it is called with no argument",
       "{bug} and {fn}() is called with nothing",
       "{bug} -- the call {fn}() passes no value",
       "{bug} while {fn}() gets no argument",
       "{bug} -- what is wrong?"],
      ["{bug} run as {fn}()",
       "{bug} -- {fn}() is called empty"],
      {"fn": "fn_name", "v": "bug_param", "k": "small_int"},
      lambda c: "%s(%s)" % (c["fn"], c["k"]),
      bug=lambda c: "def %s(%s) : return %s*%s" % (c["fn"], c["v"], c["v"], c["k"]),
      guard=lambda c: c["k"] > 0,
      rows=5000,
      suite_bindings={"bug-031": {"fn": "f", "v": "x", "k": 2}}),

    R("bf_bare_word", "bugfix", 2, ["bug-032"],
      _c("sem", "-- that word is not a variable anywhere",
         "(the text was meant to be a literal)"),
      _d("sem", "treating a word as if it were defined"),
      {"W": "Word_lit"},
      lambda c: "print(%s)" % _quoted(c["W"]),
      bug=lambda c: "print(%s)" % _raw(c["W"]),
      rows=5000,
      suite_bindings={"bug-032": {"W": "Hello"}}),

    R("bf_return_no_def", "bugfix", 2, ["bug-033"],
      _c("sem", "-- handing a value back only works inside a function",
         "(this statement has no function around it)"),
      _d("sem", "sitting outside any function"),
      {"v": "bug_param"},
      lambda c: "def f(%s): return %s" % (c["v"], c["v"]),
      bug=lambda c: "return %s" % c["v"],
      rows=5000,
      suite_bindings={"bug-033": {"v": "x"}}),

    # R5 echoed bug-034 back unchanged (`x = int('abc')`). 100 non-numeric
    # words now, so "the quoted thing" is never the same token twice.
    # The fix is always `int('1')` in R6 -- a memorised constant with nothing
    # to learn. `k` is now a slot drawn from bug_digit (any digit the suite's
    # accept regex `int\(\s*['"]\d+['"]\s*\)` takes); it is answer-only
    # (never shown in the prompt) since it is a value to GENERATE, not copy.
    R("bf_int_of_word", "bugfix", 2, ["bug-034"],
      _c("sem", "-- those characters are not digits",
         "(the conversion cannot read that text)"),
      _d("sem", "converting text that holds no number"),
      {"n": "num_var", "w": "bug_word_text", "k": "bug_digit"},
      lambda c: "%s = int(%s)" % (c["n"], c["k"]),
      bug=lambda c: "%s = int(%s)" % (c["n"], c["w"]),
      rows=5000,
      answer_only_slots=("k",),
      suite_bindings={"bug-034": {"n": "x", "w": "abc", "k": "1"}}),

    R("bf_off_by_one", "bugfix", 3, ["bug-042"],
      _c("sem", "-- it reads past the end", "(the index is shifted by one)"),
      _d("sem", "indexing one step too far"),
      {"i": "idx_var", "xs": "list_var"},
      lambda c: "for %s in range(len(%s)): print(%s[%s])" % (
          c["i"], c["xs"], c["xs"], c["i"]),
      bug=lambda c: "for %s in range(len(%s)): print(%s[%s+1])" % (
          c["i"], c["xs"], c["xs"], c["i"]),
      rows=5000,
      suite_bindings={"bug-042": {"i": "i", "xs": "xs"}}),

    R("bf_mutable_default", "bugfix", 3, ["bug-043"],
      _c("sem", "-- the default list is shared", "(that default survives calls)"),
      _d("sem", "whose default value is built once"),
      {"fn": "fn_name", "xs": "list_var", "k": "small_int"},
      lambda c: "def %s(%s=None): %s = %s or []" % (
          c["fn"], c["xs"], c["xs"], c["xs"]),
      bug=lambda c: "def %s(%s=[]): %s.append(%s)" % (
          c["fn"], c["xs"], c["xs"], c["k"]),
      rows=5000,
      suite_bindings={"bug-043": {"fn": "f", "xs": "xs", "k": 1}}),

    # R5 answered bug-045 with `f.read('f')`: the file name and the method
    # were both memorised shape. 2160 file names x 3 read methods, and the
    # context is spelled several plain ways including the suite's own
    # "without closing".
    R("bf_open_no_close", "bugfix", 3, ["bug-045"],
      ["{bug}",
       "{bug} without closing",
       "{bug} without closing the file",
       "{bug} and the file is never closed",
       "{bug} -- the handle is never released",
       "{bug} leaving the file open",
       "{bug} -- nothing ever closes it",
       "{bug} but the file stays open",
       "{bug} -- what is wrong?"],
      ["{bug} with nothing closing the file",
       "{bug} -- the handle is left dangling"],
      {"fname": "bug_file", "m": "bug_file_method"},
      lambda c: "with open(%s) as f: f.%s()" % (c["fname"], c["m"]),
      bug=lambda c: "open(%s).%s()" % (c["fname"], c["m"]),
      rows=5000,
      suite_bindings={"bug-045": {"fname": "f.txt", "m": "read"}}),

    R("bf_append_reassign", "bugfix", 3, ["bug-046"],
      _c("sem", "-- adding in place hands back nothing",
         "(the list gets replaced by the result)"),
      _d("sem", "that overwrites the list with the call result"),
      {"xs": "list_var", "k": "small_int"},
      lambda c: "%s.append(%s)" % (c["xs"], c["k"]),
      bug=lambda c: "%s = %s.append(%s)" % (c["xs"], c["xs"], c["k"]),
      rows=5000,
      suite_bindings={"bug-046": {"xs": "xs", "k": 4}}),

    R("bf_empty_mean", "bugfix", 3, ["bug-047"],
      ["{bug}",
       "{bug} when {xs} is empty",
       "{bug} if {xs} has no items",
       "{bug} for an empty {xs}",
       "{bug} when {xs} holds nothing",
       "{bug} if {xs} is empty",
       "{bug} -- {xs} can be empty",
       "{bug} and {xs} might have no items",
       "{bug} -- what is wrong?"],
      ["{bug} in the case where {xs} holds no values",
       "{bug} -- {xs} is sometimes empty"],
      {"xs": "list_var"},
      lambda c: "if len(%s) > 0: print(sum(%s)/len(%s))" % (
          c["xs"], c["xs"], c["xs"]),
      bug=lambda c: "print(sum(%s)/len(%s))" % (c["xs"], c["xs"]),
      rows=5000,
      suite_bindings={"bug-047": {"xs": "xs"}}),

    R("bf_dict_missing_key", "bugfix", 3, ["bug-048"],
      _c("sem", "-- that key was never put in", "(the lookup will raise)"),
      _d("sem", "reading a key the mapping does not hold"),
      {"d": "dict_var", "ka": "key_lit", "va": "small_int", "kb": "key_lit"},
      lambda c: "print(%s.get(%s))" % (c["d"], c["kb"]),
      bug=lambda c: "%s = {%s: %s}; print(%s[%s])" % (
          c["d"], c["ka"], c["va"], c["d"], c["kb"]),
      guard=_distinct("ka", "kb"),
      rows=5000,
      suite_bindings={"bug-048": {"d": "d", "ka": "a", "va": 1, "kb": "b"}}),

    R("bf_undefined_param", "bugfix", 3, ["bug-050"],
      ["{bug}",
       "{bug} where {v} is undefined",
       "{bug} but {v} is never defined",
       "{bug} and {v} does not exist",
       "{bug} -- {v} comes from nowhere",
       "{bug} with {v} undefined",
       "{bug} -- nothing defines {v}",
       "{bug} where {v} has no value",
       "{bug} -- what is wrong?"],
      ["{bug} and {v} has no value anywhere",
       "{bug} -- {v} is not defined anywhere"],
      {"fn": "fn_name", "v": "bug_param"},
      lambda c: "def %s(%s): print(%s)" % (c["fn"], c["v"], c["v"]),
      bug=lambda c: "def %s(): print(%s)" % (c["fn"], c["v"]),
      rows=5000,
      suite_bindings={"bug-050": {"fn": "f", "v": "x"}}),
]
