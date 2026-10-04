#!/usr/bin/env python3
"""One markdown table row per run: python3 eval/summarize_run.py RUN_DIR:TAG [...]  (reads RUN_DIR/eval/TAG-*.txt/.jsonl)"""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def val(path, pat):
    m = re.search(pat, open(path).read()) if os.path.exists(path) else None
    return m.group(1) if m else "-"


for a in sys.argv[1:]:
    d, t = a.rsplit(":", 1)
    f = lambda s, ext="txt": "%s/eval/%s-%s.%s" % (d, t, s, ext)
    cr = lambda s: re.search(r"copy rate (\d+/\d+)", subprocess.run([sys.executable, HERE + "/copy_rate.py", "--suite", HERE + "/%s.jsonl" % s, "--outputs", f(s, "jsonl")],
                                                                        capture_output=True, text=True).stdout).group(1)
    acc = "chatds-acceptance-30"
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        os.path.basename(d), val(f(acc), r"facts\s+(\d+/\d+)"), val(f(acc), r"math\s+(\d+/\d+)"), val(f(acc), r"OVERALL\s+(\d+/\d+)"),
        val(f("numwords-40"), r"OVERALL\s+(\d+/\d+)"), val(f("acceptance-draft-20"), r"OVERALL\s+(\d+/\d+)"),
        val(f("suite-v1.1"), r"OVERALL\s+(\d+/\d+)"), cr("kb-probe-150"), cr(acc)))
