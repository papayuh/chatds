#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run self-contained host gates; report absent optional training dependencies."""
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_TESTS = {
    "corpus/test_chatds_export.py": ("reference/ds-llm/tokenizer.py",),
    "corpus/test_export_train.py": ("reference/ds-llm/tokenizer.py",),
    "corpus/test_fact_answers.py": ("reference/ds-llm/tokenizer.py",),
    "train/test_train_instruct.py": ("reference/ds-llm/model.py", "reference/ds-llm/tokenizer.py", "reference/ds-llm/export.py"),
    "train/test_trim_padding.py": ("reference/ds-llm/model.py", "reference/ds-llm/tokenizer.py", "reference/ds-llm/export.py"),
}


def exclusions(root=ROOT, torch_available=None):
    """Only test dependency existence, never open upstream/source contents."""
    ignored = {}
    if torch_available is None:
        torch_available = importlib.util.find_spec("torch") is not None
    for test, paths in REFERENCE_TESTS.items():
        missing = [path for path in paths if not (root / path).is_file()]
        if test.startswith("train/") and not torch_available:
            missing.append("Python package torch")
        if missing:
            ignored[test] = "missing " + ", ".join(missing)
    if not torch_available:
        ignored["train/test_qat4.py"] = "missing Python package torch"
    return ignored, {}


def main():
    ignored, deselected = exclusions()
    for path, reason in {**ignored, **deselected}.items():
        print(f"SKIP {path}: {reason}", flush=True)
    args = [sys.executable, "-m", "pytest", "-q"]  # pytest.ini testpaths
    args.extend("--ignore=" + path for path in ignored)
    args.extend("--deselect=" + path for path in deselected)
    args.extend(sys.argv[1:])
    result = subprocess.run(args, cwd=ROOT, timeout=600)
    if result.returncode:
        return result.returncode
    # Tokenizer's suite is a standalone script, not pytest test functions.
    if (ROOT / "ds/tokenizer/test_tok.py").is_file():
        result = subprocess.run([sys.executable, "ds/tokenizer/test_tok.py"],
                                cwd=ROOT, timeout=120)
        if result.returncode:
            return result.returncode
    # Clean-room engine and UI host C unit tests are Makefile targets, not pytest.
    for lane in ("clean", "clean/ui"):
        result = subprocess.run(["make", "-C", lane, "test"], cwd=ROOT, timeout=300)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
