#!/usr/bin/env python3
"""Print the exact prompt the DS builds for a question: chatds_prompt.py [--kb KB] QUESTION"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corpus"))
import chatds_runtime as rt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kb")
ap.add_argument("question")
a = ap.parse_args()
sys.stdout.write(rt.format_prompt(a.question, rt.retrieve(a.question, a.kb) if a.kb else None))
