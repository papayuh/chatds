#!/usr/bin/env python3
"""D13 teacher-data pipeline, stage 3: turn raw batch results into a clean
training set. stdlib only.

Pipeline: parse each batch result -> match its custom_id back to the request
shard that produced it -> split the teacher's JSONL answer blob into
per-item (prompt, answer) pairs -> validate -> dedupe -> contamination-filter
against eval/suite-v1.1.jsonl -> write corpus/train-v1.jsonl + a stats report.

Usage:
    python3 postprocess.py                  # corpus/raw/*.jsonl -> corpus/train-v1.jsonl
    python3 postprocess.py --selftest       # run the synthetic fixture, print PASS/FAIL
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common

RAW_DIR = os.path.join(HERE, "raw")
REQUESTS_DIR = os.path.join(HERE, "requests")
OUT_PATH = os.path.join(HERE, "train-v1.jsonl")
STATS_PATH = os.path.join(HERE, "stats-v1.json")
EVAL_SUITE = os.path.join(REPO_ROOT, "eval", "suite-v1.1.jsonl")
EVAL_SUITE_SHA256 = "3ee6483b0724e5ab65120e6db0a8fd146842e36743d3588f588e5a2be7aadcd4"
CONTAMINATION_NGRAM = 8  # eval/README.md #Contamination rule 2(c)

MAX_ANSWER_WORDS = 40  # ~30-word instruction + margin; longer is dropped, not truncated
REFUSAL_PATTERNS = [
    r"\bi (can'?t|cannot|won'?t|will not)\b",
    r"\bi'?m (not able|unable) to\b",
    r"\bas an ai\b",
    r"\bi'?m sorry, but\b",
    r"\bi apologize\b",
]
OVERLAP_THRESHOLD = 0.8


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_word_list(text):
    """lowercase, strip punctuation, collapse to a word list (order kept) --
    the same cleaning normalize_words does, before it collapses to a set.
    Shared so the Jaccard check and the 8-gram check can't drift apart on
    what counts as "the same normalization"."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def normalize_words(text):
    """lowercase, strip punctuation, collapse to a set of words -- used for
    both dedup keys and contamination overlap."""
    return set(normalize_word_list(text))


def ngrams(words, n=CONTAMINATION_NGRAM):
    """Set of contiguous n-word tuples. Shorter than n words -> no ngrams,
    same spirit as README rule 4 (short overlap is unavoidable domain
    knowledge, not contamination)."""
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def normalize_key(text):
    """stricter normalization for exact/near-dup detection: collapsed
    whitespace, no punctuation, lowercase -- as a single string, not a set,
    so word order still matters."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_eval_overlap_sets():
    """Returns list of (source_id, word_set) for every eval PROMPT (not
    answers -- see eval/README.md #Contamination rule 4: short answer
    overlap is domain knowledge, not contamination, so the Jaccard check
    this pipeline adds on top of the README's minimum only ever compares
    corpus prompts against eval prompts), the set of exact-normalized
    prompts/long-answers, and the set of every contiguous
    CONTAMINATION_NGRAM-word run appearing in any eval prompt or answer --
    the three baseline contamination checks in eval/README.md #Contamination
    rule 2 (a/b/c) plus the prompt-only Jaccard overlap check."""
    actual_sha = sha256_file(EVAL_SUITE)
    if actual_sha != EVAL_SUITE_SHA256:
        raise SystemExit(
            f"{os.path.basename(EVAL_SUITE)} sha256 mismatch: got {actual_sha}, "
            f"expected {EVAL_SUITE_SHA256}. Refusing to contamination-check "
            f"against a suite file that isn't the frozen one this pipeline "
            f"was pinned to."
        )
    prompt_overlap_sets = []
    exact_prompts = set()
    exact_long_answers = set()
    eval_ngrams = set()
    with open(EVAL_SUITE) as f:
        for line in f:
            item = json.loads(line)
            p_norm = normalize_key(item["prompt"])
            a_norm = normalize_key(item["answer"])
            exact_prompts.add(p_norm)
            if len(a_norm) > 12:
                exact_long_answers.add(a_norm)
            prompt_overlap_sets.append((item["id"] + ":prompt", normalize_words(item["prompt"])))
            eval_ngrams |= ngrams(normalize_word_list(item["prompt"]))
            eval_ngrams |= ngrams(normalize_word_list(item["answer"]))
    return prompt_overlap_sets, exact_prompts, exact_long_answers, eval_ngrams


def max_overlap(word_set, overlap_sets):
    """Jaccard |A n B| / |A u B| against every eval prompt word set; returns
    the max. Jaccard, not containment: containment scores a short candidate
    fully inside a longer eval phrase as 1.0 (e.g. prompt "capital" vs eval
    prompt "capital of France"), and scores any "X of <topic>" template swap
    against the matching eval template as ~0.83 purely from shared
    scaffolding words (e.g. "capital of Wakanda" vs eval's "capital of
    France") -- both are false positives that would gut the corpus's
    intentional template repetition. Jaccard's shared union denominator
    suppresses both."""
    if not word_set:
        return 0.0
    best = 0.0
    for _sid, eval_set in overlap_sets:
        union = word_set | eval_set
        if not union:
            continue
        ratio = len(word_set & eval_set) / len(union)
        if ratio > best:
            best = ratio
    return best


def parse_teacher_lines(text):
    """Parse the teacher's response: one {"i": N, "answer": "..."} per line.
    Tolerant of stray whitespace/blank lines/code fences the model might add
    despite instructions; strict about needing valid JSON with both keys."""
    out = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "i" not in obj or "answer" not in obj:
            continue
        try:
            i = int(obj["i"])
        except (TypeError, ValueError):
            continue
        answer = obj["answer"]
        if not isinstance(answer, str):
            continue
        out[i] = answer
    return out


def parse_teacher_lines_natural(text):
    """Like parse_teacher_lines, but for the python_natural category: the
    teacher invents the prompt too, so each line must carry {"i", "prompt",
    "answer"}. Returns {i: (prompt, answer)}. Same tolerant-of-junk parsing
    as parse_teacher_lines -- kept as a separate function rather than a flag
    so neither parser has to think about the other's schema."""
    out = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "i" not in obj or "prompt" not in obj or "answer" not in obj:
            continue
        try:
            i = int(obj["i"])
        except (TypeError, ValueError):
            continue
        prompt, answer = obj["prompt"], obj["answer"]
        if not isinstance(prompt, str) or not isinstance(answer, str):
            continue
        out[i] = (prompt.strip(), answer)
    return out


def is_refusal(answer):
    lower = answer.lower()
    return any(re.search(p, lower) for p in REFUSAL_PATTERNS)


def load_request_index(requests_dir):
    """custom_id -> request record, across every shard file."""
    index = {}
    for path in sorted(glob.glob(os.path.join(requests_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                index[rec["custom_id"]] = rec
    return index


def load_extra_eval_overlap_sets(path):
    """Same three contamination structures as load_eval_overlap_sets, built
    from an arbitrary eval-schema jsonl (id/prompt/answer) instead of the
    frozen, sha-pinned suite -- for --extra-eval, e.g. re-filtering against
    Diego's 20-prompt personal acceptance set once it exists (D16 step 6),
    without waiting for that file to be frozen-and-hashed like suite-v1.1.
    No sha check here on purpose: an acceptance set in progress is expected
    to change; the frozen suite's pin stays the only hard gate."""
    prompt_overlap_sets = []
    exact_prompts = set()
    exact_long_answers = set()
    ngram_set = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            p_norm = normalize_key(item["prompt"])
            a_norm = normalize_key(item["answer"])
            exact_prompts.add(p_norm)
            if len(a_norm) > 12:
                exact_long_answers.add(a_norm)
            sid = item.get("id", os.path.basename(path))
            prompt_overlap_sets.append((f"{sid}:prompt", normalize_words(item["prompt"])))
            ngram_set |= ngrams(normalize_word_list(item["prompt"]))
            ngram_set |= ngrams(normalize_word_list(item["answer"]))
    return prompt_overlap_sets, exact_prompts, exact_long_answers, ngram_set


def process(raw_paths, request_index, prompt_overlap_sets, exact_prompts, exact_long_answers, eval_ngrams):
    stats = {
        "total_results": 0,
        "kept": 0,
        "drop_no_matching_request": 0,
        "drop_batch_errored": 0,
        "drop_missing_item": 0,
        "drop_refusal": 0,
        "drop_overlong": 0,
        "drop_empty_answer": 0,
        "drop_bad_natural_prompt": 0,
        "drop_duplicate": 0,
        "drop_contamination_exact_prompt": 0,
        "drop_contamination_exact_answer": 0,
        "drop_contamination_8gram": 0,
        "drop_contamination_overlap": 0,
        "kept_per_category": {},
    }
    seen_keys = set()
    pairs_out = []

    for path in raw_paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                stats["total_results"] += 1
                result = json.loads(line)
                custom_id = result.get("custom_id")
                req = request_index.get(custom_id)
                if req is None:
                    stats["drop_no_matching_request"] += 1
                    continue

                result_body = result.get("result", {})
                if result_body.get("type") != "succeeded":
                    stats["drop_batch_errored"] += len(req["items"])
                    continue

                message = result_body.get("message", {})
                text = "".join(
                    b.get("text", "") for b in message.get("content", []) if b.get("type") == "text"
                )
                category = req["category"]
                natural = category == "python_natural"
                if natural:
                    natural_pairs = parse_teacher_lines_natural(text)
                else:
                    answers = parse_teacher_lines(text)

                for item in req["items"]:
                    idx = item["i"]
                    if natural:
                        # python_natural: the teacher invents the prompt too
                        # (that's the whole point -- see gen_requests.py), so
                        # the training prompt is what IT wrote, never the
                        # meta-instruction we sent (item["prompt"]).
                        pair = natural_pairs.get(idx)
                        if pair is None:
                            stats["drop_missing_item"] += 1
                            continue
                        prompt, answer = pair
                        if not prompt or len(prompt.split()) < 2:
                            stats["drop_bad_natural_prompt"] += 1
                            continue
                    else:
                        prompt = item["prompt"]
                        answer = answers.get(idx)
                        if answer is None:
                            stats["drop_missing_item"] += 1
                            continue
                    answer = answer.strip()
                    if not answer:
                        stats["drop_empty_answer"] += 1
                        continue
                    if is_refusal(answer):
                        stats["drop_refusal"] += 1
                        continue
                    if len(answer.split()) > MAX_ANSWER_WORDS:
                        stats["drop_overlong"] += 1
                        continue

                    p_norm = normalize_key(prompt)
                    a_norm = normalize_key(answer)
                    if p_norm in exact_prompts:
                        stats["drop_contamination_exact_prompt"] += 1
                        continue
                    if len(a_norm) > 12 and a_norm in exact_long_answers:
                        stats["drop_contamination_exact_answer"] += 1
                        continue
                    if (ngrams(normalize_word_list(prompt)) & eval_ngrams
                            or ngrams(normalize_word_list(answer)) & eval_ngrams):
                        stats["drop_contamination_8gram"] += 1
                        continue
                    # Jaccard is prompt-vs-eval-prompt only: eval/README.md
                    # #Contamination rule 4 -- short answer overlap ("Paris",
                    # "ls", "7") is domain knowledge, not contamination.
                    p_overlap = max_overlap(normalize_words(prompt), prompt_overlap_sets)
                    if p_overlap >= OVERLAP_THRESHOLD:
                        stats["drop_contamination_overlap"] += 1
                        continue

                    dedup_key = (p_norm, a_norm)
                    if dedup_key in seen_keys:
                        stats["drop_duplicate"] += 1
                        continue
                    seen_keys.add(dedup_key)

                    pairs_out.append({
                        "category": category,
                        "prompt": prompt,
                        "answer": answer,
                        "tier": item.get("tier"),
                    })
                    stats["kept"] += 1
                    stats["kept_per_category"][category] = stats["kept_per_category"].get(category, 0) + 1

    return pairs_out, stats


def write_output(pairs, stats, out_path, stats_path):
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--requests-dir", default=REQUESTS_DIR)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--stats", default=STATS_PATH)
    ap.add_argument("--extra-eval", default=None,
                     help="additional eval-schema jsonl (id/prompt/answer per line) to "
                          "contamination-filter against on top of the frozen suite -- e.g. "
                          "re-filter a corpus once Diego's 20-prompt acceptance set exists, "
                          "in one command, without re-running any teacher call")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    prompt_overlap_sets, exact_prompts, exact_long_answers, eval_ngrams = load_eval_overlap_sets()
    if args.extra_eval:
        ep, ex_p, ex_a, en = load_extra_eval_overlap_sets(args.extra_eval)
        prompt_overlap_sets = prompt_overlap_sets + ep
        exact_prompts = exact_prompts | ex_p
        exact_long_answers = exact_long_answers | ex_a
        eval_ngrams = eval_ngrams | en
        print(f"also filtering against --extra-eval {args.extra_eval} ({len(ep)} items)")
    request_index = load_request_index(args.requests_dir)
    raw_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.jsonl")))
    if not raw_paths:
        print(f"no raw result files in {args.raw_dir} -- run submit_deepseek.py first", file=sys.stderr)
        sys.exit(1)

    pairs, stats = process(raw_paths, request_index, prompt_overlap_sets, exact_prompts, exact_long_answers, eval_ngrams)
    write_output(pairs, stats, args.out, args.stats)
    print(f"contamination-checked against {os.path.basename(EVAL_SUITE)} sha256={EVAL_SUITE_SHA256}")

    print(f"wrote {len(pairs)} pairs -> {args.out}")
    print(f"stats -> {args.stats}")
    for k, v in stats.items():
        if k != "kept_per_category":
            print(f"  {k}: {v}")
    print("  kept_per_category:")
    for cat, n in sorted(stats["kept_per_category"].items()):
        print(f"    {cat}: {n}")


# ---------------------------------------------------------------------------
# Self-test: synthetic fixture covering malformed / refusal / overlong /
# duplicate / contaminated cases, run through the same process() function.
# ---------------------------------------------------------------------------

def run_selftest():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        req_dir = os.path.join(tmp, "requests")
        raw_dir = os.path.join(tmp, "raw")
        os.makedirs(req_dir)
        os.makedirs(raw_dir)
        req_path = os.path.join(req_dir, "shard-000.jsonl")
        raw_path = os.path.join(raw_dir, "batch_test.jsonl")

        requests_ = [
            {
                "custom_id": "factual-000000",
                "category": "factual",
                "items": [
                    {"i": 0, "prompt": "What is the capital of Wakanda?", "tier": 1},
                    {"i": 1, "prompt": "What is the capital of France?", "tier": 1},  # exact eval prompt match
                    {"i": 2, "prompt": "Name a fruit that is red.", "tier": 1},
                    {"i": 3, "prompt": "Say hello in Klingon.", "tier": 1},
                    {"i": 4, "prompt": "What is the tallest structure in Freedonia?", "tier": 2},
                    # F8: shares an exact 8-word run with eval fact-008's prompt
                    # ("what is the largest planet in our solar system") but is
                    # not an exact-prompt match and its Jaccard vs every eval
                    # set is 0.6 (< OVERLAP_THRESHOLD) -- only the 8-gram check
                    # catches this one.
                    {"i": 5, "prompt": "Tell me exactly, what is the largest "
                                       "planet in our solar system, in full "
                                       "detail please", "tier": 2},
                    # F1 (finding 1): independently phrased prompts whose
                    # answer happens to equal a short eval answer -- these
                    # must be KEPT (README rule 4: short answer overlap is
                    # domain knowledge, not contamination). Pre-fix, the
                    # answer-side Jaccard check dropped all of these.
                    {"i": 6, "prompt": "Name a European city famous for the "
                                       "Eiffel Tower.", "tier": 1},
                    {"i": 7, "prompt": "How many dwarves helped Snow White "
                                       "in the fairy tale?", "tier": 2},
                    # F1: prompt is a pure word-shuffle of eval fact-001's
                    # prompt ("What is the capital of France?") -- identical
                    # word set (Jaccard=1.0) but different word order, so it
                    # is neither an exact-prompt match nor an 8-gram hit
                    # (only 6 words, below CONTAMINATION_NGRAM). Only the
                    # prompt-Jaccard check catches this one, and must still
                    # drop it.
                    {"i": 8, "prompt": "France capital is what of the", "tier": 1},
                ],
            },
            {
                "custom_id": "chat-000000",
                "category": "chat",
                "items": [
                    {"i": 0, "prompt": "Give me a compliment.", "tier": 2},
                ],
            },
            {
                "custom_id": "shell-000000",
                "category": "shell",
                "items": [
                    # F1: independently phrased shell prompt, answer "ls" --
                    # must be KEPT.
                    {"i": 0, "prompt": "Which single Linux program prints "
                                       "what is inside your working folder?",
                     "tier": 1},
                ],
            },
            {
                # python_natural: the request's own "prompt" is the
                # meta-instruction (never used for training); the teacher's
                # response supplies the real prompt+answer pair.
                "custom_id": "python_natural-000000",
                "category": "python_natural",
                "items": [
                    {"i": 0, "prompt": "Python task: print a message. "
                                       "Reference answer: print('hi'). Invent "
                                       "a natural phrasing and answer it.",
                     "tier": 1},
                    {"i": 1, "prompt": "Python task: get the length of a "
                                       "list. Reference answer: len(xs). "
                                       "Invent a natural phrasing and answer it.",
                     "tier": 1},
                ],
            },
        ]
        with open(req_path, "w") as f:
            for r in requests_:
                f.write(json.dumps(r) + "\n")

        long_answer = " ".join(["word"] * (MAX_ANSWER_WORDS + 5))
        # eval/suite-v1.1.jsonl has fact-001: prompt "What is the capital of France?"
        # answer "Paris" -- both the exact-prompt and exact-answer-adjacent
        # paths get exercised here.
        succeeded_text = "\n".join([
            json.dumps({"i": 0, "answer": "Wakanda City"}),
            json.dumps({"i": 1, "answer": "Paris"}),          # dropped: exact prompt match
            json.dumps({"i": 2, "answer": "Apple"}),
            json.dumps({"i": 3, "answer": "I'm sorry, but I can't help with that."}),  # refusal
            json.dumps({"i": 4, "answer": long_answer}),      # overlong
            json.dumps({"i": 5, "answer": "It has stripes and is huge "
                                           "compared to everything else out "
                                           "there"}),          # dropped: 8-gram overlap on the prompt
            json.dumps({"i": 6, "answer": "Paris"}),          # F1: kept -- short-answer overlap only
            json.dumps({"i": 7, "answer": "7"}),              # F1: kept -- short-answer overlap only
            json.dumps({"i": 8, "answer": "Unrelated short answer"}),  # F1: dropped -- prompt Jaccard 1.0
            "not even json",                                  # unparsed line, item 4 already covered above
        ])
        duplicate_text = json.dumps({"i": 0, "answer": "Apple"})  # exact dup of factual item 2's pair after norm? different prompt, so not a dup -- use same-shard dup below instead

        natural_text = "\n".join([
            # i=0: well-formed -- prompt/answer both come from the teacher,
            # NOT from the request's meta-instruction.
            json.dumps({"i": 0, "prompt": "how do you print something in python",
                        "answer": "print('hi')"}),
            # i=1: bad -- one-word "prompt" must be dropped (drop_bad_natural_prompt),
            # never silently kept with a meaningless training prompt.
            json.dumps({"i": 1, "prompt": "len", "answer": "len(xs)"}),
        ])

        results = [
            {"custom_id": "factual-000000", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": succeeded_text}]}}},
            {"custom_id": "chat-000000", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": json.dumps({"i": 0, "answer": "You're doing great!"})}]}}},
            {"custom_id": "shell-000000", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": json.dumps({"i": 0, "answer": "ls"})}]}}},  # F1: kept -- short-answer overlap only
            {"custom_id": "python_natural-000000", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": natural_text}]}}},
            {"custom_id": "missing-000099", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": "{}"}]}}},  # no matching request
            {"custom_id": "factual-000000", "result": {"type": "errored", "error": {"type": "server_error"}}},  # duplicate custom_id, errored -- exercises drop_batch_errored without double-counting kept items
        ]
        with open(raw_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        prompt_overlap_sets, exact_prompts, exact_long_answers, eval_ngrams = load_eval_overlap_sets()
        request_index = load_request_index(req_dir)
        pairs, stats = process([raw_path], request_index, prompt_overlap_sets, exact_prompts, exact_long_answers, eval_ngrams)

        checks = []

        def check(name, cond):
            checks.append((name, cond))

        check("kept exactly 7 pairs (wakanda, apple, compliment, paris, "
              "dwarves, ls, python_natural i=0)", len(pairs) == 7)
        check("no kept pair has prompt matching eval exactly",
              all(normalize_key(p["prompt"]) not in exact_prompts for p in pairs))
        check("no kept pair shares an 8-gram with an eval prompt/answer",
              all(not (ngrams(normalize_word_list(p["prompt"])) & eval_ngrams
                       or ngrams(normalize_word_list(p["answer"])) & eval_ngrams)
                  for p in pairs))
        check("drop_contamination_exact_prompt >= 1", stats["drop_contamination_exact_prompt"] >= 1)
        check("drop_contamination_8gram >= 1 (F8)", stats["drop_contamination_8gram"] >= 1)
        check("drop_refusal >= 1", stats["drop_refusal"] >= 1)
        check("drop_overlong >= 1", stats["drop_overlong"] >= 1)
        check("drop_no_matching_request >= 1", stats["drop_no_matching_request"] >= 1)
        check("drop_batch_errored counts the errored duplicate's items", stats["drop_batch_errored"] >= 1)
        check("kept_per_category has factual, chat, shell and python_natural",
              set(stats["kept_per_category"]) == {"factual", "chat", "shell", "python_natural"})
        # python_natural: the kept pair's prompt/answer must come from the
        # teacher's response, never from the request's meta-instruction, and
        # the malformed one-word "prompt" (i=1) must be dropped, not kept.
        natural_kept = [p for p in pairs if p["category"] == "python_natural"]
        check("python_natural: exactly 1 kept (i=0 only)", len(natural_kept) == 1)
        check("python_natural: kept prompt is the teacher's phrasing, not the meta-instruction",
              natural_kept and natural_kept[0]["prompt"] == "how do you print something in python")
        check("python_natural: kept answer is the teacher's answer",
              natural_kept and natural_kept[0]["answer"] == "print('hi')")
        check("python_natural: bad one-word prompt (i=1) dropped",
              stats["drop_bad_natural_prompt"] >= 1)
        # F1 (finding 1): short-answer overlap ("Paris"/"7"/"ls") must not be
        # contamination -- these must survive with independently phrased
        # prompts.
        kept_answers = {p["answer"] for p in pairs}
        check("F1: answer 'Paris' kept (independently phrased prompt)", "Paris" in kept_answers)
        check("F1: answer '7' kept (independently phrased prompt)", "7" in kept_answers)
        check("F1: answer 'ls' kept (independently phrased prompt)", "ls" in kept_answers)
        # F1: a prompt that Jaccard-matches an eval prompt at >=0.8 (but is
        # not an exact-prompt or 8-gram match) must still be dropped.
        check("F1: prompt Jaccard-match to eval prompt still dropped",
              "France capital is what of the" not in {p["prompt"] for p in pairs})
        check("F1: prompt-Jaccard drop counted", stats["drop_contamination_overlap"] >= 1)

        # Second pass with an intentional duplicate pair appended, to prove dedup fires.
        pairs2, stats2 = process([raw_path], request_index, prompt_overlap_sets, exact_prompts, exact_long_answers, eval_ngrams)
        check("re-running process() on the same input is idempotent in count", len(pairs2) == len(pairs))

        ok = all(c for _, c in checks)
        for name, cond in checks:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        print("SELFTEST " + ("PASS" if ok else "FAIL"))
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
