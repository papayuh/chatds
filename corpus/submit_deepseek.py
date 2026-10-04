#!/usr/bin/env python3
"""D13 teacher-data pipeline, stage 2 (DeepSeek variant): submit request
shards straight to the DeepSeek chat-completions API (no batch-job object --
this is a bounded pool of real-time async calls), track token usage/spend,
write results in the the same shape the old batch fetch produced so
postprocess.py needs ZERO changes to read either teacher's output.

model: deepseek-v4-flash (non-reasoning variant, D13a). Explicitly disables
thinking mode ("thinking": {"type": "disabled"}) on every request -- it is
ON by default on this endpoint and reasoning_effort does not turn it off.

Hard runtime guards (both abort the whole run, not just the one request):
  - any response with reasoning_content or a nonzero reasoning-token count
    means the disable flag stopped working -- abort rather than pay for a
    broken night.
  - cumulative cost for this --out file crosses --cost-ceiling (default
    $30, well under the $105 D13 cap) -- abort rather than run unattended
    past a sane multiple of the smoke estimate.
Either abort leaves corpus/results/ABORTED.txt with the reason, and the
process exits non-zero. Resumable as always: rerun with the same --out and
it picks up where it left off.

Needs `requests` (already installed) and DEEPSEEK_API_KEY in .env at the
repo root. Refuses to run without a key. NEVER logs/prints/commits the key;
Authorization headers are never included in any printed/error output.

(The old Anthropic Batch API submitter was removed from the tree.)

Usage:
    python3 submit_deepseek.py --selftest                  # offline converter + guard self-test, no network
    python3 submit_deepseek.py --smoke                      # requests/ shard(s) -> corpus/results/smoke.jsonl
    python3 submit_deepseek.py --all --out results/full.jsonl
    python3 submit_deepseek.py --shard shard-003.jsonl --out results/shard-003.jsonl
"""
import argparse
import asyncio
import glob
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common

REQUESTS_DIR = os.path.join(HERE, "requests")
RESULTS_DIR = os.path.join(HERE, "results")
BATCHES_JSON = os.path.join(HERE, "batches.json")
ENV_FILE = os.path.join(REPO_ROOT, ".env")
ABORT_MARKER = os.path.join(RESULTS_DIR, "ABORTED.txt")

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_CONCURRENCY = 8
DEFAULT_COST_CEILING = 30.0
MAX_RETRIES = 6
REQUEST_TIMEOUT_S = 120

# Off-peak DeepSeek pricing (Diego, 2026-08-31). Cache-hit input is ~30x
# cheaper than a miss, so track them separately, not just prompt_tokens.
PRICE_IN_MISS_PER_MTOK = 0.22
PRICE_IN_HIT_PER_MTOK = 0.007
PRICE_OUT_PER_MTOK = 0.66


def load_api_key():
    """Read DEEPSEEK_API_KEY from .env (or the environment). Never logs it."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    if not os.path.exists(ENV_FILE):
        return None
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "DEEPSEEK_API_KEY":
                return v.strip().strip('"').strip("'")
    return None


def require_key():
    key = load_api_key()
    if not key:
        print(
            "No DEEPSEEK_API_KEY found in .env or the environment.\n"
            f"Add a line to {ENV_FILE}:\n"
            "  DEEPSEEK_API_KEY=sk-...\n"
            "Refusing to submit anything to the network without a key.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def to_chat_payload(record):
    """Convert one Anthropic-batch-shaped request record (as emitted by
    gen_requests.py: {"custom_id", "category", "items":[{"i","prompt","tier"}]})
    into an OpenAI-compatible chat-completions payload. Same system/user
    content and max_tokens as the old Batch API converter -- no
    temperature key (spec.md never sets one for the Anthropic path either,
    so omitting it here keeps both teachers on the API default). No
    reasoning/thinking key beyond the one that turns it off: thinking mode
    is ON by default on this endpoint (confirmed by smoke run 1, all 500/500
    responses reasoned) and reasoning_effort alone does NOT disable it --
    per api-docs.deepseek.com/guides/thinking_mode/ the only switch is
    "thinking": {"type": "disabled"} in the request body.

    system prompt and max_tokens are looked up per category (common.
    system_prompt_for/max_tokens_for) so the python_natural addendum (which
    asks the teacher to invent the phrasing too, not just answer one) can use
    its own longer system prompt/budget without touching any other
    category's behavior -- both helpers fall back to SYSTEM_PROMPT/MAX_TOKENS
    for every category that doesn't override them."""
    category = record.get("category")
    return {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": common.system_prompt_for(category)},
            {"role": "user", "content": common.user_message(record["items"])},
        ],
        "max_tokens": common.max_tokens_for(category),
        "thinking": {"type": "disabled"},
    }


def load_shard(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_ids(out_path):
    """Resumability: custom_ids already present in the output file are
    skipped on re-run -- makes the whole submit crash-safe."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    done.add(json.loads(line)["custom_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def seed_usage(out_path, usage):
    """Fold usage from any already-written results into `usage` so cost
    tracking (and the cost-ceiling check) stays correct across a restart --
    e.g. a concurrency-8 pilot handing off to a concurrency-16 continuation,
    both appending to the same --out file."""
    if not os.path.exists(out_path):
        return
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = rec.get("usage")
            if u:
                usage.add(u)


def detects_reasoning(message, usage_dict):
    """True if a response shows any sign thinking mode fired despite the
    "thinking":{"type":"disabled"} flag -- the hard-abort trigger. Pulled out
    as its own function so the selftest can exercise it without a network
    mock."""
    if message.get("reasoning_content"):
        return True
    reasoning_tokens = (usage_dict.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return bool(reasoning_tokens)


class AbortSignal:
    """Shared flag: any task can trip it, every task checks it before
    spending another request. Not true cancellation of in-flight calls --
    with a small concurrency pool that's an acceptable fail-fast, not a
    fail-instant, and keeps this a one-file diff."""

    def __init__(self):
        self.event = asyncio.Event()
        self.reason = None

    def trigger(self, reason):
        if not self.event.is_set():
            self.reason = reason
            self.event.set()


def load_batches():
    if os.path.exists(BATCHES_JSON):
        with open(BATCHES_JSON) as f:
            return json.load(f)
    return {"spent_estimate_usd": 0.0, "batches": []}


def save_batches(state):
    with open(BATCHES_JSON, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


def estimate_worst_case_cost(records):
    """Worst-case estimate: worst case = no cache
    hits, every response maxes its category's max_tokens. Conservative on
    purpose so the shared $105 cap (D13a: cap unchanged across the teacher
    swap) never gets surprised. Per-record (not average-then-multiply) so a
    mixed-category shard list -- e.g. the python_natural addendum, whose
    system prompt and max_tokens both differ from every other category --
    prices correctly instead of diluting into an average."""
    n = len(records)
    if n == 0:
        return 0.0
    in_cost = 0.0
    out_cost = 0.0
    for r in records:
        category = r.get("category")
        sys_chars = len(common.system_prompt_for(category))
        user_chars = len(common.user_message(r["items"]))
        approx_in_tokens = (sys_chars + user_chars) / 4 + 20
        in_cost += approx_in_tokens / 1_000_000 * PRICE_IN_MISS_PER_MTOK
        out_cost += common.max_tokens_for(category) / 1_000_000 * PRICE_OUT_PER_MTOK
    return in_cost + out_cost


class Usage:
    """Cumulative token/cost tracker, printed as results come in."""

    def __init__(self):
        self.miss_in = 0
        self.hit_in = 0
        self.out = 0
        self.reasoning = 0
        self.n = 0

    def add(self, usage):
        hit = usage.get("prompt_cache_hit_tokens", 0) or 0
        miss = usage.get("prompt_cache_miss_tokens")
        if miss is None:
            miss = max(usage.get("prompt_tokens", 0) - hit, 0)
        self.hit_in += hit
        self.miss_in += miss
        self.out += usage.get("completion_tokens", 0) or 0
        details = usage.get("completion_tokens_details") or {}
        self.reasoning += details.get("reasoning_tokens", 0) or 0
        self.n += 1

    @property
    def cost(self):
        return (
            self.miss_in / 1_000_000 * PRICE_IN_MISS_PER_MTOK
            + self.hit_in / 1_000_000 * PRICE_IN_HIT_PER_MTOK
            + self.out / 1_000_000 * PRICE_OUT_PER_MTOK
        )

    def line(self):
        return (
            f"[{self.n} done] in(miss)={self.miss_in} in(hit)={self.hit_in} "
            f"out={self.out} reasoning_tokens={self.reasoning} cost=${self.cost:.4f}"
        )


def _post_once(key, payload):
    import requests

    return requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )


async def call_one(sem, key, record, usage, out_f, abort, cost_ceiling):
    if abort.event.is_set():
        return
    payload = to_chat_payload(record)
    backoff = 1.0
    for attempt in range(MAX_RETRIES):
        if abort.event.is_set():
            return
        resp = None
        async with sem:
            if abort.event.is_set():
                return
            try:
                resp = await asyncio.to_thread(_post_once, key, payload)
            except Exception as e:  # connection errors, timeouts -- retryable
                resp = None
                last_err = str(e)

        if resp is not None and resp.status_code == 200:
            body = resp.json()
            msg = body["choices"][0].get("message", {})
            text = msg.get("content") or ""
            u = body.get("usage", {})
            usage.add(u)  # count the spend either way -- money's already gone
            if detects_reasoning(msg, u):
                abort.trigger(
                    f"reasoning detected on {record['custom_id']} despite "
                    '"thinking":{"type":"disabled"} -- the disable flag '
                    "stopped working or was ignored. Aborting rather than "
                    "pay for more broken output."
                )
                return
            out_f.write(json.dumps({
                "custom_id": record["custom_id"],
                "result": {
                    "type": "succeeded",
                    "message": {"content": [{"type": "text", "text": text}]},
                },
                "usage": u,
            }) + "\n")
            out_f.flush()
            if usage.n % 25 == 0:
                print(usage.line())
            if usage.cost >= cost_ceiling:
                abort.trigger(
                    f"cost ceiling ${cost_ceiling:.2f} reached "
                    f"(cumulative for this --out file: ${usage.cost:.4f})"
                )
            return

        status = resp.status_code if resp is not None else "conn-error"
        retryable = resp is None or resp.status_code == 429 or resp.status_code >= 500
        if not retryable or attempt == MAX_RETRIES - 1:
            err_text = resp.text[:300] if resp is not None else last_err
            out_f.write(json.dumps({
                "custom_id": record["custom_id"],
                "result": {
                    "type": "errored",
                    "error": {"type": "api_error", "status": str(status), "message": err_text},
                },
            }) + "\n")
            out_f.flush()
            print(f"FAILED {record['custom_id']}: status={status}", file=sys.stderr)
            return

        sleep_s = backoff + random.uniform(0, backoff * 0.25)
        await asyncio.sleep(sleep_s)
        backoff = min(backoff * 2, 60)


async def run(todo, out_path, key, concurrency, usage, abort, cost_ceiling):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    with open(out_path, "a") as out_f:
        await asyncio.gather(
            *(call_one(sem, key, r, usage, out_f, abort, cost_ceiling) for r in todo)
        )
    return usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="offline converter self-test, no network")
    ap.add_argument("--smoke", action="store_true", help="submit whatever shard(s) are in corpus/requests/")
    ap.add_argument("--all", action="store_true", help="submit every shard in corpus/requests/")
    ap.add_argument("--shard", help="submit one shard file by name (in --requests-dir)")
    ap.add_argument("--requests-dir", default=REQUESTS_DIR,
                     help="shard directory (default corpus/requests; the python_natural "
                          "addendum lives in corpus/requests-python-natural)")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "smoke.jsonl"))
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--limit", type=int, default=None, help="cap number of requests (testing)")
    ap.add_argument("--cost-ceiling", type=float, default=DEFAULT_COST_CEILING,
                     help="abort the run if cumulative cost for --out crosses this ($)")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if args.shard:
        paths = [os.path.join(args.requests_dir, args.shard)]
    elif args.all:
        paths = sorted(glob.glob(os.path.join(args.requests_dir, "*.jsonl")))
    else:
        paths = sorted(glob.glob(os.path.join(args.requests_dir, "*.jsonl")))[:1]

    if not paths:
        print(f"no shard files found in {args.requests_dir} -- run gen_requests.py first", file=sys.stderr)
        sys.exit(1)

    key = require_key()

    records = []
    for p in paths:
        records.extend(load_shard(p))

    done_ids = load_done_ids(args.out)
    todo = [r for r in records if r["custom_id"] not in done_ids]
    if args.limit:
        todo = todo[: args.limit]
    skipped = len(records) - len(todo) if not args.limit else len(done_ids)
    print(f"{len(records)} total requests in shard(s), {skipped} already done, {len(todo)} to submit")

    state = load_batches()
    est_cost = estimate_worst_case_cost(todo)
    remaining = common.BUDGET_CAP_USD - state["spent_estimate_usd"]
    if est_cost > remaining:
        print(
            f"REFUSING: worst-case estimate ${est_cost:.2f} exceeds remaining "
            f"budget ${remaining:.2f} (cap ${common.BUDGET_CAP_USD:.2f}, "
            f"spent-estimate so far ${state['spent_estimate_usd']:.2f}).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"worst-case estimate for this run: ${est_cost:.2f} (remaining budget ${remaining:.2f})")

    if not todo:
        print("nothing to do")
        return

    # Seed usage with whatever's already in --out (a prior partial run) so
    # the cost ceiling and reasoning totals stay correct across a restart.
    # Track this invocation's own increment separately for the batches.json
    # ledger, so re-running against the same --out doesn't double-count.
    usage = Usage()
    seed_usage(args.out, usage)
    seed_cost, seed_n = usage.cost, usage.n
    abort = AbortSignal()

    usage = asyncio.run(run(todo, args.out, key, args.concurrency, usage, abort, args.cost_ceiling))
    incremental_cost = usage.cost - seed_cost
    incremental_n = usage.n - seed_n

    print(usage.line())
    print(f"reasoning_tokens billed (cumulative for this --out file): {usage.reasoning}")
    print(f"this invocation: {incremental_n} requests, ${incremental_cost:.4f}")

    if abort.reason:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(ABORT_MARKER, "w") as f:
            f.write(f"ABORTED {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            f.write(f"reason: {abort.reason}\n")
            f.write(f"out file: {args.out}\n")
            f.write(f"cumulative requests completed (this --out file): {usage.n}\n")
            f.write(f"cumulative cost (this --out file): ${usage.cost:.4f}\n")
        print(f"ABORTED: {abort.reason}", file=sys.stderr)
        print(f"marker written to {ABORT_MARKER}", file=sys.stderr)

    state["spent_estimate_usd"] = round(state["spent_estimate_usd"] + incremental_cost, 4)
    state["batches"].append({
        "id": f"deepseek-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        "provider": "deepseek",
        "shard": ",".join(os.path.basename(p) for p in paths),
        "n_requests": incremental_n,
        "estimated_cost_usd": round(incremental_cost, 4),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "aborted" if abort.reason else "ended",
        "fetched": True,
    })
    save_batches(state)
    print(
        f"running total spend across all teachers: "
        f"${state['spent_estimate_usd']:.2f} / ${common.BUDGET_CAP_USD:.2f}"
    )

    if abort.reason:
        sys.exit(2)


# ---------------------------------------------------------------------------
# Self-test: offline, no network. Checks the converter's payload shape (no
# reasoning params, correct model/content/max_tokens), resumability, and the
# usage/cost math against a synthetic response.
# ---------------------------------------------------------------------------

def run_selftest():
    checks = []

    def check(name, cond):
        checks.append((name, cond))

    record = {
        "custom_id": "factual-000000",
        "category": "factual",
        "items": [
            {"i": 0, "prompt": "What is the capital of Panama?", "tier": 1},
            {"i": 1, "prompt": "What currency is used in France?", "tier": 2},
        ],
    }
    payload = to_chat_payload(record)

    check("model is deepseek-v4-flash", payload["model"] == DEEPSEEK_MODEL)
    check("exactly 2 messages (system, user)", len(payload["messages"]) == 2)
    check("system role first", payload["messages"][0]["role"] == "system")
    check("system content matches common.SYSTEM_PROMPT",
          payload["messages"][0]["content"] == common.SYSTEM_PROMPT)
    check("user role second", payload["messages"][1]["role"] == "user")
    check("user content matches common.user_message(items)",
          payload["messages"][1]["content"] == common.user_message(record["items"]))
    check("max_tokens matches common.MAX_TOKENS", payload["max_tokens"] == common.MAX_TOKENS)
    check("no temperature key (matches Anthropic path, which doesn't set one either)",
          "temperature" not in payload)
    check("thinking mode explicitly disabled (the only real switch on this endpoint)",
          payload.get("thinking") == {"type": "disabled"})
    check("no other reasoning/thinking key beyond the disable switch",
          [k for k in payload if "reason" in k.lower() or "think" in k.lower()] == ["thinking"])

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "out.jsonl")
        with open(out_path, "w") as f:
            f.write(json.dumps({"custom_id": "factual-000000", "result": {"type": "succeeded"}}) + "\n")
        done = load_done_ids(out_path)
        check("resumability: already-written custom_id is skippable", done == {"factual-000000"})

    u = Usage()
    u.add({"prompt_tokens": 1000, "prompt_cache_hit_tokens": 400, "prompt_cache_miss_tokens": 600,
           "completion_tokens": 100})
    expected_cost = 600 / 1_000_000 * PRICE_IN_MISS_PER_MTOK + 400 / 1_000_000 * PRICE_IN_HIT_PER_MTOK \
        + 100 / 1_000_000 * PRICE_OUT_PER_MTOK
    check("cost math uses cache-hit/miss split when present", abs(u.cost - expected_cost) < 1e-9)

    u2 = Usage()
    u2.add({"prompt_tokens": 500, "completion_tokens": 50})  # no cache fields at all
    expected_cost2 = 500 / 1_000_000 * PRICE_IN_MISS_PER_MTOK + 50 / 1_000_000 * PRICE_OUT_PER_MTOK
    check("cost math falls back to treating all prompt tokens as cache-miss", abs(u2.cost - expected_cost2) < 1e-9)

    u3 = Usage()
    u3.add({"prompt_tokens": 100, "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 42}})
    check("reasoning_tokens tracked if the API ever reports them", u3.reasoning == 42)

    # Hard-abort guard: detects_reasoning is what call_one checks before
    # writing a result. Exercise it directly (no network mock needed).
    check("detects_reasoning fires on reasoning_content",
          detects_reasoning({"reasoning_content": "hmm let me think"}, {}))
    check("detects_reasoning fires on nonzero reasoning_tokens usage",
          detects_reasoning({}, {"completion_tokens_details": {"reasoning_tokens": 5}}))
    check("detects_reasoning is quiet on a clean response",
          not detects_reasoning({"content": "Paris"}, {"completion_tokens_details": {"reasoning_tokens": 0}}))
    check("detects_reasoning is quiet when usage has no completion_tokens_details at all",
          not detects_reasoning({"content": "Paris"}, {}))

    abort = AbortSignal()
    check("AbortSignal starts untripped", not abort.event.is_set())
    abort.trigger("first reason")
    abort.trigger("second reason should be ignored")
    check("AbortSignal latches the first reason and stays set",
          abort.event.is_set() and abort.reason == "first reason")

    ok = all(c for _, c in checks)
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
