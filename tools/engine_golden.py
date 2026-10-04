#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Black-box ChatDS parity fixtures and isolated, headless melonDS execution.

No engine/tokenizer implementation is imported. Device protocol is the existing
run.txt -> out.txt/ids.txt/ctx.txt contract. See docs/ENGINE-GOLDEN.md.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = ROOT / "ds/engine-golden/fixtures.json"
APP = "net.kuribo64.melonDS"


def command(args, timeout=30, **kwargs):
    """All external commands, including setup and mtools, have a deadline."""
    return subprocess.run([str(a) for a in args], check=True, timeout=timeout,
                          **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parse_result(out, ids, context="", *, vocab_size=2048, allow_context=False):
    """Parse only the final trailer; answer text can contain metadata-looking lines."""
    lines = out.splitlines()
    keys = ["tokens", "prompt_tokens", "generated", "stop", "ms", "status"]
    if len(lines) < 6 or any(not line.startswith(key + "=")
                             for key, line in zip(keys, lines[-6:])):
        raise ValueError("missing complete six-line ChatDS trailer")
    meta = dict(line.split("=", 1) for line in lines[-6:])
    tokens = [int(value) for value in ids.split()]
    n_prompt, generated = int(meta["prompt_tokens"]), int(meta["generated"])
    stops = ("bos", "eos", "budget", "context", "cancelled") if allow_context else ("bos", "eos", "budget")
    if (meta["status"] != "OK" or meta["stop"] not in stops
            or n_prompt < 1 or generated != len(tokens)
            or int(meta["tokens"]) != n_prompt - 1 + generated
            or any(token < 0 or token >= vocab_size for token in tokens)):
        raise ValueError("invalid result metadata/token count")
    ctx_lines = context.splitlines()
    if ctx_lines and not ctx_lines[0].startswith("ctx="):
        raise ValueError("invalid context record")
    return {"ids": tokens, "stop": meta["stop"], "prompt_tokens": n_prompt,
            "context": ctx_lines[0][4:] if ctx_lines else "",
            "text": "\n".join(lines[:-6])}


def owned_pids(rom):
    """Exact argv + comm match: never kill another worker's emulator/shell."""
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if ((path.parent / "comm").read_text().strip() == "melonDS"
                    and os.fsencode(rom) in path.read_bytes().split(b"\0")):
                found.append(int(path.parent.name))
        except (OSError, ProcessLookupError):
            pass
    return found


def kill_owned(rom):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in owned_pids(rom):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 2
        while owned_pids(rom) and time.monotonic() < deadline:
            time.sleep(0.05)
    if owned_pids(rom):
        raise RuntimeError("owned melonDS process did not exit")


def pack_image(image, files):
    size = max(65536, (sum(Path(p).stat().st_size for p in files.values())
                      + 16 * 1024**2 + 1023) // 1024)
    command(["mkfs.vfat", "-F", "32", "-C", image, size],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    directories = {str(Path(name).parent) for name in files} - {"."}
    for directory in sorted(directories, key=lambda d: (d.count("/"), d)):
        command(["mmd", "-i", image, "::/" + directory], capture_output=True)
    for name, path in files.items():
        command(["mcopy", "-i", image, path, "::/" + name], capture_output=True)


def run_rom(rom, model, tokenizer, kb, case, work, timeout, card_dir="chatds", script=None):
    """Use a private config, ROM copy, FAT image and preload shim per boot."""
    if card_dir != "chatds":
        raise ValueError("card_dir must be chatds")
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    rom_copy = work / "boot.nds"
    shutil.copyfile(rom, rom_copy)
    config = work / "config/melonDS"
    config.mkdir(parents=True)
    for name in ("home", "cache"):
        (work / name).mkdir()
    question = case["question"]
    if "\n" in question or "\r" in question or "\0" in question:
        raise ValueError("run.txt prompt must be a single line")
    steps = case.get("steps", 64)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    runfile = work / "run.txt"
    runfile.write_text(f"steps={steps}\nkbd={int(script is not None)}\ninstruct={int(case.get('instruct', True))}\nprompt={question}\n")
    files = {f"{card_dir}/model.bin": Path(model), f"{card_dir}/tok.bin": Path(tokenizer),
             f"{card_dir}/run.txt": runfile}
    if kb:
        files[f"{card_dir}/kb.bin"] = Path(kb)
    if script is not None:
        files[f"{card_dir}/ui-script.txt"] = Path(script)
    image = work / "sd.img"
    pack_image(image, files)
    # JSON quoted strings are also valid TOML basic strings for filesystem paths.
    (config / "melonDS.toml").write_text(f"""LimitFPS = false
[Emu]
DirectBoot = true
ExternalBIOSEnable = false
[JIT]
Enable = true
[DLDI]
Enable = true
ImagePath = {json.dumps(str(image))}
ImageSize = 0
ReadOnly = false
FolderSync = false
FolderPath = ""
""")
    shim = work / "image-io.so"
    command(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared",
             "-fPIC", ROOT / "tools/melon-image-io.c", "-ldl", "-o", shim])
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    launch_script = ('export XDG_CONFIG_HOME="$1" HOME="$2" XDG_CACHE_HOME="$3" '
              'QT_QPA_PLATFORM=offscreen; exec melonDS "$4"')
    args = ["flatpak", "run", "--filesystem=" + str(work),
            "--env=QT_QPA_PLATFORM=offscreen", "--env=LD_PRELOAD=" + str(shim),
            "--env=MELON_UNBUFFERED_IMAGE=" + str(image), "--command=sh", APP,
            "-c", launch_script, "sh", str(work / "config"), str(work / "home"),
            str(work / "cache"), str(rom_copy)]
    output = work / "out.txt"
    completion = "ui-result.txt" if script is not None else "out.txt"
    polled = work / completion
    deadline = time.monotonic() + timeout
    with (work / "emulator.log").open("w") as log:
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=log)
        try:
            while time.monotonic() < deadline:
                remaining = max(0.01, min(5, deadline - time.monotonic()))
                result = subprocess.run(["mcopy", "-o", "-i", str(image),
                                         f"::/{card_dir}/{completion}", str(polled)],
                                        capture_output=True, timeout=remaining)
                if result.returncode == 0:
                    text = polled.read_text()
                    if text.endswith("status=FAIL\n"):
                        if case.get("class") == "baseline-fails":
                            return {"status": "FAIL", "failure": text}
                        raise RuntimeError("ROM reported FAIL; see " + str(output))
                    if text.endswith("status=OK\n"):
                        break
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
            else:
                raise TimeoutError(f"ROM timed out after {timeout}s; see {work / 'emulator.log'}")
            for name in ("ids", "ctx", "out"):
                command(["mcopy", "-o", "-i", image, f"::/{card_dir}/{name}.txt",
                         work / (name + ".txt")], capture_output=True)
            if f"[melon-image] unbuffered: {image}" not in (work / "emulator.log").read_text():
                raise RuntimeError("image I/O shim was not activated")
            vocab = 2048
            if card_dir == "chatds":
                with Path(model).open('rb') as model_file:
                    vocab = struct.unpack_from('<I', model_file.read(40), 36)[0]
            return parse_result(output.read_text(), (work / "ids.txt").read_text(),
                                (work / "ctx.txt").read_text(), vocab_size=vocab,
                                allow_context=card_dir == "chatds")
        finally:
            kill_owned(rom_copy)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            # Flatpak startup can outlive the first scan after a very short
            # timeout. Recheck once the launcher has also been reaped.
            kill_owned(rom_copy)


def compare(expected, actual):
    keys = ("ids", "stop", "prompt_tokens", "context", "text")
    return [key for key in keys if actual.get(key) != expected.get(key)]


def validate_actual(actual, case):
    """Adapters must prove completed inference or an explicit clean failure."""
    if case.get("class") == "baseline-fails" and actual.get("status") == "FAIL":
        if not isinstance(actual.get("failure"), str) or not actual["failure"]:
            raise ValueError("clean failure must include a diagnostic")
        return
    if (actual.get("status", "OK") != "OK"
            or not isinstance(actual.get("ids"), list)
            or any(type(i) is not int or not 0 <= i < 2048 for i in actual["ids"])
            or actual.get("stop") not in ("bos", "eos", "budget")
            or type(actual.get("prompt_tokens")) is not int
            or not 1 <= actual["prompt_tokens"] <= 256
            or len(actual["ids"]) > case.get("steps", 64)
            or actual["prompt_tokens"] - 1 + len(actual["ids"]) > 256
            or (actual["stop"] == "budget" and len(actual["ids"]) != case.get("steps", 64))
            or (actual["stop"] != "budget" and len(actual["ids"]) >= case.get("steps", 64))
            or not isinstance(actual.get("context"), str)
            or not isinstance(actual.get("text"), str)):
        raise ValueError("invalid completed result")


def adapter_request(case, args):
    """Adapters get inputs only, never expected.* or host.*, so context is tested."""
    request = {"name": case["name"], "question": case["question"],
               "instruct": case.get("instruct", True), "steps": case.get("steps", 64)}
    for name in ("model", "tokenizer", "kb"):
        if getattr(args, name):
            request[name] = str(getattr(args, name).resolve())
    return request


def validate_fixture_set(fixtures):
    if fixtures.get("version") != 1 or not fixtures.get("cases"):
        raise ValueError("expected nonempty version 1 fixture set")
    names = set()
    for case in fixtures["cases"]:
        if (not isinstance(case.get("name"), str) or not case["name"]
                or case["name"] in names):
            raise ValueError("duplicate/missing fixture name")
        names.add(case["name"])
        if case.get("class") not in (None, "baseline-fails"):
            raise ValueError("unknown fixture class")
        if not isinstance(case.get("question"), str) or not isinstance(case.get("expected"), dict):
            raise ValueError("fixture missing question/expected result")
        if case.get("class") == "baseline-fails" and case["expected"].get("status") != "FAIL":
            raise ValueError("baseline-fails class must record an observed baseline failure")
        validate_actual(case["expected"], case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "freeze", "check"))
    parser.add_argument("--rom", type=Path)
    parser.add_argument("--card-dir", choices=("chatds",), default="chatds")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--kb", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--no-instruct", action="store_true", help="run: send a raw prompt")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--cases", type=Path, help="freeze: input cases JSON")
    parser.add_argument("--output", type=Path, help="run: captured normalized JSON")
    parser.add_argument("--work", type=Path, default=ROOT / "build/engine-golden")
    parser.add_argument("--timeout", type=float, default=120)
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument("--adapter", type=Path,
                               help="check: executable REQUEST.json RESULT.json for alternative build protocols")
    adapter_group.add_argument("--session-adapter", type=Path,
                               help="check: one executable FIXTURES.json RESULTS.json; shared-process repeated calls")
    args = parser.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("timeout must be finite and positive")
    if not (args.adapter or args.session_adapter) and not all((args.rom, args.model, args.tokenizer)):
        parser.error("--rom, --model and --tokenizer required unless checking with an adapter")
    if (args.adapter or args.session_adapter) and args.mode != "check":
        parser.error("adapters only supported for check")
    if args.mode == "freeze" and not args.cases:
        parser.error("freeze requires --cases")
    if args.mode == "run" and args.question is None:
        parser.error("run requires --question")
    if args.mode == "run":
        cases = [{"name": "single", "question": args.question, "steps": args.steps,
                  "instruct": not args.no_instruct}]
        fixtures = None
    elif args.mode == "freeze":
        cases = json.loads(args.cases.read_text())
        fixtures = {"version": 1, "provenance": {}, "cases": cases}
        # Freeze may never overwrite an existing golden by accident.
        if args.fixtures.exists():
            parser.error("freeze destination exists; choose a new --fixtures path")
    else:
        fixtures = json.loads(args.fixtures.read_text())
        validate_fixture_set(fixtures)
        cases = fixtures["cases"]
        provenance = fixtures.get("provenance", {})
        for name in ("model", "tokenizer", "kb"):
            path = getattr(args, name)
            if path is None and (args.adapter or args.session_adapter):
                continue
            if (sha256(path) if path else None) != provenance.get(name + "_sha256"):
                parser.error(f"{name} does not match the frozen asset identity")
    args.work.mkdir(parents=True, exist_ok=True)
    failures = []
    session_results = None
    if args.session_adapter:
        session = Path(tempfile.mkdtemp(prefix="session-", dir=args.work)).resolve()
        request, output = session / "fixtures.json", session / "results.json"
        write_json(request, {"version": fixtures["version"],
                             "cases": [adapter_request(case, args) for case in cases]})
        try:
            command([args.session_adapter.resolve(), request, output], timeout=args.timeout,
                    env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
            session_results = json.loads(output.read_text())
            if not isinstance(session_results, list) or len(session_results) != len(cases):
                raise ValueError("session adapter must return one result per fixture, in order")
        except Exception:
            print(f"Artifacts retained: {session}", flush=True)
            raise
    for index, case in enumerate(cases):
        work = Path(tempfile.mkdtemp(prefix="boot-", dir=args.work)).resolve()
        # Retain logs/images on any failure; remove only successful boots.
        try:
            if session_results is not None:
                actual = session_results[index]
                write_json(work / "result.json", actual)
            elif args.adapter:
                request, output = work / "request.json", work / "result.json"
                write_json(request, adapter_request(case, args))
                command([args.adapter.resolve(), request, output], timeout=args.timeout,
                        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
                actual = json.loads(output.read_text())
            else:
                actual = run_rom(args.rom, args.model, args.tokenizer, args.kb,
                                 case, work, args.timeout, card_dir=args.card_dir)
            validate_actual(actual, case)
            if args.mode == "freeze":
                if case.get("class") == "baseline-fails" and actual.get("status") != "FAIL":
                    case.pop("class")  # A successful baseline is always a parity target.
                case["expected"] = actual
                write_json(args.work / "capture-checkpoint.json", fixtures)
            elif args.mode == "check":
                mismatch = ([] if case.get("class") == "baseline-fails"
                            else compare(case["expected"], actual))
                if mismatch:
                    failures.append(case["name"])
                    print(f"FAIL {case['name']}: {','.join(mismatch)}", flush=True)
            else:
                if args.output:
                    write_json(args.output, actual)
                print(json.dumps(actual), flush=True)
            if args.mode != "run" and case["name"] not in failures:
                print(f"PASS {case['name']}", flush=True)
            if case["name"] not in failures:
                shutil.rmtree(work)
        except Exception as exc:
            if args.mode != "check":
                print(f"Artifacts retained: {work}", flush=True)
                raise
            failures.append(case["name"])
            print(f"FAIL {case['name']}: {type(exc).__name__}: {exc}", flush=True)
            print(f"Artifacts retained: {work}", flush=True)
    if args.mode == "freeze":
        fixtures["provenance"] = {"rom_sha256": sha256(args.rom),
                                  "model_sha256": sha256(args.model),
                                  "tokenizer_sha256": sha256(args.tokenizer),
                                  "kb_sha256": sha256(args.kb) if args.kb else None,
                                  "device": "melonDS 1.1; offscreen; DS; direct boot; JIT",
                                  "repeats": "separate boots; not same-process KV reset coverage"}
        validate_fixture_set(fixtures)
        write_json(args.fixtures, fixtures)
    if args.session_adapter and not failures:
        shutil.rmtree(session)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
