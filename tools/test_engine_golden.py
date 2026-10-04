"""Host-only regression tests: no private assets or emulator needed."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

import engine_golden as golden

GOOD = {"ids": [1885, 276], "stop": "eos", "prompt_tokens": 15,
        "context": "", "text": "  calc"}
TRAILER = "tokens=16\nprompt_tokens=15\ngenerated=2\nstop=eos\nms=123\nstatus=OK\n"


def test_parse_final_trailer_only():
    out = "status=FAIL\ntokens=999\n  calc\n" + TRAILER
    got = golden.parse_result(out, "1885\n276\n", "ctx=\nretrieval_ms=0\n")
    assert got == dict(GOOD, text="status=FAIL\ntokens=999\n  calc")


@pytest.mark.parametrize("out,ids", [
    ("calc\n" + TRAILER.replace("status=OK", "status=FAIL"), "1885 276"),
    ("calc\n" + TRAILER.replace("generated=2", "generated=3"), "1885 276"),
    ("calc\n" + TRAILER.replace("tokens=16", "tokens=17"), "1885 276"),
    ("calc\n" + TRAILER.replace("stop=eos", "stop=cancel"), "1885 276"),
    ("calc\n" + TRAILER, "1885 -1"),
    ("calc\n" + TRAILER, "1885 2048"),
    ("calc\n" + TRAILER + "junk\n", "1885 276"),
    ("status=OK\n", ""),
])
def test_reject_malformed_results(out, ids):
    with pytest.raises(ValueError):
        golden.parse_result(out, ids)


def test_exact_ids_and_eos_no_prefix_tolerance():
    assert golden.compare(GOOD, GOOD) == []
    assert golden.compare(GOOD, dict(GOOD, ids=[1885])) == ["ids"]
    assert golden.compare(GOOD, dict(GOOD, stop="budget")) == ["stop"]
    assert golden.compare(GOOD, dict(GOOD, text="calc")) == ["text"]


def test_fail_class_accepts_only_completion_or_explicit_failure():
    case = {"class": "baseline-fails"}
    golden.validate_actual(GOOD, case)
    golden.validate_actual({"status": "FAIL", "failure": "unsupported input"}, case)
    for actual in ({}, {"status": "FAIL"}, {"status": "FAIL", "failure": ""},
                   {"status": "timeout"}, dict(GOOD, ids=[True])):
        with pytest.raises(ValueError):
            golden.validate_actual(actual, case)
    with pytest.raises(ValueError):
        golden.validate_actual({"status": "FAIL", "failure": "unsupported"}, {})


def test_fixture_set_schema_and_duplicate_names():
    fixture = {"version": 1, "cases": [{"name": "a", "question": "q", "expected": GOOD}]}
    golden.validate_fixture_set(fixture)
    fixture["cases"].append(copy.deepcopy(fixture["cases"][0]))
    with pytest.raises(ValueError):
        golden.validate_fixture_set(fixture)


def test_cli_adapter_checks_every_fixture_and_mismatch_exits_one(tmp_path):
    fixtures = tmp_path / "fixtures.json"
    golden.write_json(fixtures, {"version": 1, "cases": [
        {"name": "a", "question": "first", "expected": GOOD},
        {"name": "b", "question": "again", "expected": GOOD}]})
    adapter = tmp_path / "adapter"
    adapter.write_text("#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
                       f"value={GOOD!r}\n"
                       "request=json.loads(Path(sys.argv[1]).read_text())\n"
                       "if request['name']=='b': value['stop']='bos'\n"
                       "Path(sys.argv[2]).write_text(json.dumps(value))\n")
    adapter.chmod(0o755)
    result = subprocess.run([sys.executable, str(Path(golden.__file__)), "check",
                             "--fixtures", str(fixtures), "--adapter", str(adapter),
                             "--work", str(tmp_path / "work")], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 1
    assert "PASS a" in result.stdout
    assert "FAIL b: stop" in result.stdout
    assert len(list((tmp_path / "work").glob("boot-*"))) == 1  # mismatch artifacts retained


def test_command_timeout_is_always_supplied():
    with mock.patch.object(golden.subprocess, "run") as run:
        golden.command(["true"])
        assert run.call_args.kwargs["timeout"] == 30
        assert run.call_args.kwargs["check"] is True


def test_rom_timeout_cleanup_and_headless_isolation(tmp_path):
    rom = tmp_path / "source.nds"
    rom.write_bytes(b"fake ROM")
    process = mock.Mock()
    # Don't need toolchain, FAT tools, or flatpak in this host test.
    times = iter([0, 2])
    with mock.patch.object(golden, "pack_image"), \
            mock.patch.object(golden, "command"), \
            mock.patch.object(golden.time, "monotonic", side_effect=lambda: next(times)), \
            mock.patch.object(golden.subprocess, "Popen", return_value=process) as launch, \
            mock.patch.object(golden, "kill_owned") as cleanup:
        with pytest.raises(TimeoutError):
            golden.run_rom(rom, "model", "tok", None, {"question": "q"},
                           tmp_path / "private", 1)
    args = launch.call_args.args[0]
    assert "--env=QT_QPA_PLATFORM=offscreen" in args
    assert "QT_QPA_PLATFORM=offscreen" in args[args.index("-c") + 1]
    assert launch.call_args.kwargs["env"]["QT_QPA_PLATFORM"] == "offscreen"
    assert str(tmp_path / "private/boot.nds") == args[-1]
    assert cleanup.call_count == 2
    cleanup.assert_has_calls([mock.call(tmp_path / "private/boot.nds")] * 2)
    process.wait.assert_called_once_with(timeout=5)
    assert (tmp_path / "private/config/melonDS/melonDS.toml").exists()


def test_owned_pid_scan_exact_argv_and_comm(tmp_path):
    def proc(pid, comm, args):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "comm").write_text(comm)
        (directory / "cmdline").write_bytes(b"\0".join(args) + b"\0")
        return directory / "cmdline"
    target = "/private/boot.nds"
    paths = [proc(1, "melonDS\n", [b"melonDS", target.encode()]),
             proc(2, "melonDS\n", [b"melonDS", b"/private/boot.nds.extra"]),
             proc(3, "bash\n", [b"bash", target.encode()]),
             proc(4, "melonDS\n", [b"melonDS", b"/another/boot.nds"])]
    with mock.patch.object(golden.Path, "glob", return_value=paths):
        assert golden.owned_pids(target) == [1]


def test_checked_in_goldens_have_broad_coverage():
    fixture = json.loads(golden.DEFAULT_FIXTURES.read_text())
    golden.validate_fixture_set(fixture)
    cases = fixture["cases"]
    tags = {tag for case in cases for tag in case.get("tags", [])}
    assert {"smoke", "kb", "budget", "near-tie", "length-range", "full-context", "repeated-boot"} <= tags
    by_name = {case["name"]: case for case in cases}
    for name in ("repeat-math-a", "repeat-math-b"):
        assert by_name[name]["expected"] == by_name["smoke-math"]["expected"]
    assert all(len(case["expected"]["ids"]) > 0
               for case in cases if "smoke" in case.get("tags", []))
    assert any(case.get("class") == "baseline-fails" for case in cases)
    assert any(case.get("host", {}).get("near_tie_generation_positions") for case in cases)
    full_context = [case["host"]["prompt_tokens"] for case in cases if "full-context" in case.get("tags", [])]
    assert full_context == [255, 256]
    for case in cases:
        host = case.get("host")
        if not host:
            continue
        assert host["source"] == "host-float32-not-device"
        assert len(host["top2"]) == host["tensor_shape"][0]
        for row in host["top2"]:
            assert len(row["ids"]) == len(row["values"]) == 2
            assert row["margin"] == row["values"][0] - row["values"][1]
        assert host["near_tie_positions"] == [i for i, row in enumerate(host["top2"]) if row["margin"] < .1]


def test_session_adapter_invoked_once_for_repeated_calls(tmp_path):
    fixtures = tmp_path / "fixtures.json"
    golden.write_json(fixtures, {"version": 1, "cases": [
        {"name": "a", "question": "first", "expected": GOOD},
        {"name": "b", "question": "intervening", "expected": GOOD},
        {"name": "c", "question": "first", "expected": GOOD}]})
    adapter = tmp_path / "adapter"
    counter = tmp_path / "calls.txt"
    # Mock outputs test plumbing only, not engine parity.
    adapter.write_text("#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
                       f"counter=Path({str(counter)!r})\n"
                       "counter.write_text(counter.read_text()+'call\\n' if counter.exists() else 'call\\n')\n"
                       "request=json.loads(Path(sys.argv[1]).read_text())\n"
                       f"value={GOOD!r}\n"
                       "Path(sys.argv[2]).write_text(json.dumps([value for _ in request['cases']]))\n")
    adapter.chmod(0o755)
    result = subprocess.run([sys.executable, str(Path(golden.__file__)), "check",
                             "--fixtures", str(fixtures), "--session-adapter", str(adapter),
                             "--work", str(tmp_path / "work")], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert counter.read_text() == "call\n"
    assert result.stdout == "PASS a\nPASS b\nPASS c\n"
    assert not list((tmp_path / "work").iterdir())


def test_check_records_adapter_crash_and_continues_with_inputs_only(tmp_path):
    fixtures = tmp_path / "fixtures.json"
    golden.write_json(fixtures, {"version": 1, "cases": [
        {"name": "a", "question": "q", "expected": dict(GOOD, context="oracle ctx"),
         "host": {"prompt": "C: oracle ctx\nQ: q\nA:"}},
        {"name": "b", "question": "q", "expected": GOOD},
        {"name": "c", "question": "q", "steps": 8, "instruct": False, "expected": GOOD}]})
    adapter = tmp_path / "adapter"
    # Echoing any supplied oracle context would pass "a"; inputs-only must not.
    adapter.write_text("#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
                       "request=json.loads(Path(sys.argv[1]).read_text())\n"
                       "assert set(request) == {'name','question','instruct','steps'}, request\n"
                       "if request['name']=='b': sys.exit(3)\n"
                       f"value={GOOD!r}\n"
                       "value['context']=request.get('expected',{}).get('context','')\n"
                       "Path(sys.argv[2]).write_text(json.dumps(value))\n")
    adapter.chmod(0o755)
    result = subprocess.run([sys.executable, str(Path(golden.__file__)), "check",
                             "--fixtures", str(fixtures), "--adapter", str(adapter),
                             "--work", str(tmp_path / "work")], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 1
    assert "FAIL a: context" in result.stdout
    assert "FAIL b: CalledProcessError" in result.stdout
    assert "PASS c" in result.stdout


def test_adapter_check_rejects_supplied_unpinned_asset_before_running(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"pinned")
    fixtures = tmp_path / "fixtures.json"
    golden.write_json(fixtures, {"version": 1, "provenance": {"model_sha256": golden.sha256(model)},
                                 "cases": [{"name": "a", "question": "q", "expected": GOOD}]})
    adapter = tmp_path / "adapter"
    adapter.write_text("#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
                       f"Path(sys.argv[2]).write_text(json.dumps({GOOD!r}))\n")
    adapter.chmod(0o755)

    def check(path):
        return subprocess.run([sys.executable, str(Path(golden.__file__)), "check",
                               "--fixtures", str(fixtures), "--adapter", str(adapter),
                               "--model", str(path), "--work", str(tmp_path / "work")],
                              capture_output=True, text=True, timeout=15)
    assert check(model).returncode == 0
    wrong = tmp_path / "wrong.bin"
    wrong.write_bytes(b"other")
    result = check(wrong)
    assert result.returncode == 2
    assert "model does not match the frozen asset identity" in result.stderr
    assert "PASS" not in result.stdout
