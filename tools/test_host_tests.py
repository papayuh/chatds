"""Dependency selection never opens optional upstream implementation files."""
import run_host_tests as host


def test_missing_dependencies_skip_only_dependent_tests(tmp_path):
    ignored, deselected = host.exclusions(tmp_path, torch_available=False)
    assert set(ignored) == set(host.REFERENCE_TESTS) | {"train/test_qat4.py"}
    assert not deselected
    assert all(reason for reason in ignored.values())


def test_available_reference_dependencies_enable_their_tests(tmp_path):
    for paths in host.REFERENCE_TESTS.values():
        for name in paths:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    ignored, deselected = host.exclusions(tmp_path, torch_available=True)
    assert ignored == {}
    assert deselected == {}


def test_missing_torch_skips_training_but_not_corpus_with_reference_present(tmp_path):
    for paths in host.REFERENCE_TESTS.values():
        for name in paths:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    ignored, _ = host.exclusions(tmp_path, torch_available=False)
    assert "corpus/test_chatds_export.py" not in ignored
    assert "train/test_train_instruct.py" in ignored


def test_standalone_tokenizer_and_engine_hooks_execute_not_just_collect(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    path = tmp_path / "ds/tokenizer/test_tok.py"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(host, "ROOT", tmp_path)
    monkeypatch.setattr(host, "exclusions", lambda: ({}, {}))
    monkeypatch.setattr(sys, "argv", ["run_host_tests.py"])
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(host.subprocess, "run", run)
    assert host.main() == 0
    assert calls[0][0][:3] == [sys.executable, "-m", "pytest"]
    assert calls[1][0] == [sys.executable, "ds/tokenizer/test_tok.py"]
    assert calls[2][0] == ["make", "-C", "clean", "test"]
    assert calls[3][0] == ["make", "-C", "clean/ui", "test"]
    assert len(calls) == 4
    assert all(call[1]["cwd"] == tmp_path and call[1]["timeout"] > 0 for call in calls)
