import json
import subprocess
import sys
from mosi import Sitter
from mosi.environment import capture_environment


def _git(cwd, *a):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=cwd, check=True, capture_output=True)


def _repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "init")
    monkeypatch.setattr(sys, "argv", [str(repo / "train.py"), "--lr", "0.1"])
    return repo


def test_env_json_records_command_and_clean_git(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    Sitter(tmp_path / "runs" / "a").finish()
    env = json.loads((tmp_path / "runs" / "a" / "launch" / "env.json").read_text())
    assert env["command"]["argv"][1:] == ["--lr", "0.1"]
    assert env["git"]["dirty"] is False and len(env["git"]["commit"]) == 40
    assert not (tmp_path / "runs" / "a" / "launch" / "git_diff.patch").exists()


def test_dirty_tree_saves_diff_and_untracked(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    (repo / "train.py").write_text("x = 2\n")
    (repo / "new.py").write_text("")
    Sitter(tmp_path / "runs" / "a").finish()
    run = tmp_path / "runs" / "a"
    env = json.loads((run / "launch" / "env.json").read_text())
    assert env["git"]["dirty"] and env["git"]["untracked"] == ["new.py"]
    assert "+x = 2" in (run / "launch" / "git_diff.patch").read_text()


def test_resume_keeps_original_and_writes_new_capture(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    Sitter(tmp_path / "runs" / "a").finish()
    first = json.loads((tmp_path / "runs" / "a" / "launch" / "env.json").read_text())["git"]["commit"]
    (repo / "train.py").write_text("x = 3\n")
    _git(repo, "commit", "-qam", "change")
    Sitter(tmp_path / "runs" / "a", resume=True).finish()
    run = tmp_path / "runs" / "a"
    assert json.loads((run / "launch" / "env.json").read_text())["git"]["commit"] == first
    assert json.loads((run / "launch" / "resume_1" / "env.json").read_text())["git"]["commit"] != first


def test_outside_git_repo_and_eval_mode(tmp_path, monkeypatch):
    script = tmp_path / "s.py"; script.write_text("")
    monkeypatch.setattr(sys, "argv", [str(script)])
    env, diff = capture_environment()
    assert env["git"] is None and diff is None
    Sitter(tmp_path / "runs" / "a").finish()
    Sitter(tmp_path / "runs" / "a", eval_mode=True)
    assert not list((tmp_path / "runs" / "a" / "launch").glob("resume_*"))


def _index_bytes(repo):
    return (repo / ".git" / "index").read_bytes()


def test_untracked_files_in_patch_and_real_index_untouched(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    (repo / "new.py").write_text("brand_new = 1\n")
    before = _index_bytes(repo)
    Sitter(tmp_path / "runs" / "a").finish()
    patch = (tmp_path / "runs" / "a" / "launch" / "git_diff.patch").read_text()
    assert "+++ b/new.py" in patch and "+brand_new = 1" in patch
    assert _index_bytes(repo) == before
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    assert status.strip() == "?? new.py"  # still untracked: no intent-to-add leaked into the real index


def test_diff_truncated_flag_and_full_size(tmp_path, monkeypatch):
    from mosi import environment
    repo = _repo(tmp_path, monkeypatch)
    (repo / "train.py").write_text("x = 1\n" + "y = 2\n" * 200)
    monkeypatch.setattr(environment, "MAX_DIFF_BYTES", 100)
    env, diff = capture_environment()
    assert env["git"]["diff_truncated"] is True and env["git"]["diff_bytes"] > 100 and len(diff.encode()) <= 100
    monkeypatch.setattr(environment, "MAX_DIFF_BYTES", 10**6)
    env, diff = capture_environment()
    assert env["git"]["diff_truncated"] is False and env["git"]["diff_bytes"] == len(diff.encode())


def test_env_records_gpu_packages_and_pixi(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    (repo / "pixi.lock").write_text("lock")
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "dev")
    env, _ = capture_environment()
    import hashlib
    assert env["pixi"] == {"lock_sha256": hashlib.sha256(b"lock").hexdigest(), "environment": "dev"}
    assert "numpy" in env["packages"] and "model-sitter" in env["packages"]
    assert env["gpu"] is None or {"names", "driver", "cuda", "source"} <= set(env["gpu"])


def test_provenance_failure_warns_and_does_not_break(tmp_path, monkeypatch, caplog):
    import logging
    from mosi import environment
    _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(environment, "_package_versions", lambda: 1 / 0)
    log = logging.getLogger("t")
    with caplog.at_level(logging.WARNING, logger="t"):
        env, _ = capture_environment(log)
    assert env["packages"] is None and any("packages" in r.message and r.levelname == "WARNING" for r in caplog.records)
