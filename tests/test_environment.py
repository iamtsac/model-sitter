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
    env = json.loads((tmp_path / "runs" / "a" / "env.json").read_text())
    assert env["command"]["argv"][1:] == ["--lr", "0.1"]
    assert env["git"]["dirty"] is False and len(env["git"]["commit"]) == 40
    assert not (tmp_path / "runs" / "a" / "git_diff.patch").exists()


def test_dirty_tree_saves_diff_and_untracked(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    (repo / "train.py").write_text("x = 2\n")
    (repo / "new.py").write_text("")
    Sitter(tmp_path / "runs" / "a").finish()
    run = tmp_path / "runs" / "a"
    env = json.loads((run / "env.json").read_text())
    assert env["git"]["dirty"] and env["git"]["untracked"] == ["new.py"]
    assert "+x = 2" in (run / "git_diff.patch").read_text()


def test_resume_keeps_original_and_writes_new_capture(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    Sitter(tmp_path / "runs" / "a").finish()
    first = json.loads((tmp_path / "runs" / "a" / "env.json").read_text())["git"]["commit"]
    (repo / "train.py").write_text("x = 3\n")
    _git(repo, "commit", "-qam", "change")
    Sitter(tmp_path / "runs" / "a", resume=True).finish()
    run = tmp_path / "runs" / "a"
    assert json.loads((run / "env.json").read_text())["git"]["commit"] == first
    assert json.loads((run / "env.resume_1.json").read_text())["git"]["commit"] != first


def test_outside_git_repo_and_eval_mode(tmp_path, monkeypatch):
    script = tmp_path / "s.py"; script.write_text("")
    monkeypatch.setattr(sys, "argv", [str(script)])
    env, diff = capture_environment()
    assert env["git"] is None and diff is None
    Sitter(tmp_path / "runs" / "a").finish()
    Sitter(tmp_path / "runs" / "a", eval_mode=True)
    assert not list((tmp_path / "runs" / "a").glob("env.resume*"))
