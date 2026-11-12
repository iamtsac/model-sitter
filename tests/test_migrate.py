import json
import os
import time
from pathlib import Path
from conftest import Stateful
from mosi import Sitter
from mosi.migrations import migrate, find_runs
from mosi.migrations.cli import main

OLD = time.time() - 3600


def touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (OLD, OLD))


def build_v1(root: Path, group_member=False) -> Path:
    """A run as written by layout 1 (real pickle checkpoints so they can be loaded after migrating)."""
    import pickle
    run = root / "old_run"
    for step in (1, 2):
        ck = run / f"checkpoint_{step}"
        ck.mkdir(parents=True)
        (ck / "checkpoint.pkl").write_bytes(pickle.dumps({"step": step, "model": {"v": step * 10}}))
    best = run / "best_model"
    best.mkdir()
    (best / "checkpoint.pkl").write_bytes(pickle.dumps({"step": 1, "model": {"v": 10}}))
    touch(best / "best_metrics.jsonl", '{"step": 1}\n')
    touch(run / "env.json", "{}"); touch(run / "git_diff.patch", "d")
    touch(run / "env.resume_1.json", "{}"); touch(run / "git_diff.resume_1.patch", "d1")
    touch(run / "run.log", "log"); touch(run / "stats.jsonl", '{"step": 1, "loss": 1}\n{"step": 2, "loss": 0.5}\n')
    touch(run / "wandb" / "x.txt"); touch(run / "config.yaml", "a: 1\n")
    touch(run / "wandb_id.txt", "idA\nidB\n")
    touch(run / "images" / "step_1" / "a.png"); touch(run / "videos" / "step_2" / "v.mp4")
    touch(run / "analysis" / "step_2" / "w.csv"); touch(run / "analysis" / "notes.txt")
    touch(run / "evaluation" / "eval_stats.jsonl", '{"step": 2, "acc": 1}\n'); touch(run / "evaluation" / "config.yaml")
    touch(run / "evaluation" / "images" / "step_2" / "e.png")
    touch(run / "evaluation" / "videos" / "step_1" / "a.mp4"); touch(run / "evaluation" / "videos" / "step_2" / "b.mp4")
    touch(run / "evaluation" / "stray.txt")
    touch(run / "clean.npz")
    if group_member:
        touch(run.parent / "sibling" / "run.json", json.dumps({"group": run.parent.name, "name": "sibling"}))
    return run


def tree(root: Path):
    return sorted(str(p.relative_to(root)) + ("/" if p.is_dir() else "") for p in root.rglob("*"))


def test_dry_run_changes_nothing_then_apply_builds_target_tree(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    run = build_v1(tmp_path)
    before = tree(run)
    lines = []
    counts = migrate(run, out=lines.append)
    assert tree(run) == before and counts == {"migrated": 1, "current": 0, "refused": 0}
    text = "\n".join(lines)
    assert "layout 1 -> 2" in text and "move   checkpoint_1/" in text and "keep   clean.npz" in text
    assert "dry run: nothing changed. Re-run with --apply." in lines[-1]

    migrate(run, apply=True, out=lambda m: None)
    assert tree(run) == sorted([
        "analysis/", "analysis/notes.txt", "artifacts/", "artifacts/step_1/", "artifacts/step_1/images/",
        "artifacts/step_1/images/a.png", "artifacts/step_2/", "artifacts/step_2/analysis/",
        "artifacts/step_2/analysis/w.csv", "artifacts/step_2/videos/", "artifacts/step_2/videos/v.mp4",
        "checkpoints/", "checkpoints/best/", "checkpoints/best/best_metrics.jsonl", "checkpoints/best/checkpoint.pkl",
        "checkpoints/step_1/", "checkpoints/step_1/checkpoint.pkl", "checkpoints/step_2/",
        "checkpoints/step_2/checkpoint.pkl", "clean.npz", "config.yaml", "evaluation/", "evaluation/stray.txt",
        "evaluations/", "evaluations/legacy/", "evaluations/legacy/artifacts/",
        "evaluations/legacy/artifacts/images/", "evaluations/legacy/artifacts/images/e.png",
        "evaluations/legacy/artifacts/videos/", "evaluations/legacy/artifacts/videos/step_1/",
        "evaluations/legacy/artifacts/videos/step_1/a.mp4", "evaluations/legacy/artifacts/videos/step_2/",
        "evaluations/legacy/artifacts/videos/step_2/b.mp4", "evaluations/legacy/config.yaml",
        "evaluations/legacy/logs/", "evaluations/legacy/logs/stats.jsonl", "launch/", "launch/env.json",
        "launch/git_diff.patch", "launch/resume_1/", "launch/resume_1/env.json", "launch/resume_1/git_diff.patch",
        "logs/", "logs/run.log", "logs/stats.jsonl", "logs/wandb/", "logs/wandb/x.txt", "run.json"])
    info = json.loads((run / "run.json").read_text())
    assert info["layout_version"] == 2 and info["name"] == "old_run" and info["group"] is None
    assert info["wandb"] == {"project": "proj", "entity": None, "runs": [
        {"id": i, "name": None, "started": None, "resumed_from_step": None, "url": None} for i in ("idA", "idB")]}


def test_migrated_run_loads_and_resumes(tmp_path):
    run = build_v1(tmp_path)
    migrate(run, apply=True, out=lambda m: None)
    log = Sitter(run, eval_mode=True, checkpoint_backend="pickle", eval_name="e")
    m = Stateful()
    assert log.load_checkpoint("latest", model=m)["step"] == 2 and m.v == 20
    assert log.load_checkpoint("best", model=m)["step"] == 1 and m.v == 10
    assert log.load_checkpoint(2, model=m)["step"] == 2
    assert log.load_checkpoint(str(run / "checkpoints" / "step_1"), model=m)["step"] == 1
    log.finish()
    resumed = Sitter(run, resume=True, checkpoint_backend="pickle")
    resumed.load_checkpoint(1, model=Stateful())
    assert resumed.current_step == 2
    resumed.finish()
    assert [json.loads(l)["step"] for l in (run / "logs" / "stats.jsonl").read_text().splitlines()] == [1]
    assert (run / "launch" / "resume_1" / "env.json").exists()  # launch/resume_1 came from the old capture


def test_rerun_reports_current(tmp_path):
    run = build_v1(tmp_path)
    migrate(run, apply=True, out=lambda m: None)
    lines = []
    counts = migrate(run, apply=True, out=lines.append)
    assert counts == {"migrated": 0, "current": 1, "refused": 0} and "layout 2 -> 2" in lines[0]


def test_refuses_running_and_recent_log(tmp_path):
    run = build_v1(tmp_path)
    touch(run / "run.json", json.dumps({"name": "old_run", "status": "running"}))
    before = tree(run)
    lines = []
    assert migrate(run, apply=True, out=lines.append)["refused"] == 1
    assert "status is 'running'" in "\n".join(lines) and tree(run) == before
    touch(run / "run.json", json.dumps({"name": "old_run", "status": "finished"}))
    os.utime(run / "run.log")  # modified just now
    lines = []
    assert migrate(run, apply=True, out=lines.append)["refused"] == 1
    assert "run.log was modified in the last 5 minutes" in "\n".join(lines)
    assert (run / "checkpoint_1").exists()


def test_folder_walk_and_group_membership(tmp_path):
    run = build_v1(tmp_path / "g", group_member=True)
    other = tmp_path / "plain" / "r2"
    touch(other / "stats.jsonl", "")
    assert sorted(p.name for p in find_runs(tmp_path)) == ["old_run", "r2", "sibling"]
    counts = migrate(tmp_path, apply=True, out=lambda m: None)
    assert counts["migrated"] == 3
    assert json.loads((run / "run.json").read_text())["group"] == "g"
    assert json.loads((other / "run.json").read_text())["group"] is None


def test_existing_target_refuses_without_changes(tmp_path):
    run = build_v1(tmp_path)
    (run / "checkpoints" / "step_1").mkdir(parents=True)
    before = tree(run)
    lines = []
    assert migrate(run, apply=True, out=lines.append)["refused"] == 1
    assert tree(run) == before and "already exists" in "\n".join(lines)


def test_no_wandb_project_warns(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    run = build_v1(tmp_path)
    lines = []
    migrate(run, out=lines.append)
    assert any("warn" in l and "WANDB_PROJECT" in l for l in lines)


def test_cli_exit_codes(tmp_path, capsys):
    run = build_v1(tmp_path)
    assert main(["migrate", str(run)]) == 0 and (run / "checkpoint_1").exists()
    assert main(["migrate", str(run), "--apply"]) == 0 and (run / "checkpoints" / "step_1").exists()
    assert main(["migrate", str(tmp_path / "missing")]) == 2
