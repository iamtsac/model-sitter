import json
import subprocess
import sys
import pytest
from conftest import Stateful
from mosi import Sitter, LegacyLayoutError, load_group_stats


class Img:
    def save(self, path, **kw): open(path, "wb").write(b"x")


def info(path):
    return json.loads(path.read_text())


def make(tmp_path, **kw):
    return Sitter(tmp_path / "r", checkpoint_backend="pickle", **kw)


def test_fresh_run_layout(tmp_path):
    log = Sitter(
        tmp_path, group="g", run_name="s0", job_type="train", tags=["a"], checkpoint_backend="pickle")
    log.add_config(lr=1)
    log.add_stats(loss=1.0)
    log.checkpoint(model=Stateful(1))
    log.step()
    log.finish()
    r = tmp_path / "g" / "s0"
    assert sorted(p.name for p in r.iterdir()) == ["checkpoints", "config.yaml", "launch", "logs", "run.json"]
    assert (r / "launch" / "env.json").exists() and (r / "logs" / "run.log").exists()
    assert (r / "logs" / "stats.jsonl").exists() and (r / "checkpoints" / "step_1" / "checkpoint.pkl").exists()
    data = info(r / "run.json")
    assert data["group"] == "g" and data["name"] == "s0" and data["job_type"] == "train" and data["tags"] == ["a"]
    assert data["layout_version"] == 2 and data["status"] == "finished" and data["finished"] and data["wandb"] is None
    assert load_group_stats(tmp_path / "g")["s0"] == [{"step": 1, "loss": 1.0}]


def test_ungrouped_run_also_has_run_json(tmp_path):
    make(tmp_path).finish()
    assert info(tmp_path / "r" / "run.json")["group"] is None


def test_resume_launch_info_goes_to_resume_folder(tmp_path):
    make(tmp_path).finish()
    make(tmp_path, resume=True).finish()
    assert (tmp_path / "r" / "launch" / "resume_1" / "env.json").exists()
    assert info(tmp_path / "r" / "run.json")["status"] == "finished"


def test_wandb_block_lives_in_run_json(tmp_path, fake_wandb):
    make(tmp_path, use_wandb=True).finish()
    block = info(tmp_path / "r" / "run.json")["wandb"]
    (entry,) = block["runs"]
    assert (block["project"], block["entity"]) == ("proj", None)
    assert entry["id"] == fake_wandb.inits[0]["id"] and entry["resumed_from_step"] is None and entry["started"]
    assert fake_wandb.inits[0]["dir"] == str(tmp_path / "r" / "logs")


def test_status_running_finished_crashed(tmp_path):
    log = make(tmp_path)
    assert info(tmp_path / "r" / "run.json")["status"] == "running"
    log.finish()
    assert info(tmp_path / "r" / "run.json")["status"] == "finished"
    with pytest.raises(ValueError, match="boom"):
        with Sitter(tmp_path / "c", checkpoint_backend="pickle"):
            raise ValueError("boom")
    data = info(tmp_path / "c" / "run.json")
    assert data["status"] == "crashed" and data["finished"]


def test_uncaught_exception_marks_crashed_in_a_real_process(tmp_path):
    code = ("from mosi import Sitter\n"
            f"s = Sitter({str(tmp_path / 'x')!r}, checkpoint_backend='pickle')\n"
            "raise RuntimeError('original')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 1 and "RuntimeError: original" in out.stderr  # traceback untouched
    assert info(tmp_path / "x" / "run.json")["status"] == "crashed"


def test_normal_exit_without_exception_is_not_crashed(tmp_path):
    code = f"from mosi import Sitter\nSitter({str(tmp_path / 'y')!r}, checkpoint_backend='pickle')\n"
    subprocess.run([sys.executable, "-c", code], check=True)
    assert info(tmp_path / "y" / "run.json")["status"] == "running"


def test_eval_default_name_with_checkpoint_step(tmp_path):
    make(tmp_path).checkpoint(model=Stateful(1))
    ev = make(tmp_path, eval_mode=True)
    assert not (tmp_path / "r" / "evaluations").exists()  # opens at the first load / write
    ev.load_checkpoint("latest", model=Stateful())
    (folder,) = (tmp_path / "r" / "evaluations").iterdir()
    assert folder.name.startswith("step_1_") and len(folder.name) == len("step_1_2000-01-01T00-00-00")
    data = info(folder / "eval.json")
    assert data["checkpoint"] == {"path": str(tmp_path / "r" / "checkpoints" / "step_1"), "step": 1}
    assert data["status"] == "running" and data["created"] and ev.loaded_step == 1
    ev.finish()
    assert info(folder / "eval.json")["status"] == "finished"
    assert info(tmp_path / "r" / "run.json")["status"] == "running"  # the run itself is untouched


def test_eval_default_name_without_checkpoint_is_timestamp_only(tmp_path):
    make(tmp_path).finish()
    ev = make(tmp_path, eval_mode=True)
    ev.add_stats(acc=0.5)
    (folder,) = (tmp_path / "r" / "evaluations").iterdir()
    assert not folder.name.startswith("step_") and len(folder.name) == len("2000-01-01T00-00-00")
    assert info(folder / "eval.json")["checkpoint"] is None


def test_eval_never_overwrites_an_evaluation(tmp_path):
    make(tmp_path).finish()
    make(tmp_path, eval_mode=True, eval_name="e1").add_config(a=1)
    with pytest.raises(FileExistsError):
        make(tmp_path, eval_mode=True, eval_name="e1").add_config(a=2)
    assert (tmp_path / "r" / "evaluations" / "e1" / "config.yaml").exists()
    with pytest.raises(ValueError):
        make(tmp_path, eval_mode=True, eval_name="../x").add_config(a=2)


def test_eval_writes_launch_info_logs_stats_config(tmp_path):
    make(tmp_path).finish()
    ev = make(tmp_path, eval_mode=True, eval_name="e")
    ev.add_config(seed=1)
    ev.add_stats(acc=0.5)
    ev.finish()
    e = tmp_path / "r" / "evaluations" / "e"
    assert sorted(p.name for p in e.iterdir()) == ["config.yaml", "eval.json", "launch", "logs"]
    assert (e / "launch" / "env.json").exists()
    assert "Initialized Experiment" in (e / "logs" / "run.log").read_text()  # logged before the folder existed
    assert json.loads((e / "logs" / "stats.jsonl").read_text())["acc"] == 0.5
    assert not (tmp_path / "r" / "logs" / "stats.jsonl").exists()


def test_eval_wandb_default_off_and_opt_in(tmp_path, fake_wandb):
    make(tmp_path).finish()
    ev = make(tmp_path, eval_mode=True, use_wandb=True, eval_name="e")
    ev.add_stats(acc=1.0); ev.finish()
    assert fake_wandb.inits == []
    ev = make(tmp_path, eval_mode=True, use_wandb=True, eval_wandb=True, eval_name="e2")
    ev.add_stats(acc=1.0); ev.finish()
    assert len(fake_wandb.inits) == 1 and fake_wandb.inits[0]["name"] == "r_e2"
    assert info(tmp_path / "r" / "evaluations" / "e2" / "eval.json")["wandb"]["runs"][0]["id"] == fake_wandb.inits[0]["id"]
    assert info(tmp_path / "r" / "run.json")["wandb"] is None


def test_artifact_dir_training_and_eval(tmp_path):
    log = make(tmp_path)
    log.step(7)
    d = log.artifact_dir("rollouts")
    assert d == tmp_path / "r" / "artifacts" / "step_7" / "rollouts" and d.is_dir() and log.current_step == 7
    log.add_image("a.png", Img())
    assert (tmp_path / "r" / "artifacts" / "step_7" / "images" / "a.png").exists()
    log.finish()
    ev = make(tmp_path, eval_mode=True, eval_name="e")
    assert ev.artifact_dir("analysis") == tmp_path / "r" / "evaluations" / "e" / "artifacts" / "analysis"
    assert ev.artifact_dir("analysis").is_dir()
    with pytest.raises(ValueError):
        ev.artifact_dir("../x")


def test_old_layout_run_raises_with_migrate_hint(tmp_path):
    old = tmp_path / "old"
    (old / "checkpoint_1").mkdir(parents=True)
    (old / "stats.jsonl").write_text("{}\n")
    for kw in ({"resume": True}, {"eval_mode": True}, {"overwrite": True, "confirm": lambda m: True}):
        with pytest.raises(LegacyLayoutError, match="mosi migrate"):
            Sitter(old, **kw)
    (old / "run.json").write_text(json.dumps({"name": "old", "layout_version": 1}))
    with pytest.raises(LegacyLayoutError, match="mosi migrate"):
        Sitter(old, eval_mode=True)
    grouped = tmp_path / "g" / "s0"
    grouped.mkdir(parents=True)
    (grouped / "run.json").write_text(json.dumps({"name": "s0", "group": "g"}))
    with pytest.raises(LegacyLayoutError, match="mosi migrate"):
        load_group_stats(tmp_path / "g")


def test_wandb_overwrite_deletes_listed_ids_in_stored_project(tmp_path, fake_wandb, monkeypatch):
    first = make(tmp_path, use_wandb=True)
    first.checkpoint(model=Stateful(1)); first.finish()
    r = make(tmp_path, use_wandb=True, resume=True); r.load_checkpoint("latest", model=Stateful()); r.finish()
    ids = [e["id"] for e in info(tmp_path / "r" / "run.json")["wandb"]["runs"]]
    assert len(ids) == 2
    monkeypatch.setenv("WANDB_PROJECT", "other")
    make(tmp_path, use_wandb=True, overwrite=True, confirm=lambda m: True).finish()
    assert fake_wandb.deleted == [f"proj/{i}" for i in ids]
    block = info(tmp_path / "r" / "run.json")["wandb"]
    assert block["project"] == "other" and len(block["runs"]) == 1 and block["runs"][0]["id"] not in ids
