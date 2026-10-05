import json
import pytest
from conftest import Stateful
from mosi import Sitter, load_group_stats, list_group_runs

YES = lambda msg: True


def test_flat_layout_unchanged(tmp_path):
    log = Sitter(tmp_path / "exp")
    assert log.env.fpath == tmp_path / "exp" and log.group_dir is None


def test_grouped_layout_and_manifest(tmp_path):
    a = Sitter(tmp_path, group="seeds_run", name="test_s0", tags=["x"], job_type="train")
    b = Sitter(tmp_path, group="seeds_run", name="test_s1")
    assert a.env.fpath == tmp_path / "seeds_run" / "test_s0"
    assert b.env.fpath == tmp_path / "seeds_run" / "test_s1"
    runs = list_group_runs(tmp_path / "seeds_run")
    assert set(runs) == {"test_s0", "test_s1"} and runs["test_s0"]["tags"] == ["x"]
    assert not (tmp_path / "seeds_run" / "group.json").exists()


@pytest.mark.parametrize("group,name", [("../x", "a"), ("g", "a/b"), ("g", "..")])
def test_rejects_path_escapes(tmp_path, group, name):
    with pytest.raises(ValueError):
        Sitter(tmp_path, group=group, name=name)


def test_grouped_run_needs_name(tmp_path):
    with pytest.raises(ValueError):
        Sitter(tmp_path, group="g")


def test_stats_and_load_group_stats(tmp_path):
    for i in (0, 1):
        log = Sitter(tmp_path, group="abl", name=f"s{i}")
        log.add_stats(loss=1.0 + i, val={"acc": 0.5})
        log.finish()
    stats = load_group_stats(tmp_path / "abl")
    assert stats["s1"] == [{"step": 1, "loss": 2.0, "val/acc": 0.5}]
    assert set(stats) == {"s0", "s1"}


def test_overwrite_only_touches_own_run(tmp_path):
    for n in ("s0", "s1"):
        log = Sitter(tmp_path, group="g", name=n)
        log.add_stats(loss=1.0); log.finish()
    Sitter(tmp_path, group="g", name="s0", overwrite=True, confirm=YES).finish()
    assert not (tmp_path / "g" / "s0" / "stats.jsonl").exists()
    assert (tmp_path / "g" / "s1" / "stats.jsonl").exists()


def test_confirm_can_abort(tmp_path):
    Sitter(tmp_path, group="g", name="s0")
    with pytest.raises(SystemExit):
        Sitter(tmp_path, group="g", name="s0", overwrite=True, confirm=lambda m: False)


def test_rerun_without_flags_archives_only_that_run(tmp_path):
    Sitter(tmp_path, group="g", name="s0").finish()
    Sitter(tmp_path, group="g", name="s0")
    names = sorted(p.name for p in (tmp_path / "g").iterdir() if p.is_dir())
    assert "s0" in names and len(names) == 2
    assert list(load_group_stats(tmp_path / "g")) == ["s0"]


def test_wandb_receives_group_metadata(tmp_path, fake_wandb):
    log = Sitter(tmp_path, group="seeds_run", name="test_s3", job_type="train", tags=["base"], notes="n", use_wandb=True)
    kw = fake_wandb.inits[0]
    assert (kw["group"], kw["job_type"], kw["name"], kw["notes"]) == ("seeds_run", "train", "test_s3", "n")
    assert kw["tags"] == ["base"] and kw["project"] == "proj"
    assert "/" not in kw["id"] and kw["id"].startswith("seeds_run_test_s3_")
    assert fake_wandb.config.updates == []  # nothing we invented leaks into the W&B config
    log.finish()


def test_wandb_overwrite_deletes_only_own_lineage(tmp_path, fake_wandb):
    Sitter(tmp_path, group="g", name="s0", use_wandb=True).finish()
    Sitter(tmp_path, group="g", name="s1", use_wandb=True).finish()
    Sitter(tmp_path, group="g", name="s0", overwrite=True, confirm=YES, use_wandb=True).finish()
    assert len(fake_wandb.deleted) == 1 and "g_s0" in fake_wandb.deleted[0]


def test_wandb_resume_branches_without_name_growth(tmp_path, fake_wandb):
    def run(**kw):
        log = Sitter(tmp_path, group="g", name="s0", use_wandb=True, checkpoint_backend="pickle", **kw)
        return log
    first = run()
    first.add_stats(loss=1.0); first.checkpoint(model=Stateful(1)); first.step(); first.finish()
    for _ in range(2):
        log = run(resume=True)
        log.load_checkpoint("latest", model=Stateful())
        assert log.telemetry.run.name == "s0_resume_from_step_1"      # always from the original name
        assert "resumed" in log.telemetry.run.tags
        log.finish()
    ids = (tmp_path / "g" / "s0" / "wandb_id.txt").read_text().split()
    assert len(ids) == 3 and len(set(ids)) == 3                       # one fresh id per run, no duplicates
    assert [step for step, d in fake_wandb.logs if "loss" in d].count(1) == 3  # history replayed into each branch


def test_ungrouped_wandb_keeps_old_naming(tmp_path, fake_wandb):
    Sitter(tmp_path / "exp", use_wandb=True)
    kw = fake_wandb.inits[0]
    assert kw["group"] is None and kw["id"].startswith("exp_") and kw["name"] == kw["id"]
