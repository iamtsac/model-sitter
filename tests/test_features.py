import dataclasses
import pytest
from mosi import Sitter, SitterConfig, dump_config, load_config
from conftest import Stateful


class Img:
    def save(self, path, **kw): open(path, "wb").write(b"x")


@dataclasses.dataclass
class Config:  # module level so dump_config can import it back
    lr: float = 1e-3
    logger: SitterConfig = dataclasses.field(default_factory=SitterConfig)


def test_keep_last_prunes_old_but_keeps_best(tmp_path):
    log = Sitter(tmp_path / "r", checkpoint_backend="pickle", keep_last=2)
    log.enable_monitor(decrease_keys=["loss"])
    for loss in (1.0, 5.0, 6.0, 7.0, 8.0):
        log.checkpoint(metrics={"loss": loss}, model=Stateful(1)); log.step()
    left = sorted(p.name for p in (tmp_path / "r").glob("checkpoint_*"))
    assert left == ["checkpoint_4", "checkpoint_5"]
    assert (tmp_path / "r" / "best_model" / "checkpoint.pkl").exists()


def test_keep_last_validation(tmp_path):
    with pytest.raises(ValueError):
        Sitter(tmp_path / "r", keep_last=0)


def test_artifacts_split_into_folders(tmp_path):
    log = Sitter(tmp_path / "r")
    log.step(7)
    log.add_image("a.png", Img())
    log.add_video("v", str(tmp_path / "none.mp4"))  # missing file: logged as error, no crash
    log.add_analysis("w", {"i": [1, 2], "w": [0.1, 0.2]})
    r = tmp_path / "r"
    assert (r / "images" / "step_7" / "a.png").exists()
    assert (r / "analysis" / "step_7" / "w.csv").read_text().splitlines()[0] == "i,w"
    assert not (r / "checkpoint_7").exists()  # artifacts no longer live inside checkpoint folders


def test_eval_mode_artifacts_stay_in_evaluation_dir(tmp_path):
    Sitter(tmp_path / "r").finish()
    log = Sitter(tmp_path / "r", eval_mode=True)
    log.add_image("a.png", Img())
    assert (tmp_path / "r" / "evaluation" / "images" / "step_1" / "a.png").exists()


def test_monitor_is_quiet_without_improvement(tmp_path, caplog):
    import logging
    log = Sitter(tmp_path / "r", checkpoint_backend="pickle")
    log.enable_monitor(decrease_keys=["loss"])
    log.logger.addHandler(caplog.handler)
    with caplog.at_level(logging.INFO, logger=log.logger.name):
        for loss in (1.0, 1.0, 1.0):
            log.checkpoint(metrics={"loss": loss}, model=Stateful()); log.step()
    msgs = [r.message for r in caplog.records]
    assert sum("New Best" in m for m in msgs) == 1 and not any("No improvement" in m for m in msgs)
    assert not any("Net Relative" in m for m in msgs)


def test_sitter_config_builds_logger_and_roundtrips_yaml(tmp_path):
    cfg = SitterConfig(fpath=str(tmp_path), group="g", name="n", tags=("a", "b"), keep_last=3, checkpoint_backend="pickle")
    log = Sitter.from_config(cfg)
    assert log.env.fpath == tmp_path / "g" / "n" and log.checkpointer.keep_last == 3
    out = tmp_path / "cfg.yaml"
    out.write_text(dump_config(cfg))
    assert load_config(out) == cfg


def test_sitter_config_works_with_tyro(tmp_path):
    tyro = pytest.importorskip("tyro")
    cfg = tyro.cli(Config, args=["--lr", "0.5", "--logger.group", "g", "--logger.name", "n",
                                 "--logger.tags", "x", "y", "--logger.keep-last", "2", "--logger.fpath", str(tmp_path)])
    assert cfg.lr == 0.5 and cfg.logger.tags == ("x", "y") and cfg.logger.keep_last == 2
    log = cfg.logger.build(checkpoint_backend="pickle")
    log.add_config(cfg)
    assert load_config(tmp_path / "g" / "n" / "config.yaml") == cfg
