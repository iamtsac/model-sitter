import json
import dataclasses
import numpy as np
import pytest
from conftest import Stateful
from mosi import Sitter, PickleBackend, load_config


@dataclasses.dataclass
class Cfg:
    lr: float = 1e-3


def make(tmp_path, **kw):
    return Sitter(tmp_path / "r", checkpoint_backend="pickle", **kw)


@pytest.mark.parametrize("order", ["obj_first", "dict_first"])
def test_add_config_keeps_both_kinds(tmp_path, order):
    log = make(tmp_path)
    calls = [lambda: log.add_config(Cfg(lr=0.5)), lambda: log.add_config(seed=3)]
    for call in (calls if order == "obj_first" else calls[::-1]):
        call()
    saved = load_config(tmp_path / "r" / "config.yaml")
    assert saved == {"config": Cfg(lr=0.5), "extra": {"seed": 3}}


def test_add_config_single_kind_stays_plain(tmp_path):
    log = make(tmp_path)
    log.add_config(seed=3)
    assert load_config(tmp_path / "r" / "config.yaml") == {"seed": 3}
    log = Sitter(tmp_path / "o", checkpoint_backend="pickle")
    log.add_config(Cfg())
    assert load_config(tmp_path / "o" / "config.yaml") == Cfg()


def test_failed_save_leaves_no_partial_checkpoint(tmp_path):
    class Boom(PickleBackend):
        def save(self, obj, path):
            path.write_bytes(b"half")
            raise RuntimeError("disk full")
    log = Sitter(tmp_path / "r", checkpoint_backend=Boom())
    with pytest.raises(RuntimeError):
        log.checkpoint(model=Stateful(1))
    assert not list((tmp_path / "r" / "checkpoint_1").glob("checkpoint.*"))


def test_load_latest_and_no_side_effects(tmp_path):
    log = make(tmp_path)
    for v in (1, 2, 3):
        log.checkpoint(model=Stateful(v)); log.step()
    m = Stateful()
    cp = log.load_checkpoint("latest", model=m)
    assert m.v == 3 and cp["step"] == 3
    before = (log.env.fpath, log.env.work_dir, log.history.stats_file)
    log.load_checkpoint(str(tmp_path / "r" / "checkpoint_1"), model=m)
    assert (log.env.fpath, log.env.work_dir, log.history.stats_file) == before


def test_load_latest_without_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError):
        make(tmp_path).load_checkpoint("latest", model=Stateful())


def test_missing_key_error_names_it(tmp_path):
    log = make(tmp_path)
    log.checkpoint(model=Stateful(1))
    with pytest.raises(KeyError, match="optimizer"):
        log.load_checkpoint(1, optimizer=Stateful())


def test_should_stop_after_patience(tmp_path):
    log = make(tmp_path)
    log.enable_monitor(decrease_keys=["loss"], patience=2)
    assert not log.should_stop
    for loss in (1.0, 1.1, 1.2):
        log.add_stats(loss=loss); log.checkpoint(model=Stateful()); log.step()
    assert log.should_stop
    log.add_stats(loss=0.1); log.checkpoint(model=Stateful())
    assert not log.should_stop


def test_checkpoint_defaults_to_staged_stats(tmp_path):
    log = make(tmp_path)
    log.enable_monitor(decrease_keys=["val/loss"])
    log.add_stats(val={"loss": 0.5}); log.checkpoint(model=Stateful(1)); log.step()
    assert (tmp_path / "r" / "best_model").exists()


def test_monitor_log_every(tmp_path, caplog):
    import logging
    log = make(tmp_path)
    log.logger.addHandler(caplog.handler)
    log.enable_monitor(decrease_keys=["loss"], log_every=3)
    with caplog.at_level(logging.INFO, logger=log.logger.name):
        for i in range(6):
            log.checkpoint(metrics={"loss": 1.0 - 0.1 * i}, model=Stateful()); log.step()
    assert sum("New Best" in r.message for r in caplog.records) == 2


def test_stats_scalars_nan_and_arrays(tmp_path):
    log = make(tmp_path)
    log.add_stats(a=np.float32(1.5), b=np.array(2.0), c=float("nan"))
    log.step()
    row = json.loads((tmp_path / "r" / "stats.jsonl").read_text())
    assert (row["a"], row["b"], row["c"]) == (1.5, 2.0, None)
    with pytest.raises(ValueError, match="scalar"):
        log.add_stats(vec=np.zeros(3))


def test_context_manager_flushes_and_closes_log_file(tmp_path, fake_wandb):
    with make(tmp_path, use_wandb=True) as log:
        log.add_stats(loss=1.0)
    assert (tmp_path / "r" / "stats.jsonl").exists() and fake_wandb.run is None
    assert "Initialized Experiment" in (tmp_path / "r" / "run.log").read_text()


def test_log_level_quiets_console_but_not_file(tmp_path, capsys):
    log = make(tmp_path, log_level="WARNING")
    log.logger.info("hidden on console")
    log.finish()
    assert "hidden on console" not in capsys.readouterr().out
    assert "hidden on console" in (tmp_path / "r" / "run.log").read_text()


def test_rerun_within_one_second_does_not_collide(tmp_path):
    for _ in range(3):
        make(tmp_path).finish()
    assert len([p for p in tmp_path.iterdir()]) == 3
