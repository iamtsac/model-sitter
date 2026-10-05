import sys
import pytest
from mosi import Sitter, PickleBackend
from conftest import Stateful


def test_core_import_needs_no_torch_or_wandb():
    import subprocess
    code = "import sys, mosi; assert not {'torch','wandb','imageio'} & set(sys.modules), sorted(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pickle_roundtrip_and_best(tmp_path):
    log = Sitter(tmp_path / "r", checkpoint_backend="pickle")
    log.enable_monitor(decrease_keys=["loss"])
    m, opts = Stateful(1), [Stateful(10), Stateful(20)]
    log.add_stats(loss=2.0); log.checkpoint(metrics={"loss": 2.0}, model=m, optimizer=opts); log.step()
    m.v = 2
    log.add_stats(loss=1.0); log.checkpoint(metrics={"loss": 1.0}, model=m, optimizer=opts); log.step()
    assert (tmp_path / "r" / "checkpoint_1" / "checkpoint.pkl").exists()
    assert (tmp_path / "r" / "best_model" / "checkpoint.pkl").exists()

    fresh = Sitter(tmp_path / "r", eval_mode=True, checkpoint_backend="pickle")
    m2, opts2 = Stateful(), [Stateful(), Stateful()]
    cp = fresh.load_checkpoint(1, model=m2, optimizer=opts2)
    assert (m2.v, opts2[1].v, cp["step"]) == (1, 20, 1)
    fresh.load_checkpoint("best", model=m2)
    assert m2.v == 2


def test_missing_checkpoint_errors(tmp_path):
    with pytest.raises(FileNotFoundError):   # eval_mode never creates a run
        Sitter(tmp_path / "missing", eval_mode=True)
    (tmp_path / "r").mkdir()
    log = Sitter(tmp_path / "r", eval_mode=True, checkpoint_backend="pickle")
    with pytest.raises(FileNotFoundError):
        log.load_checkpoint(5, model=Stateful())
    with pytest.raises(FileNotFoundError):
        log.load_checkpoint("best", model=Stateful())


def test_custom_backend(tmp_path):
    class Json(PickleBackend):
        suffix = "pkl"
    log = Sitter(tmp_path / "r", checkpoint_backend=Json())
    log.checkpoint(model=Stateful(3))
    m = Stateful()
    log.load_checkpoint(1, model=m)
    assert m.v == 3


def test_torch_backend(tmp_path):
    torch = pytest.importorskip("torch")
    net = torch.nn.Linear(3, 2)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    log = Sitter(tmp_path / "r")  # auto -> torch
    log.checkpoint(model=net, optimizer=opt)
    assert (tmp_path / "r" / "checkpoint_1" / "checkpoint.pt").exists()
    net2 = torch.nn.Linear(3, 2)
    Sitter(tmp_path / "r", eval_mode=True).load_checkpoint(1, model=net2, optimizer=torch.optim.SGD(net2.parameters(), lr=0.1))
    assert torch.equal(net.weight, net2.weight)


def test_config_roundtrip(tmp_path):
    import dataclasses
    from mosi import load_config
    log = Sitter(tmp_path / "r")
    log.add_config(lr=1e-3, shape=(2, 3))
    cfg = load_config(tmp_path / "r" / "config.yaml")
    assert cfg == {"lr": 1e-3, "shape": (2, 3)}


def test_plain_data_roundtrip_like_jax_pytrees(tmp_path):
    import numpy as np
    log = Sitter(tmp_path / "r", checkpoint_backend="pickle")
    params = {"w": np.arange(3.0), "layers": [{"b": np.ones(2)}]}
    log.checkpoint(params=params, ema=Stateful(5), rng=np.array([1, 2], dtype=np.uint32), epoch=7)
    out = log.load_checkpoint(1, ema=Stateful())
    assert out["epoch"] == 7 and out["ema"].v == 5
    assert np.array_equal(out["params"]["w"], params["w"]) and out["rng"].tolist() == [1, 2]
    # asking for a plain key explicitly gives the saved value too
    assert log.load_checkpoint(1, params=None)["params"]["layers"][0]["b"].tolist() == [1.0, 1.0]


def test_reserved_names_rejected(tmp_path):
    log = Sitter(tmp_path / "r", checkpoint_backend="pickle")
    with pytest.raises(TypeError):  # `step` / `monitor_state` are the logger's own keys
        log.checkpoint(monitor_state=3, step=2)


def test_dict_of_stateful_saves_states_not_objects(tmp_path):
    import pickle
    log = Sitter(tmp_path / "r", checkpoint_backend="pickle")
    log.checkpoint(nets={"gen": Stateful(1), "disc": Stateful(2), "n": 9})
    raw = pickle.load(open(tmp_path / "r" / "checkpoint_1" / "checkpoint.pkl", "rb"))
    assert raw["nets"] == {"gen": {"v": 1}, "disc": {"v": 2}, "n": 9}
    live = {"gen": Stateful(), "disc": Stateful()}
    out = log.load_checkpoint(1, nets=live)
    assert (live["gen"].v, live["disc"].v, out["nets"]["n"]) == (1, 2, 9)
