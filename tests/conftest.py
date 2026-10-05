import sys
import types
import pytest


class FakeRun:
    """What `wandb.init` returns: the logger talks to this per-run handle, not the wandb module."""
    def __init__(self, mod, **kw):
        self._mod = mod
        self.init_kwargs = kw
        self.name, self.id, self.project = kw.get("name"), kw.get("id"), kw.get("project")
        self.path = f"ent/{self.project}/{self.id}"
        self.tags = tuple(kw.get("tags") or ())
        self.config = types.SimpleNamespace(
            update=lambda d, allow_val_change=False: mod.config.updates.append(d))

    def log(self, data, step=None, commit=True): self._mod.logs.append((step, data))
    def define_metric(self, *a, **k): pass
    def finish(self): self._mod.run = None


@pytest.fixture
def fake_wandb(monkeypatch):
    """A minimal stand-in for the wandb module recording what the logger sends."""
    mod = types.ModuleType("wandb")
    mod.inits, mod.logs, mod.deleted, mod.renamed = [], [], [], []
    mod.run = None

    def init(**kw):
        mod.inits.append(kw)
        mod.run = FakeRun(mod, **kw)
        return mod.run

    class Api:
        def run(self, path):
            class R:
                def delete(_): mod.deleted.append(path)
                def update(_): mod.renamed.append((path, _.name))
            return R()

    mod.init, mod.Api = init, Api
    mod.log = lambda data, step=None, commit=True: mod.logs.append((step, data))
    mod.define_metric = lambda *a, **k: None
    mod.Settings = lambda **k: k
    mod.finish = lambda: setattr(mod, "run", None)
    mod.config = types.SimpleNamespace(update=lambda *a, **k: None)
    mod.config.updates = []
    mod.config.update = lambda d, allow_val_change=False: mod.config.updates.append(d)

    sdk = types.ModuleType("wandb.sdk"); lib = types.ModuleType("wandb.sdk.lib"); rid = types.ModuleType("wandb.sdk.lib.runid")
    counter = iter(range(1000))
    rid.generate_id = lambda: f"id{next(counter)}"
    for n, m in {"wandb": mod, "wandb.sdk": sdk, "wandb.sdk.lib": lib, "wandb.sdk.lib.runid": rid}.items():
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    return mod


class Stateful:
    """Framework-free object with the state_dict protocol."""
    def __init__(self, v=0): self.v = v
    def state_dict(self): return {"v": self.v}
    def load_state_dict(self, d): self.v = d["v"]
