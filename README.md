# 🍼 Model-Sitter

A small experiment logger that sits next to your training loop and keeps everything tidy:

- **Stats** per step in `stats.jsonl`, nested dicts flattened (`val/acc`).
- **Lossless configs** in `config.yaml` (dataclasses, tuples, paths, enums, arrays, partials).
- **Checkpoints** of anything with `state_dict()` / `load_state_dict()` *or* plain data (jax pytrees, numpy
  arrays, rng keys, sklearn estimators). Torch is optional. Best model tracking, `keep_last`, atomic writes.
- **Groups**: `group="seeds_run", name="test_s0"` gives `runs/seeds_run/test_s0/` on disk and the same
  group / name in Weights & Biases, so multi-seed experiments stay together and filterable.
- **Weights & Biases** sync with resume (branching) and overwrite (deletes only the run's own lineage).
- **Early stopping monitor** that tells your loop when to stop.
- **Provenance**: `env.json` records the command line and the git state of your script's repo (commit, dirty flag, `git_diff.patch`).
- Images, plots, videos and analysis tables in tidy per-step folders.

```
pip install model-sitter                  # core: numpy + pyyaml only
pip install "model-sitter[wandb]"         # + Weights & Biases
pip install "model-sitter[torch]"         # + torch checkpoint backend (device aware)
pip install "model-sitter[media]"         # + video arrays (imageio)
pip install "model-sitter[all]"
```

Every example below is a runnable script in [`examples/`](examples/).

## Quickstart

```python
from mosi import ModelLogger

with ModelLogger("runs/mnist") as log:            # `with` flushes stats and closes W&B at the end
    log.add_config(lr=1e-3, batch_size=64)
    for epoch in range(10):
        loss, acc = train_one_epoch()
        log.add_stats(loss=loss, val={"acc": acc})  # nested dicts become "val/acc"
        log.checkpoint(model=model, optimizer=opt)  # anything with state_dict()
        log.step()                                  # writes the row for this step (and logs to W&B)
```

Call order inside a step: `add_stats` / `add_image` / ... / `checkpoint` / `step()`. The step counter
starts at 1 and `step()` advances it.

## Examples

### 1. No torch, no wandb ([01_basic_numpy.py](examples/01_basic_numpy.py))

```python
class Params:                       # state_dict / load_state_dict is all a checkpointable object needs
    def state_dict(self): return {"w": w, "b": b}
    def load_state_dict(self, d): w[...], b[...] = d["w"], d["b"]

with ModelLogger("runs/basic", checkpoint_backend="pickle", keep_last=3) as log:
    log.add_config(lr=0.1, steps=50)
    for step in range(50):
        ...
        log.add_stats(loss=float((err ** 2).mean()))
        log.checkpoint(params=Params())
        log.step()

with ModelLogger("runs/basic", eval_mode=True, checkpoint_backend="pickle") as log:
    log.load_checkpoint("latest", params=Params())      # restored in place
```

### 2. PyTorch, with resume ([02_torch_resume.py](examples/02_torch_resume.py))

Model, optimizer, scheduler and EMA are just named keyword arguments, there are no predefined slots:

```python
log = ModelLogger("runs/torch", resume=True)
start = log.load_checkpoint("latest", model=model, optimizer=opt, lr_scheduler=sched, ema=ema)["step"]
for step in range(start, total):
    ...
    log.add_stats(loss=loss)                    # tensors are fine, they become floats
    log.checkpoint(model=model, optimizer=opt, lr_scheduler=sched, ema=ema)
    log.step()
```

`resume=True` truncates `stats.jsonl` to the checkpoint's step and continues counting from there.

### 3. JAX / optax ([03_jax.py](examples/03_jax.py))

Pytrees and arrays are stored as they are and come back in the returned dict:

```python
with ModelLogger("runs/jax", checkpoint_backend="pickle") as log:
    ...
    log.add_stats(loss=loss)                                # jax scalars are fine
    log.checkpoint(params=params, opt_state=opt_state)

out = ModelLogger("runs/jax", eval_mode=True, checkpoint_backend="pickle").load_checkpoint("latest")
params, opt_state = out["params"], out["opt_state"]
```

Use `checkpoint_backend="pickle"` explicitly in jax / sklearn projects: `"auto"` picks torch whenever it is installed.

### 4. Seeds as a group ([04_seeds_group.py](examples/04_seeds_group.py))

```python
for seed in range(3):
    with ModelLogger("runs", group="seeds_run", name=f"test_s{seed}",
                     job_type="train", tags=["baseline"], use_wandb=True) as log:
        log.add_config(seed=seed)           # put the seed in the config to filter on it in W&B
        train(log, seed)

from mosi import load_group_stats
stats = load_group_stats("runs/seeds_run")  # {"test_s0": [rows...], "test_s1": [...], ...}
```

```
runs/seeds_run/test_s0/   test_s1/   test_s2/     each has a run.json naming its group
```

In W&B: group by *Group* to compare seeds, filter on `job_type`, `tags` or `config.seed`.
`group` and `name` are plain names you choose (no `/`). A grouped run needs a `name`. `overwrite=True`
only touches the run being re-created, never its siblings.

### 5. Early stopping ([05_early_stopping.py](examples/05_early_stopping.py))

```python
log.enable_monitor(decrease_keys=["val/loss"], patience=3, log_every=5)
for step in ...:
    log.add_stats(val={"loss": val_loss})
    log.checkpoint(model=model)            # metrics default to the stats staged for this step
    log.step()
    if log.should_stop:                    # patience ran out; the monitor never stops training itself
        break

log.load_checkpoint("best", model=model)   # best_model/ is kept up to date and never pruned
```

`increase_keys` / `decrease_keys` take a list or a nested dict (`{"val": ["acc", "f1"]}`). Several keys are
combined as the sum of relative improvements. `tolerance` controls the "catastrophic degradation" warning.

### 6. Weights & Biases

```python
ModelLogger("runs/exp", use_wandb=True, project="my-project")   # or export WANDB_PROJECT=my-project
```

| you pass | local folder | W&B |
|---|---|---|
| nothing special | new run; an existing folder is archived to `<name>_<timestamp>` | new run `<folder>_<id>` |
| `resume=True` | keeps the folder, `load_checkpoint` truncates stats to the checkpoint | a **new branch run** `<name>_resume_from_step_N` (tag `resumed`) with history up to the checkpoint replayed, so it never collides with steps the server already holds |
| `overwrite=True` | wipes stats (asks first, or pass `confirm=lambda msg: True`) | deletes this run's own lineage (`wandb_id.txt`) |
| `eval_mode=True` | read only, extras go to `evaluation/` | disabled |

Use `WANDB_MODE=offline` to log without a network. Media (`images/`, `videos/`, `analysis/`) are synced too.
Each logger holds its own W&B run handle, but `wandb` itself allows one active run per process, so use
`with` / `finish()` between runs of a sweep.

### 7. CLI with tyro ([06_tyro_cli.py](examples/06_tyro_cli.py))

`LoggerConfig` is a plain dataclass, so it nests in your own config and works with tyro (or any dataclass
CLI) without model-sitter depending on it:

```python
@dataclass
class Config:
    lr: float = 1e-3
    logger: LoggerConfig = field(default_factory=LoggerConfig)

cfg = tyro.cli(Config)       # --logger.group seeds_run --logger.name test_s0 --logger.use-wandb
with cfg.logger.build() as log:      # or ModelLogger.from_config(cfg.logger, confirm=...)
    log.add_config(cfg)              # lossless YAML of the whole config
```

`load_config("config.yaml")` rebuilds the config objects. Config classes must live in an importable module
(a class defined in the script you run as `__main__` can only be reloaded from that same script).
`add_config` takes either one object (e.g. a dataclass) and/or keyword values; when both are given,
`config.yaml` holds `{config: <object>, extra: {<values>}}`.

### 8. Images, videos and tables ([07_artifacts.py](examples/07_artifacts.py))

```python
log.add_image("sample.png", pil_image)             # anything with .save(path); plots: .savefig(path)
log.add_plot("curve.png", matplotlib_figure)       # saved under images/ too, logged to W&B as a plot
log.add_video("rollout", frames_uint8_THWC, fps=15)    # needs model-sitter[media]; or pass an .mp4 path
log.add_analysis("weights", {"idx": [0, 1], "val": [0.1, 0.2]})   # CSV (+ W&B table)
```

## On disk

```
runs/seeds_run/test_s0/
  run.log                    everything the logger printed (also DEBUG lines)
  stats.jsonl                one row per step: {"step": 3, "loss": 0.2, "val/acc": 0.9}
  config.yaml
  env.json                   launch command, python/host, git commit / branch / dirty of the script's repo
  git_diff.patch             uncommitted changes at launch (only when the tree was dirty)
  env.resume_<k>.json        same capture for each resume (the original env.json is kept)
  wandb_id.txt               ids of this run's W&B lineage
  checkpoint_<N>/checkpoint.{pt,pkl}
  best_model/                checkpoint of the best step + best_metrics.jsonl
  images/step_<N>/   videos/step_<N>/   analysis/step_<N>/
```

## Checkpoints in detail

`log.checkpoint(**named_things)` stores, per name:

| value | stored as | on `load_checkpoint(step, name=obj)` |
|---|---|---|
| object with `state_dict()` + `load_state_dict()` | its state | `obj.load_state_dict(...)` |
| list / tuple of such objects | `name_1`, `name_2`, ... | each restored in place |
| dict containing such objects | their states, other entries as is | stateful ones restored, the rest returned |
| anything else (pytree, array, scalar, rng key, estimator) | as is | the saved value is returned in the dict |

`load_checkpoint` accepts a step number, `"latest"`, `"best"` or a path to a checkpoint folder, and returns
`{"step": N, ...plain values}`. A key you ask for that was not saved raises a `KeyError` naming it.
Saves are atomic (written to a temp file, then renamed), so a crash never leaves a half-written checkpoint.
`keep_last=N` prunes the oldest `checkpoint_*` folders; `best_model/` is never pruned.

Backends: `checkpoint_backend="auto"` (torch if installed, else pickle), `"torch"`, `"pickle"`, or your own
object with `suffix`, `save(obj, path)` and `load(path, like)` (e.g. orbax / safetensors).

## API

| | |
|---|---|
| `ModelLogger(fpath, overwrite=False, resume=False, eval_mode=False, use_wandb=False, *, group, name, job_type, tags, notes, project, checkpoint_backend, keep_last, confirm, log_level)` | the logger, also a context manager |
| `.add_stats(**stats)`, `.step(step=None)`, `.finish()` | per-step stats |
| `.add_config(obj=None, **values)` | `config.yaml` + W&B config |
| `.checkpoint(metrics=None, **named)`, `.load_checkpoint(step, **named)` | checkpoints |
| `.enable_monitor(increase_keys, decrease_keys, tolerance, patience, log_every)`, `.should_stop` | best model + early stop |
| `.add_image / add_plot / add_video / add_analysis` | artifacts |
| `LoggerConfig`, `ModelLogger.from_config(cfg)` | CLI friendly config |
| `load_group_stats(group_dir)` | stats of every run in a group |
| `dump_config(obj)`, `load_config(path)` | lossless YAML |

## Good to know

- Pickle (and torch with `weights_only=False`) checkpoints run code on load: only load files you trust. The same goes for `load_config`.
- The monitor never stops training by itself, check `log.should_stop`.
- `add_stats` values must be scalars (python / numpy / torch / jax); NaN and inf are stored as `null`.
- The logger is not thread or multi-process safe: log from one process (e.g. rank 0).

## Development

```
pip install -e ".[test]"       # add torch / wandb / tyro / jax to run their tests
pytest
```
