# 🍼 Model-Sitter

A small experiment logger that sits next to your training loop and keeps everything tidy:

- **Stats** per step in `logs/stats.jsonl`, nested dicts flattened (`val/acc`).
- **Lossless configs** in `config.yaml` (dataclasses, tuples, paths, enums, arrays, partials).
- **Checkpoints** of anything with `state_dict()` / `load_state_dict()` *or* plain data (jax pytrees, numpy
  arrays, rng keys, sklearn estimators). Torch is optional. Best model tracking, `keep_last`, atomic writes.
- **Groups**: `group="seeds_run", run_name="test_s0"` gives `runs/seeds_run/test_s0/` on disk and the same
  group / name in Weights & Biases, so multi-seed experiments stay together and filterable.
- **Weights & Biases** sync with resume (branching) and overwrite (deletes only the run's own lineage).
- **Early stopping monitor** that tells your loop when to stop.
- **Provenance** in `launch/`: `env.json` records the command line, git state, GPU, package versions and pixi lock; `git_diff.patch` holds the uncommitted changes, untracked files included.
- Images, plots, videos and analysis tables (and your own files via `artifact_dir`) in tidy per-step folders.
- **Evaluations** nested in the run (`evaluations/<eval_name>/`), run status in `run.json`, `mosi migrate` for older runs.

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
from mosi import Sitter

with Sitter("runs/mnist") as sitter:            # `with` flushes stats and closes W&B at the end
    sitter.add_config(lr=1e-3, batch_size=64)
    for epoch in range(10):
        loss, acc = train_one_epoch()
        sitter.add_stats(loss=loss, val={"acc": acc})  # nested dicts become "val/acc"
        sitter.checkpoint(model=model, optimizer=opt)  # anything with state_dict()
        sitter.step()                                  # writes the row for this step (and logs to W&B)
```

Call order inside a step: `add_stats` / `add_image` / ... / `checkpoint` / `step()`. The step counter
starts at 1 and `step()` advances it.

## Examples

### 1. No torch, no wandb ([01_basic_numpy.py](examples/01_basic_numpy.py))

```python
class Params:                       # state_dict / load_state_dict is all a checkpointable object needs
    def state_dict(self): return {"w": w, "b": b}
    def load_state_dict(self, d): w[...], b[...] = d["w"], d["b"]

with Sitter("runs/basic", checkpoint_backend="pickle", keep_last=3) as sitter:
    sitter.add_config(lr=0.1, steps=50)
    for step in range(50):
        ...
        sitter.add_stats(loss=float((err ** 2).mean()))
        sitter.checkpoint(params=Params())
        sitter.step()

with Sitter("runs/basic", eval_mode=True, checkpoint_backend="pickle") as sitter:
    sitter.load_checkpoint("latest", params=Params())      # restored in place
```

### 2. PyTorch, with resume ([02_torch_resume.py](examples/02_torch_resume.py))

Model, optimizer, scheduler and EMA are just named keyword arguments, there are no predefined slots:

```python
sitter = Sitter("runs/torch", resume=True)
start = sitter.load_checkpoint("latest", model=model, optimizer=opt, lr_scheduler=sched, ema=ema)["step"]
for step in range(start, total):
    ...
    sitter.add_stats(loss=loss)                    # tensors are fine, they become floats
    sitter.checkpoint(model=model, optimizer=opt, lr_scheduler=sched, ema=ema)
    sitter.step()
```

`resume=True` truncates `stats.jsonl` to the checkpoint's step and continues counting from there.

### 3. JAX / optax ([03_jax.py](examples/03_jax.py))

Pytrees and arrays are stored as they are and come back in the returned dict:

```python
with Sitter("runs/jax", checkpoint_backend="pickle") as sitter:
    ...
    sitter.add_stats(loss=loss)                                # jax scalars are fine
    sitter.checkpoint(params=params, opt_state=opt_state)

out = Sitter("runs/jax", eval_mode=True, checkpoint_backend="pickle").load_checkpoint("latest")
params, opt_state = out["params"], out["opt_state"]
```

Use `checkpoint_backend="pickle"` explicitly in jax / sklearn projects: `"auto"` picks torch whenever it is installed.

### 4. Seeds as a group ([04_seeds_group.py](examples/04_seeds_group.py))

```python
for seed in range(3):
    with Sitter("runs", group="seeds_run", run_name=f"test_s{seed}",
                     job_type="train", tags=["baseline"], use_wandb=True) as sitter:
        sitter.add_config(seed=seed)           # put the seed in the config to filter on it in W&B
        train(sitter, seed)

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
sitter.enable_monitor(decrease_keys=["val/loss"], patience=3, log_every=5)
for step in ...:
    sitter.add_stats(val={"loss": val_loss})
    sitter.checkpoint(model=model)            # metrics default to the stats staged for this step
    sitter.step()
    if sitter.should_stop:                    # patience ran out; the monitor never stops training itself
        break

sitter.load_checkpoint("best", model=model)   # checkpoints/best/ is kept up to date and never pruned
```

`increase_keys` / `decrease_keys` take a list or a nested dict (`{"val": ["acc", "f1"]}`). Several keys are
combined as the sum of relative improvements. `tolerance` controls the "catastrophic degradation" warning.

### 6. Weights & Biases

```python
Sitter("runs/exp", use_wandb=True, project="my-project")   # or export WANDB_PROJECT=my-project
```

| you pass | local folder | W&B |
|---|---|---|
| nothing special | new run; an existing folder is archived to `<name>_<timestamp>` | new run `<folder>_<id>` |
| `resume=True` | keeps the folder, `load_checkpoint` truncates stats to the checkpoint | a **new branch run** `<name>_resume_from_step_N` (tag `resumed`) with history up to the checkpoint replayed, so it never collides with steps the server already holds |
| `overwrite=True` | wipes stats (asks first, or pass `confirm=lambda msg: True`) | deletes this run's own lineage (the `wandb.runs` listed in `run.json`, in the project stored there) |
| `eval_mode=True` | run is read only, outputs go to `evaluations/<eval_name>/` | none unless `eval_wandb=True` (then its ids go in that `eval.json`) |

Use `WANDB_MODE=offline` to log without a network. Media (`images`, `videos`, `analysis` artifacts) are synced too. The ids of the W&B runs a run started (one per start / resume) are kept in the `wandb` block of `run.json`.
Each logger holds its own W&B run handle, but `wandb` itself allows one active run per process, so use
`with` / `finish()` between runs of a sweep.

### 7. CLI with tyro ([06_tyro_cli.py](examples/06_tyro_cli.py))

`SitterConfig` is a plain dataclass, so it nests in your own config and works with tyro (or any dataclass
CLI) without model-sitter depending on it:

```python
@dataclass
class Config:
    lr: float = 1e-3
    logger: SitterConfig = field(default_factory=SitterConfig)

cfg = tyro.cli(Config)       # --logger.group seeds_run --logger.run-name test_s0 --logger.use-wandb
with cfg.logger.build() as sitter:      # or Sitter.from_config(cfg.logger, confirm=...)
    sitter.add_config(cfg)              # lossless YAML of the whole config
```

`load_config("config.yaml")` rebuilds the config objects. Config classes must live in an importable module
(a class defined in the script you run as `__main__` can only be reloaded from that same script).
`add_config` takes either one object (e.g. a dataclass) and/or keyword values; when both are given,
`config.yaml` holds `{config: <object>, extra: {<values>}}`.

### 8. Images, videos and tables ([07_artifacts.py](examples/07_artifacts.py))

```python
sitter.add_image("sample.png", pil_image)             # anything with .save(path); plots: .savefig(path)
sitter.add_plot("curve.png", matplotlib_figure)       # saved under artifacts/step_<N>/images/ too, logged to W&B as a plot
sitter.add_video("rollout", frames_uint8_THWC, fps=15)    # needs model-sitter[media]; or pass an .mp4 path
sitter.add_analysis("weights", {"idx": [0, 1], "val": [0.1, 0.2]})   # CSV (+ W&B table)
```

## On disk

```
runs/seeds_run/test_s0/
├── run.json        group, name, created, job_type, tags, wandb, status, finished, layout_version
├── config.yaml
├── launch/         env.json, git_diff.patch; resume_<k>/ for each resume (same two files)
├── logs/           run.log, stats.jsonl, wandb/
├── checkpoints/    step_<N>/checkpoint.{pt,pkl}, best/ (+ best_metrics.jsonl)
├── artifacts/step_<N>/<kind>/      kind = images | videos | analysis | anything passed to artifact_dir
└── evaluations/<eval_name>/
    ├── eval.json   checkpoint evaluated (path + step), created, status, finished (+ wandb block if eval_wandb)
    ├── config.yaml
    ├── launch/     env.json, git_diff.patch
    ├── logs/       run.log, stats.jsonl
    └── artifacts/<kind>/           (no step level inside an evaluation)
```

- `run.json`: `status` is `running` from the start, `finished` after `finish()` / a clean `with` block and `crashed`
  when the process ends through an exception (`finished` is the end time; a killed process stays `running`).
  `layout_version` is the on-disk layout (2). `wandb` is `{"project", "entity", "runs": [{"id", "name", "started",
  "resumed_from_step", "url"}, ...]}`, oldest first: one entry per start or resume, all deleted on `overwrite=True`
  (in the stored project / entity).
- `launch/env.json`: `created`, `command` (argv, cwd), `python`, `host`, `git` (root, commit, branch, dirty,
  `untracked` (first 50), `diff_bytes` = full diff size, `diff_truncated`), `gpu` (names, driver, CUDA; from torch.cuda,
  else nvidia-smi, else null), `packages` (versions of torch, numpy, jax, ... and model-sitter), `pixi` (`pixi.lock`
  sha256 and `PIXI_ENVIRONMENT_NAME`). A part that cannot be captured is null and logs a WARNING; it never breaks the run.
- `launch/git_diff.patch`: `git diff HEAD` including untracked files (added intent-to-add in a temporary index, so
  your real index is never touched), cut at 1 MB. Only written when the tree is dirty. A resume keeps the original
  and writes `launch/resume_<k>/`.
- `logs/run.log`: everything the logger printed (also DEBUG lines). `logs/stats.jsonl`: one row per step,
  `{"step": 3, "loss": 0.2, "val/acc": 0.9}`.

### Evaluations

`Sitter(run, eval_mode=True)` never writes into the run itself. Its outputs go to
`<run>/evaluations/<eval_name>/`, created at the first `load_checkpoint` or write. `eval_name` defaults to
`step_<N>_<YYYY-mm-ddTHH-MM-SS>` (N = the loaded checkpoint step, only the timestamp when nothing was loaded);
an existing folder is never reused (`FileExistsError`). An evaluation has no W&B run unless `eval_wandb=True`.

### Artifacts

`sitter.artifact_dir("rollouts")` returns (and creates) `<run>/artifacts/step_<current step>/rollouts/` in training and
`<evaluation>/artifacts/rollouts/` in an evaluation, for any files of your own. `add_image` / `add_plot` / `add_video` /
`add_analysis` use it with the kinds `images`, `videos`, `analysis`. `sitter.current_step` and `sitter.loaded_step`
are read-only.

## Migrating old runs

Every run records `layout_version` in `run.json` (a run without it is layout 1). Opening a run of another layout
(`resume`, `eval_mode`, `overwrite`, `load_group_stats`) raises `LegacyLayoutError` telling you to migrate:

```
mosi migrate runs/old_run            # dry run: prints each run and every move / write / delete / keep, changes nothing
mosi migrate runs --apply            # a folder: every run below it is migrated
```

Run folders are found as folders holding `run.json`, `checkpoint_*/` or `stats.jsonl`. Files the step does not know
are reported `keep` and never touched. A run is refused (and left alone) when its `run.json` says `running` or its
`run.log` changed in the last 5 minutes. The last line counts migrated / current / refused; a second run reports
`current`. Steps are chained (`migrate_1_to_2`, ...) up to `CURRENT_LAYOUT` (one place: `mosi/layout.py`); each step is its own file in `mosi/migrations/` (`v1_to_v2.py`, ...), registered in `STEPS` in `mosi/migrations/runner.py`.
In layout 1 -> 2, `wandb_id.txt` becomes the `wandb` block (project from `WANDB_PROJECT` at migration time) and
`evaluation/` becomes `evaluations/legacy/`.

**A change of the layout ships its migration step and its test in the same commit.**

## Checkpoints in detail

`sitter.checkpoint(**named_things)` stores, per name:

| value | stored as | on `load_checkpoint(step, name=obj)` |
|---|---|---|
| object with `state_dict()` + `load_state_dict()` | its state | `obj.load_state_dict(...)` |
| list / tuple of such objects | `name_1`, `name_2`, ... | each restored in place |
| dict containing such objects | their states, other entries as is | stateful ones restored, the rest returned |
| anything else (pytree, array, scalar, rng key, estimator) | as is | the saved value is returned in the dict |

`load_checkpoint` accepts a step number, `"latest"`, `"best"` or a path to a checkpoint folder, and returns
`{"step": N, ...plain values}`. A key you ask for that was not saved raises a `KeyError` naming it.
Saves are atomic (written to a temp file, then renamed), so a crash never leaves a half-written checkpoint.
`keep_last=N` prunes the oldest `checkpoints/step_<N>` folders; `checkpoints/best/` is never pruned.
Checkpoints live in `<run>/checkpoints/step_<N>/checkpoint.{pt,pkl}`, the best one in `checkpoints/best/` (with
`best_metrics.jsonl`). Loading works on runs of the current layout only; older runs need `mosi migrate` first.

Backends: `checkpoint_backend="auto"` (torch if installed, else pickle), `"torch"`, `"pickle"`, or your own
object with `suffix`, `save(obj, path)` and `load(path, like)` (e.g. orbax / safetensors).

## API

| | |
|---|---|
| `Sitter(fpath, overwrite=False, resume=False, eval_mode=False, use_wandb=False, *, eval_name, eval_wandb, group, run_name, job_type, tags, notes, project, checkpoint_backend, keep_last, confirm, log_level)` | the main object, also a context manager |
| `.add_stats(**stats)`, `.step(step=None)`, `.finish()` | per-step stats |
| `.add_config(obj=None, **values)` | `config.yaml` + W&B config |
| `.checkpoint(metrics=None, **named)`, `.load_checkpoint(step, **named)` | checkpoints |
| `.enable_monitor(increase_keys, decrease_keys, tolerance, patience, log_every)`, `.should_stop` | best model + early stop |
| `.add_image / add_plot / add_video / add_analysis`, `.artifact_dir(kind)` | artifacts |
| `.current_step`, `.loaded_step` | read-only: the current step, the step of the last loaded checkpoint |
| `SitterConfig`, `Sitter.from_config(cfg)` | CLI friendly config |
| `load_group_stats(group_dir)` | stats of every run in a group |
| `dump_config(obj)`, `load_config(path)` | lossless YAML |

## Good to know

- Pickle (and torch with `weights_only=False`) checkpoints run code on load: only load files you trust. The same goes for `load_config`.
- The monitor never stops training by itself, check `sitter.should_stop`.
- `add_stats` values must be scalars (python / numpy / torch / jax); NaN and inf are stored as `null`.
- The logger is not thread or multi-process safe: log from one process (e.g. rank 0).

## Development

```
pip install -e ".[test]"       # add torch / wandb / tyro / jax to run their tests
pytest
```
