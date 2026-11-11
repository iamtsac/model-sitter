import atexit
import datetime
import dataclasses
import logging
import logging.handlers
import sys
import weakref
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Sequence, Callable, Tuple

from .checkpointing import CheckpointManager, Backend
from .environment import write_environment
from .groups import resolve_run_dir, write_run_info, update_info
from .experiment_manager import ExperimentEnvironment, HistoryManager
from .artifact_manager import ArtifactManager
from .model_monitor import _ModelMonitor


_LIVE: "weakref.WeakSet[Sitter]" = weakref.WeakSet()  # Sitters whose run has not been finished yet
_exception_seen = False
_hooks_installed = False


def _install_exit_hooks() -> None:
    """Once per process: remember that an uncaught exception ended the program (excepthook) and, at exit,
    mark the still unfinished runs `crashed`. The original hook always runs, so the traceback is untouched."""
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        global _exception_seen
        _exception_seen = True
        previous(exc_type, exc, tb)

    sys.excepthook = hook
    atexit.register(_mark_unfinished_crashed)


def _mark_unfinished_crashed() -> None:
    if not _exception_seen:
        return
    for sitter in list(_LIVE):
        try:
            sitter._set_status("crashed")
        except Exception:  # an exit hook must never raise
            pass


@dataclasses.dataclass
class SitterConfig:
    """Plain, CLI-friendly description of a `Sitter` (a dataclass, so tyro/argparse-style tools,
    `dump_config` and W&B configs all handle it without extra dependencies)."""

    fpath: str = "runs"
    """Root directory. The run lives in `<fpath>/<run_name>`, with `group` in `<fpath>/<group>/<run_name>`."""
    group: Optional[str] = None
    """Groups related runs, also the W&B group."""
    run_name: Optional[str] = None
    """The run's folder name, also its W&B run name."""
    job_type: Optional[str] = None
    tags: Tuple[str, ...] = ()
    notes: Optional[str] = None
    project: Optional[str] = None
    """W&B project (falls back to $WANDB_PROJECT)."""
    use_wandb: bool = False
    overwrite: bool = False
    resume: bool = False
    eval_mode: bool = False
    eval_name: Optional[str] = None
    """Evaluation folder name in `<run>/evaluations/` (default `step_<N>_<timestamp>`)."""
    eval_wandb: bool = False
    """Also create a W&B run for an evaluation (off by default)."""
    keep_last: Optional[int] = None
    """Keep only the newest N checkpoints (checkpoints/best is always kept)."""
    checkpoint_backend: str = "auto"
    """auto, torch or pickle."""

    def build(self, **kwargs) -> "Sitter":
        return Sitter(**{**dataclasses.asdict(self), **kwargs})


class Sitter:
    """
    Initializes the Sitter for experiment tracking, checkpointing, and cloud synchronization.

    The logger manages a state machine via 'resume', 'overwrite', and 'eval_mode' to ensure
    local and remote (WandB) data consistency.

    Args:
        fpath (str): The root directory for the experiment. With `group`, this is the root that holds
            all groups and the run lives in `fpath/group/run_name`.
        group (str): Groups related runs (e.g. several seeds of one experiment). Mirrored as the W&B `group`.
        run_name (str): The run's folder name (also its W&B run name), e.g. group="seeds_run", run_name="test_s0".
        job_type (str), tags (Sequence[str]), notes (str): Forwarded to W&B for filtering.
        project (str): W&B project, falls back to the 'WANDB_PROJECT' env variable.
        checkpoint_backend: "auto" (torch if installed, else pickle), "torch", "pickle" or a `Backend`.
        keep_last: Keep only the newest N `checkpoints/step_*` folders (`checkpoints/best/` is always kept).
        confirm: Callable(msg) -> bool replacing the interactive overwrite prompt (for sweeps / CI).
        log_level: Console level ("INFO", "WARNING", "DEBUG"...). The full log is also written to `logs/run.log` of the run (or evaluation) dir.
        overwrite (bool): If True, wipes local stats and deletes the entire WandB lineage 
            from the server. Prompts for user verification if the folder exists.
        resume (bool): If True, attempts to load existing stats and continue the experiment. 
            Creates a new 'branch' in WandB to avoid data overlap.
        eval_mode (bool): If True, opens the existing run read-only for weights; everything the evaluation writes
            (config, logs, launch info, stats, artifacts) goes to `<run>/evaluations/<eval_name>/`, created on the
            first `load_checkpoint` / write. No W&B run unless `eval_wandb=True`. Perfect for inference scripts.
        eval_name (str): Name of that evaluation folder; default `step_<N>_<YYYY-mm-ddTHH-MM-SS>` (N = the loaded
            checkpoint step, only the timestamp if none was loaded). An existing folder is never reused (raises).
        eval_wandb (bool): Create a W&B run for the evaluation too (default False).
        use_wandb (bool): Enables Weights & Biases integration. Requires 'WANDB_PROJECT' 
            env variable.

    Internal State Matrix:
        - Safe Start (False, False, False): Creates a timestamped folder and a new WandB run.
        - Recovery (False, True, False): Re-attaches to 'fpath', loads weights/stats, 
            truncates 'stats.jsonl' to remove post-checkpoint data, branches the WandB 
            run with a lineage-tracking name, and replays history into the new run.
        - Hard Reset (True, True, False): Loads weights from 'fpath', but deletes ALL 
            previous WandB runs in the lineage and wipes local 'stats.jsonl' history.
        - Evaluation (False, False, True): Static access to 'fpath' for weight loading; the run itself is
            never mutated, outputs go to its own `evaluations/<eval_name>/` folder.

    WandB Lineage Tracking:
        The logger keeps a `wandb` block (project, entity, one entry per W&B run started) in 'run.json'. This acts as a 'hit-list'
        for overwrites and a 'breadcrumb' for resumes. Resumed runs are renamed on the
        server to "ParentName_resume_from_step_X" to maintain clear experiment history.
    """

    def __init__(
        self,
        fpath: str,
        overwrite: bool = False,
        resume: bool = False,
        eval_mode: bool = False,
        use_wandb: bool = False,
        *,
        eval_name: Optional[str] = None,
        eval_wandb: bool = False,
        group: Optional[str] = None,
        run_name: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Sequence[str] = (),
        notes: Optional[str] = None,
        project: Optional[str] = None,
        checkpoint_backend: Union[str, Backend] = "auto",
        keep_last: Optional[int] = None,
        confirm: Optional[Callable[[str], bool]] = None,
        log_level: Union[int, str] = "INFO",
    ):
        self.group = group
        run_dir = resolve_run_dir(fpath, group, run_name)
        self.run_name = run_dir.name
        self.group_dir = run_dir.parent if group is not None else None
        self._log_name = f"{group}.{self.run_name}" if group else self.run_name
        self._setup_logging(self._log_name, log_level)

        # Initialize orthogonal managers
        self.env = ExperimentEnvironment(run_dir, overwrite, resume, eval_mode, self.logger, confirm)
        self.history = HistoryManager(self.env.fpath, self.env.overwrite, self.logger)
        self.artifacts = ArtifactManager(self.env.work_dir, self.logger, self.env.eval_mode)
        self.checkpointer = CheckpointManager(self.env.fpath, self.logger, checkpoint_backend, keep_last)

        # Internal control state
        self._step = 1
        self._loaded_step: Optional[int] = None
        self._monitor = None
        self._use_rollback = False
        self.telemetry = None
        self._info_path: Optional[Path] = None  # run.json (training) / eval.json (evaluation)
        self._status_set = False
        self._start_args = dict(group=group, run_name=run_name, job_type=job_type, tags=list(tags), notes=notes, project=project,
                                use_wandb=use_wandb, eval_name=eval_name, eval_wandb=eval_wandb)

        if self.env.eval_mode:
            # the evaluation folder is named after the checkpoint step, so it opens at the first load / write;
            # until then the log lines wait in memory
            self._buffer = logging.handlers.MemoryHandler(capacity=10_000, flushLevel=logging.CRITICAL + 1)
            self.logger.addHandler(self._buffer)
        else:
            self._open_work_dir()
        self.logger.info(f"Initialized Experiment at: {self.env.fpath}")

    def _open_work_dir(self) -> None:
        """Training: sets up run.json, logs, launch info and W&B at construction. Evaluation: the same inside
        `evaluations/<eval_name>/`, once the loaded checkpoint step is known (or at the first write)."""
        a = self._start_args
        env = self.env
        if env.eval_mode:
            self._eval_name = a["eval_name"] or self._default_eval_name()
            env.open_eval(self._eval_name)
        self.history.bind(env.work_dir)
        self.artifacts.work_dir = env.work_dir
        self._attach_log_file()

        if env.eval_mode:
            self._info_path = env.work_dir / "eval.json"
            update_info(self._info_path, name=self._eval_name, run=self.run_name,
                        created=datetime.datetime.now().isoformat(timespec="seconds"),
                        checkpoint=None, status="running", finished=None)
        else:
            self._info_path = env.fpath / "run.json"
            write_run_info(env.fpath, a["group"], self.run_name, job_type=a["job_type"], tags=a["tags"] or None)
            if env.resume or env.overwrite:
                update_info(self._info_path, status="running", finished=None)
        self._status_set = False
        _install_exit_hooks()
        _LIVE.add(self)

        try:
            write_environment(env.launch_dir, keep_previous=env.resume and not env.overwrite, logger=self.logger)
        except OSError as e:
            self.logger.warning(f"Provenance: could not write env.json / git_diff.patch: {e}")

        group, name = a["group"], self.run_name
        if a["use_wandb"] and (not env.eval_mode or a["eval_wandb"]):
            from .telemetry import WandbManager
            prefix = f"{group}_{name}" if group else name
            if env.eval_mode:
                wandb_name, prefix = f"{name}_{self._eval_name}", f"{prefix}_{self._eval_name}"
            else:
                wandb_name = name if group or a["run_name"] else None
            self.telemetry = WandbManager(
                self._info_path, env.logs_dir, env.overwrite, env.resume, self.logger,
                project=a["project"], group=group, run_name=wandb_name,
                job_type=a["job_type"], tags=a["tags"], notes=a["notes"], id_prefix=prefix,
            )

    def _default_eval_name(self) -> str:
        stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        return f"step_{self._loaded_step}_{stamp}" if self._loaded_step is not None else stamp

    def _ensure_open(self) -> None:
        """Evaluation: creates its folder (once). Training: nothing to do."""
        if self.env.eval_mode and self.env.work_dir is None:
            self._open_work_dir()

    @property
    def current_step(self) -> int:
        """The step the next `add_*` / `checkpoint` call belongs to (read-only; change it with `step()`)."""
        return self._step

    @property
    def loaded_step(self) -> Optional[int]:
        """Step of the checkpoint last restored by `load_checkpoint`, None before any."""
        return self._loaded_step

    def artifact_dir(self, kind: str) -> Path:
        """Folder for files of your own: `<run>/artifacts/step_<current step>/<kind>/` in training,
        `<evaluation dir>/artifacts/<kind>/` in an evaluation. Created, `kind` is a plain folder name."""
        self._ensure_open()
        return self.artifacts.artifact_dir(kind, self._step)

    def step(self, step: Optional[int] = None) -> None:
        if len(self.history._stage_buffer):
            entry = self.history.flush_buffer_to_jsonl(self._step)
            if self.telemetry: self.telemetry.log_metrics(entry, self._step)

        self._step = step if step is not None else self._step + 1

    def finish(self) -> None:
        """Flushes pending stats, marks the run `finished` and closes the W&B run (needed between runs of a sweep)."""
        self._finish("finished")

    def _finish(self, status: str) -> None:
        self.step()
        self._set_status(status)
        if self.telemetry: self.telemetry.finish()
        for h in list(self.logger.handlers):
            if isinstance(h, (logging.FileHandler, logging.handlers.MemoryHandler)):
                h.close()
                self.logger.removeHandler(h)

    def _set_status(self, status: str) -> None:
        """Writes `status` (+ `finished` time) into run.json / eval.json; the first final status wins."""
        if self._info_path is None or self._status_set:
            return
        self._status_set = True
        _LIVE.discard(self)
        try:
            update_info(self._info_path, status=status,
                        finished=datetime.datetime.now().isoformat(timespec="seconds"))
        except OSError as e:
            self.logger.warning(f"Could not write the status to {self._info_path}: {e}")

    def __enter__(self) -> "Sitter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None or (issubclass(exc_type, SystemExit) and not exc.code):
            self.finish()
            return
        try:  # the original exception must come out of the `with` untouched
            self._finish("crashed")
        except Exception as e:
            self.logger.warning(f"Could not close the run cleanly after {exc_type.__name__}: {e!r}")

    @property
    def should_stop(self) -> bool:
        """True once the monitor's patience ran out: `if log.should_stop: break`."""
        return bool(self._monitor and self._monitor.should_stop)

    def add_stats(self, **stats) -> None:
        self._ensure_open()
        flat_stats = self.history.stage_metrics(stats)
        if self.telemetry: self.telemetry.define_metrics(flat_stats)
        if self.env.eval_mode:
            entry = self.history.flush_buffer_to_jsonl(self._step)
            if self.telemetry: self.telemetry.log_metrics(entry, self._step)

    def checkpoint(self, metrics: dict = None, **kwargs) -> None:
        if self.env.eval_mode:
            return

        is_best = False
        monitor_state = None

        if getattr(self, "_use_monitor", False) and self._monitor:
            if metrics is None and self.history._stage_buffer:
                metrics = dict(self.history._stage_buffer)
            if metrics is not None:
                is_best = self._monitor(step=self._step, metrics=metrics)
                monitor_state = self._monitor.state()
            else:
                self.logger.debug("Monitor is enabled, but no metrics were provided to checkpoint().")

        self.checkpointer.save(
            step=self._step, 
            monitor_state=monitor_state, 
            is_best=is_best, 
            **kwargs
        )

    def load_checkpoint(self, step: Union[int, str], **kwargs) -> dict:
        checkpoint, ckpt_path = self.checkpointer.load(step, **kwargs)
        loaded_step = checkpoint.get("step", step)
        if isinstance(loaded_step, int):
            self._loaded_step = loaded_step
        if self.env.eval_mode:
            self._ensure_open()
            update_info(self._info_path, checkpoint={"path": str(ckpt_path), "step": self._loaded_step})

        if self.env.resume:
            self._step = loaded_step + 1
            history = self.history.recover_state(loaded_step, self.env.overwrite)
            if self.telemetry and not self.env.overwrite:
                self.telemetry.append_resume_step(loaded_step)
                self.telemetry.replay_history(history)
        else:
            self._step = loaded_step

        if getattr(self, '_use_monitor', False) and self._monitor:
            monitor_state = checkpoint.get('monitor_state')
            if monitor_state:
                self._monitor.load_state(monitor_state)

        self.logger.info(f"State restored from step {loaded_step}")
        return checkpoint

    def enable_monitor(
        self, 
        increase_keys: Union[list, dict] = None, 
        decrease_keys: Union[list, dict] = None, 
        tolerance: float = 0.1, 
        patience: int = -1,
        log_every: int = 1,
    ):
        """`log_every=N` prints only every Nth "New Best" line (the monitor still tracks every one)."""
        self._use_monitor = True
        self._monitor = _ModelMonitor(
            self.logger, increase_keys, decrease_keys, tolerance, patience, log_every
        )
        mode = "Infinite" if patience == -1 else str(patience)
        self.logger.info(f"Monitor Enabled (Patience: {mode}, Tolerance: {tolerance*100:.0f}%)")

    def add_config(self, config: Any = None, **values):
        """`config` is any object `dump_config` can store (e.g. a dataclass), `values` merge as a dict."""
        self._ensure_open()
        for c in (config, values or None):
            if c is None:
                continue
            self.artifacts.save_config(c)
            if self.telemetry:
                flat = dataclasses.asdict(c) if dataclasses.is_dataclass(c) else c
                self.telemetry.update_config(ArtifactManager._sanitize_config(flat))

    def add_plot(self, name: str, plot: Any, **kwargs):
        self._ensure_open()
        path = self.artifacts.save_media(name, plot, self._step, **kwargs)
        if self.telemetry and path: self.telemetry.log_plot(name, plot, self._step)

    def add_image(self, name: str, image: Any, **kwargs):
        self._ensure_open()
        path = self.artifacts.save_media(name, image, self._step, **kwargs)
        if self.telemetry and path: self.telemetry.log_image(name, path, self._step)

    def add_video(self, name: str, video: Any, **imageio_kwargs):
        self._ensure_open()
        path = self.artifacts.save_video(name, video, self._step, **imageio_kwargs)
        if self.telemetry and path: self.telemetry.log_video(name, path, self._step)

    def add_analysis(self, name: str, data: Dict[str, List]):
        self._ensure_open()
        self.artifacts.save_csv(name, data, self._step)
        if self.telemetry: self.telemetry.log_table(name, data, self._step)

    @classmethod
    def from_config(cls, cfg: "SitterConfig", **kwargs) -> "Sitter":
        """Builds the Sitter from a `SitterConfig` (e.g. one parsed by tyro); `kwargs` add non-CLI args like `confirm`."""
        return cfg.build(**kwargs)

    def _setup_logging(self, name: str, level: Union[int, str] = "INFO") -> None:
        self.logger = logging.getLogger(f"mosi.{name}")
        self.logger.setLevel(logging.DEBUG)  # handlers filter; the file keeps everything
        self.logger.propagate = False
        for h in list(self.logger.handlers):  # a re-created logger of the same name starts clean
            h.close()
            self.logger.removeHandler(h)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level.upper() if isinstance(level, str) else level)
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)

    def _attach_log_file(self) -> None:
        """`logs/run.log` of the work dir. In an evaluation, the lines logged before its folder existed go first."""
        fh = logging.FileHandler(self.env.logs_dir / "run.log", mode="a" if self.env.resume else "w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        buffer = getattr(self, "_buffer", None)
        if buffer is not None:
            buffer.setTarget(fh)
            buffer.flush()
            self.logger.removeHandler(buffer)
            buffer.close()
        self.logger.addHandler(fh)
