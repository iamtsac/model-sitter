import sys
import dataclasses
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Sequence, Callable

from .checkpointing import CheckpointManager, Backend
from .groups import resolve_run_dir, update_manifest
from .logger_config import LoggerConfig
from .experiment_manager import ExperimentEnvironment, HistoryManager
from .artifact_manager import ArtifactManager
from .model_monitor import _ModelMonitor

class ModelLogger:
    """
    Initializes the ModelLogger for experiment tracking, checkpointing, and cloud synchronization.

    The logger manages a state machine via 'resume', 'overwrite', and 'eval_mode' to ensure
    local and remote (WandB) data consistency.

    Args:
        fpath (str): The root directory for the experiment. With `group`, this is the root that holds
            all groups and the run lives in `fpath/group/name`.
        group (str): Groups related runs (e.g. several seeds of one experiment). Mirrored as the W&B `group`.
        name (str): Run name inside the group (also the W&B run name), e.g. group="seeds_run", name="test_s0".
        job_type (str), tags (Sequence[str]), notes (str): Forwarded to W&B for filtering.
        project (str): W&B project, falls back to the 'WANDB_PROJECT' env variable.
        checkpoint_backend: "auto" (torch if installed, else pickle), "torch", "pickle" or a `Backend`.
        keep_last: Keep only the newest N `checkpoint_*` folders (`best_model/` is always kept).
        confirm: Callable(msg) -> bool replacing the interactive overwrite prompt (for sweeps / CI).
        log_level: Console level ("INFO", "WARNING", "DEBUG"...). The full log is also written to `run.log` in the run dir.
        overwrite (bool): If True, wipes local stats and deletes the entire WandB lineage 
            from the server. Prompts for user verification if the folder exists.
        resume (bool): If True, attempts to load existing stats and continue the experiment. 
            Creates a new 'branch' in WandB to avoid data overlap.
        eval_mode (bool): If True, locks onto the existing folder without creating 
            timestamps and disables all WandB logging. Perfect for inference scripts.
        use_wandb (bool): Enables Weights & Biases integration. Requires 'WANDB_PROJECT' 
            env variable.

    Internal State Matrix:
        - Safe Start (False, False, False): Creates a timestamped folder and a new WandB run.
        - Recovery (False, True, False): Re-attaches to 'fpath', loads weights/stats, 
            truncates 'stats.yaml' to remove post-checkpoint data, branches the WandB 
            run with a lineage-tracking name, and replays history into the new run.
        - Hard Reset (True, True, False): Loads weights from 'fpath', but deletes ALL 
            previous WandB runs in the lineage and wipes local 'stats.yaml' history.
        - Evaluation (False, False, True): Static access to 'fpath' for weight loading; 
            no logging or directory mutation.

    WandB Lineage Tracking:
        The logger maintains a 'wandb_id.txt' file in 'fpath'. This acts as a 'hit-list'
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
        group: Optional[str] = None,
        name: Optional[str] = None,
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
        run_dir = resolve_run_dir(fpath, group, name)
        self.name = run_dir.name
        self.group_dir = run_dir.parent if group is not None else None
        self._setup_logging(f"{group}.{self.name}" if group else self.name, log_level)

        # Initialize orthogonal managers
        self.env = ExperimentEnvironment(run_dir, overwrite, resume, eval_mode, self.logger, confirm)
        self._attach_log_file()
        self.history = HistoryManager(self.env.work_dir, self.env.overwrite, self.env.eval_mode, self.logger)
        self.artifacts = ArtifactManager(self.env.work_dir, self.logger)
        self.checkpointer = CheckpointManager(self.env.fpath, self.logger, checkpoint_backend, keep_last)

        if self.group_dir is not None and not self.env.eval_mode:
            update_manifest(self.group_dir, group, self.name, job_type=job_type, tags=list(tags) or None)

        self.telemetry = None
        if not self.env.eval_mode and use_wandb:
            from .telemetry import WandbManager
            self.telemetry = WandbManager(
                self.env.fpath, self.env.overwrite, self.env.resume, self.logger,
                project=project, group=group, name=self.name if group or name else None,
                job_type=job_type, tags=tags, notes=notes,
                id_prefix=f"{group}_{self.name}" if group else None,
            )

        # Internal control state
        self._step = 1
        self._monitor = None
        self._use_rollback = False
        self.logger.info(f"Initialized Experiment at: {self.env.fpath}")

    def step(self, step: Optional[int] = None) -> None:
        if len(self.history._stage_buffer):
            entry = self.history.flush_buffer_to_jsonl(self._step)
            if self.telemetry: self.telemetry.log_metrics(entry, self._step)

        self._step = step if step is not None else self._step + 1

    def finish(self) -> None:
        """Flushes pending stats and closes the W&B run (needed between runs of a sweep)."""
        self.step()
        if self.telemetry: self.telemetry.finish()
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
                self.logger.removeHandler(h)

    def __enter__(self) -> "ModelLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()

    @property
    def should_stop(self) -> bool:
        """True once the monitor's patience ran out: `if log.should_stop: break`."""
        return bool(self._monitor and self._monitor.should_stop)

    def add_stats(self, **stats) -> None:
        flat_stats = self.history.stage_metrics(stats)
        if self.telemetry: self.telemetry.define_metrics(flat_stats)
        if self.env.eval_mode: self.history.flush_buffer_to_jsonl(self._step)

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
        checkpoint, _path = self.checkpointer.load(step, **kwargs)
        loaded_step = checkpoint.get("step", step)

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
        if self.env.eval_mode:
            return
        for c in (config, values or None):
            if c is None:
                continue
            self.artifacts.save_config(c)
            if self.telemetry:
                flat = dataclasses.asdict(c) if dataclasses.is_dataclass(c) else c
                self.telemetry.update_config(ArtifactManager._sanitize_config(flat))

    def add_plot(self, name: str, plot: Any, **kwargs):
        path = self.artifacts.save_media(name, plot, self._step, **kwargs)
        if self.telemetry and path: self.telemetry.log_plot(name, plot, self._step)

    def add_image(self, name: str, image: Any, **kwargs):
        path = self.artifacts.save_media(name, image, self._step, **kwargs)
        if self.telemetry and path: self.telemetry.log_image(name, path, self._step)

    def add_video(self, name: str, video: Any, **imageio_kwargs):
        path = self.artifacts.save_video(name, video, self._step, **imageio_kwargs)
        if self.telemetry and path: self.telemetry.log_video(name, path, self._step)

    def add_analysis(self, name: str, data: Dict[str, List]):
        self.artifacts.save_csv(name, data, self._step)
        if self.telemetry: self.telemetry.log_table(name, data, self._step)

    @classmethod
    def from_config(cls, cfg: "LoggerConfig", **kwargs) -> "ModelLogger":
        """Builds the logger from a `LoggerConfig` (e.g. one parsed by tyro); `kwargs` add non-CLI args like `confirm`."""
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
        if self.env.eval_mode:
            return
        fh = logging.FileHandler(self.env.work_dir / "run.log", mode="a" if self.env.resume else "w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)
