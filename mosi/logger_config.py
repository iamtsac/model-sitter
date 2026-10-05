import dataclasses
from typing import Optional, Tuple


@dataclasses.dataclass
class LoggerConfig:
    """Plain, CLI-friendly description of a `ModelLogger` (a dataclass, so tyro/argparse-style tools,
    `dump_config` and W&B configs all handle it without extra dependencies)."""

    fpath: str = "runs"
    """Root directory. With `group` the run lives in `<fpath>/<group>/<name>`."""
    group: Optional[str] = None
    """Groups related runs, also the W&B group."""
    name: Optional[str] = None
    """Run name inside the group, also the W&B run name."""
    job_type: Optional[str] = None
    tags: Tuple[str, ...] = ()
    notes: Optional[str] = None
    project: Optional[str] = None
    """W&B project (falls back to $WANDB_PROJECT)."""
    use_wandb: bool = False
    overwrite: bool = False
    resume: bool = False
    eval_mode: bool = False
    keep_last: Optional[int] = None
    """Keep only the newest N checkpoints (best_model is always kept)."""
    checkpoint_backend: str = "auto"
    """auto, torch or pickle."""

    def build(self, **kwargs) -> "ModelLogger":
        from .model_logger import ModelLogger
        return ModelLogger(**{**dataclasses.asdict(self), **kwargs})
