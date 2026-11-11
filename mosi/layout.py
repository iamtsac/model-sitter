import json
from pathlib import Path
from typing import Optional

CURRENT_LAYOUT = 2
RUN_INFO = "run.json"


class LegacyLayoutError(RuntimeError):
    """The run folder was written by an older (or newer) layout than this mosi understands."""


def read_layout_version(run_dir: Path) -> int:
    """`layout_version` of `run.json`; a run without it (or without `run.json`) is layout 1."""
    path = Path(run_dir) / RUN_INFO
    if not path.exists():
        return 1
    try:
        return int(json.loads(path.read_text()).get("layout_version", 1))
    except (OSError, ValueError, AttributeError) as e:
        raise LegacyLayoutError(f"{path} is not readable ({e}); fix or remove it") from e


def check_layout(run_dir: Path) -> None:
    """Raises when `run_dir` holds a run of another layout. An empty or missing folder is fine."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir() or not any(run_dir.iterdir()):
        return
    version = read_layout_version(run_dir)
    if version < CURRENT_LAYOUT:
        raise LegacyLayoutError(
            f"{run_dir} uses the old run layout {version} (current: {CURRENT_LAYOUT}). "
            f"Convert it first: mosi migrate {run_dir} --apply  (omit --apply for a dry run)")
    if version > CURRENT_LAYOUT:
        raise LegacyLayoutError(
            f"{run_dir} uses run layout {version}, newer than this mosi understands ({CURRENT_LAYOUT}); upgrade model-sitter")
