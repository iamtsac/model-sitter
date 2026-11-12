import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..layout import CURRENT_LAYOUT, RUN_INFO, read_layout_version
from .plan import MigrationError, Plan
from .v1_to_v2 import migrate_1_to_2

RECENT_SECONDS = 5 * 60

# STEPS[v] upgrades layout v to v + 1. Add `migrate_2_to_3` here when the layout changes again.
STEPS: Dict[int, Callable[[Path, Plan], None]] = {1: migrate_1_to_2}


def is_run(path: Path) -> bool:
    return (path / RUN_INFO).exists() or (path / "stats.jsonl").exists() or any(path.glob("checkpoint_*/"))


def find_runs(path: Path) -> List[Path]:
    """`path` itself when it is a run, else every run below it (runs are not searched for further runs)."""
    if is_run(path):
        return [path]
    runs: List[Path] = []
    for dirpath, dirnames, _ in os.walk(path):
        dirnames.sort()
        keep = []
        for d in dirnames:
            full = Path(dirpath) / d
            if is_run(full):
                runs.append(full)
            else:
                keep.append(d)
        dirnames[:] = keep
    return runs


def refusal_reason(run_dir: Path, now: float) -> Optional[str]:
    """Why the run must not be touched now: it says it is running, or its log changed recently."""
    try:
        info = json.loads((run_dir / RUN_INFO).read_text()) if (run_dir / RUN_INFO).exists() else {}
    except (OSError, ValueError):
        return f"{RUN_INFO} is not readable"
    if info.get("status") == "running":
        return f"{RUN_INFO} status is 'running'"
    for log in (run_dir / "logs" / "run.log", run_dir / "run.log"):
        if log.exists() and now - log.stat().st_mtime < RECENT_SECONDS:
            return f"{log.relative_to(run_dir)} was modified in the last {RECENT_SECONDS // 60} minutes"
    return None


def plan_run(run_dir: Path, version: int, apply: bool) -> Tuple[List[Plan], int]:
    """Plans the chain from `version`; with `apply` each step runs before the next is planned (later steps
    need the tree the earlier ones leave). A dry run plans the first step only. Returns `(plans, reached)`."""
    plans: List[Plan] = []
    while version < CURRENT_LAYOUT:
        plan = Plan(Path(run_dir))
        STEPS[version](Path(run_dir), plan)
        plan.title = f"step {version} -> {version + 1}: {plan.title}"
        plans.append(plan)
        if not apply:
            break
        plan.apply()
        version += 1
    return plans, version


def _print_plan(plan: Plan, out: Callable[[str], None]) -> None:
    out(plan.title)
    width = max((len(a.src) for a in plan.actions), default=0) + 2
    for a in plan.actions:
        if a.kind == "move":
            out(f"  move   {a.src:<{width}} -> {a.dst}")
        elif a.kind == "keep":
            out(f"  keep   {a.src:<{width}} ({a.dst})")
        elif a.kind == "warn":
            out(f"  warn   {a.dst}")
        elif a.kind == "write":
            out(f"  write  {a.src:<{width}} {a.dst}")
        else:
            out(f"  delete {a.src}")


def migrate(path: Path, apply: bool = False, now: Optional[float] = None,
            out: Callable[[str], None] = print) -> Dict[str, int]:
    """Migrates one run or every run below `path`; returns the counts `migrated`, `current`, `refused`."""
    path = Path(path)
    now = time.time() if now is None else now
    counts = {"migrated": 0, "current": 0, "refused": 0}
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a folder")
    runs = find_runs(path)
    for run_dir in runs:
        label = run_dir.name if run_dir == path else str(run_dir.relative_to(path))
        try:
            version = read_layout_version(run_dir)
        except RuntimeError as e:
            out(f"run: {label}   REFUSED: {e}")
            counts["refused"] += 1
            continue
        if version == CURRENT_LAYOUT:
            out(f"run: {label}   layout {version} -> {version}   (current: {CURRENT_LAYOUT})")
            counts["current"] += 1
            continue
        if version > CURRENT_LAYOUT:
            out(f"run: {label}   REFUSED: layout {version} is newer than this mosi ({CURRENT_LAYOUT})")
            counts["refused"] += 1
            continue
        out(f"run: {label}   layout {version} -> {CURRENT_LAYOUT}   (current: {CURRENT_LAYOUT})")
        reason = refusal_reason(run_dir, now)
        if reason:
            out(f"  REFUSED: {reason}; nothing changed")
            counts["refused"] += 1
            continue
        try:
            plans, reached = plan_run(run_dir, version, apply)
        except (MigrationError, OSError) as e:
            out(f"  REFUSED: {e}")
            counts["refused"] += 1
            continue
        for plan in plans:
            _print_plan(plan, out)
        if not apply and version + len(plans) < CURRENT_LAYOUT:
            out(f"  (steps up to layout {CURRENT_LAYOUT} are planned once the first one is applied)")
        counts["migrated"] += 1
    word = "migrated" if apply else "would migrate"
    out(f"{len(runs)} run(s): {counts['migrated']} {word}, {counts['current']} current, {counts['refused']} refused")
    if not apply and counts["migrated"]:
        out("dry run: nothing changed. Re-run with --apply.")
    return counts
