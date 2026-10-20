"""Run grouping: `<root>/<group>/<run>` on disk, mirrored by `group` / `job_type` / `tags` in W&B."""
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

RUN_INFO = "run.json"


def resolve_run_dir(fpath: Union[str, Path], group: Optional[str], run_name: Optional[str]) -> Path:
    """No group: `fpath/run_name`, or `fpath` itself without a run name. With a group: `fpath/group/run_name`."""
    fpath = Path(fpath)
    if group is None:
        return fpath / run_name if run_name else fpath
    for part in (group, run_name):
        if part and (Path(part).name != part or part in (".", "..")):
            raise ValueError(f"group and run_name must be plain folder names, got {part!r}")
    if not run_name:
        raise ValueError("A grouped run needs a `run_name`, e.g. group='seeds_run', run_name='test_s0'")
    return fpath / group / run_name


def write_run_info(run_dir: Path, group: str, run: str, **info: Any) -> None:
    """Marks `run_dir` as a member of `group` via its own `run.json`.

    Each run only ever writes its own file, so runs of a group can start in parallel without racing
    on a shared manifest."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_INFO
    data = json.loads(path.read_text()) if path.exists() else {
        "group": group, "name": run, "created": datetime.datetime.now().isoformat(timespec="seconds")}
    data.update({k: v for k, v in info.items() if v is not None})
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def list_group_runs(group_dir: Union[str, Path]) -> Dict[str, dict]:
    """`{run_name: run.json contents}` for the runs of the group. Archived re-runs (folder renamed to
    `<name>_<timestamp>`, so it no longer matches the recorded name) are skipped."""
    group_dir = Path(group_dir)
    runs: Dict[str, dict] = {}
    for info_file in sorted(group_dir.glob(f"*/{RUN_INFO}")):
        try:
            info = json.loads(info_file.read_text())
        except (OSError, ValueError):
            continue
        if info.get("name") == info_file.parent.name:
            runs[info["name"]] = info
    return runs


def load_group_stats(group_dir: Union[str, Path]) -> Dict[str, List[dict]]:
    """`{run_name: [stats rows]}` for the runs of the group."""
    group_dir = Path(group_dir)
    runs = list_group_runs(group_dir)
    if not runs:
        raise FileNotFoundError(f"{group_dir} is not a group directory (no run folders with {RUN_INFO})")
    out: Dict[str, List[dict]] = {}
    for run in runs:
        stats = group_dir / run / "stats.jsonl"
        out[run] = [json.loads(l) for l in stats.read_text().splitlines() if l.strip()] if stats.exists() else []
    return out
