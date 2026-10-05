"""Run grouping: `<root>/<group>/<run>` on disk, mirrored by `group` / `job_type` / `tags` in W&B."""
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

MANIFEST = "group.json"


def resolve_run_dir(fpath: Union[str, Path], group: Optional[str], name: Optional[str]) -> Path:
    """No group: `fpath` is the run dir (flat, as before). With a group: `fpath/group/name`."""
    fpath = Path(fpath)
    if group is None:
        return fpath / name if name else fpath
    for part in (group, name):
        if part and (Path(part).name != part or part in (".", "..")):
            raise ValueError(f"group/name must be plain folder names, got {part!r}")
    if not name:
        raise ValueError("A grouped run needs a `name`, e.g. group='seeds_run', name='test_s0'")
    return fpath / group / name


def update_manifest(group_dir: Path, group: str, run: str, **info: Any) -> None:
    """Records `run` as a member of the group (idempotent, one small json per group)."""
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / MANIFEST
    data = json.loads(path.read_text()) if path.exists() else {
        "group": group, "created": datetime.datetime.now().isoformat(timespec="seconds"), "runs": {}}
    data["runs"].setdefault(run, {}).update({k: v for k, v in info.items() if v is not None})
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_group_stats(group_dir: Union[str, Path]) -> Dict[str, List[dict]]:
    """`{run_name: [stats rows]}` for the runs registered in the group (archived re-runs are skipped)."""
    group_dir = Path(group_dir)
    manifest = group_dir / MANIFEST
    if not manifest.exists():
        raise FileNotFoundError(f"{group_dir} is not a group directory (no {MANIFEST})")
    out: Dict[str, List[dict]] = {}
    for run in json.loads(manifest.read_text())["runs"]:
        stats = group_dir / run / "stats.jsonl"
        out[run] = [json.loads(l) for l in stats.read_text().splitlines() if l.strip()] if stats.exists() else []
    return out
