import datetime
import json
import os
import re
from pathlib import Path
from typing import Dict

from ..layout import RUN_INFO
from .plan import Plan

ARTIFACT_KINDS = ("images", "videos", "analysis")
_STEP = re.compile(r"step_(\d+)$")


def migrate_1_to_2(run_dir: Path, plan: Plan) -> None:
    """Layout 1 -> 2: group root files, nest evaluations (see README "Migrating old runs")."""
    plan.title = "group root files, nest evaluations"
    root = Path(run_dir)
    top = sorted(p.name for p in root.iterdir())
    for name in top:
        path = root / name
        if m := re.fullmatch(r"checkpoint_(\d+)", name):
            if path.is_dir():
                plan.move(name, f"checkpoints/step_{m.group(1)}"); continue
        if name == "best_model" and path.is_dir():
            plan.move(name, "checkpoints/best")
        elif name in ("env.json", "git_diff.patch"):
            plan.move(name, f"launch/{name}")
        elif m := re.fullmatch(r"env\.resume_(\d+)\.json", name):
            plan.move(name, f"launch/resume_{m.group(1)}/env.json")
        elif m := re.fullmatch(r"git_diff\.resume_(\d+)\.patch", name):
            plan.move(name, f"launch/resume_{m.group(1)}/git_diff.patch")
        elif name in ("run.log", "stats.jsonl", "wandb"):
            plan.move(name, f"logs/{name}")
        elif name in ARTIFACT_KINDS and path.is_dir():
            for sub in sorted(p.name for p in path.iterdir()):
                if (m := _STEP.fullmatch(sub)) and (path / sub).is_dir():
                    plan.move(f"{name}/{sub}", f"artifacts/step_{m.group(1)}/{name}")
                else:
                    plan.keep(f"{name}/{sub}", "unknown file: left in place")
            plan.delete_if_emptied(name)
        elif name == "evaluation" and path.is_dir():
            _plan_legacy_evaluation(root, plan)
        elif name in ("config.yaml", RUN_INFO):
            if name == "config.yaml":
                plan.keep(name, "already in place")
        elif name in ("wandb_id.txt", "checkpoints", "launch", "logs", "artifacts", "evaluations"):
            if name != "wandb_id.txt":
                plan.keep(name, "already in place")
        else:
            plan.keep(name, "unknown file: left in place")

    payload: Dict[str, object] = {"layout_version": 2}
    ids_file = root / "wandb_id.txt"
    if ids_file.exists():
        ids = [l.strip() for l in ids_file.read_text().splitlines() if l.strip()]
        if ids:
            project = os.environ.get("WANDB_PROJECT")
            if project is None:
                plan.warn("WANDB_PROJECT is not set: wandb.project will be null (overwrite cannot delete these runs)")
            runs = [{"id": i, "name": None, "started": None, "resumed_from_step": None, "url": None} for i in ids]
            payload["wandb"] = {"project": project, "entity": None, "runs": runs}
    plan.write(payload, defaults=_default_run_info(root))
    if ids_file.exists():
        plan.delete("wandb_id.txt")


def _plan_legacy_evaluation(root: Path, plan: Plan) -> None:
    """`evaluation/` -> `evaluations/legacy/`; its unknown files stay in `evaluation/`."""
    ev = root / "evaluation"
    for name in sorted(p.name for p in ev.iterdir()):
        rel = f"evaluation/{name}"
        if name == "eval_stats.jsonl":
            plan.move(rel, "evaluations/legacy/logs/stats.jsonl")
        elif name == "config.yaml":
            plan.move(rel, "evaluations/legacy/config.yaml")
        elif name in ARTIFACT_KINDS and (ev / name).is_dir():
            steps = sorted(p.name for p in (ev / name).iterdir() if _STEP.fullmatch(p.name) and p.is_dir())
            for sub in sorted(p.name for p in (ev / name).iterdir()):
                if sub not in steps:
                    plan.keep(f"{rel}/{sub}", "unknown file: left in place")
            for sub in steps:
                target = f"evaluations/legacy/artifacts/{name}" + ("" if len(steps) == 1 else f"/{sub}")
                plan.move(f"{rel}/{sub}", target)
            plan.delete_if_emptied(rel)
        else:
            plan.keep(rel, "unknown file: left in place")
    plan.delete_if_emptied("evaluation")


def _default_run_info(root: Path) -> dict:
    """run.json for a run that had none: name from the folder, group only if its parent folder is a group
    (a sibling run already names it as its group), created from the folder's modification time."""
    group = None
    for sibling in root.parent.glob(f"*/{RUN_INFO}"):
        try:
            if json.loads(sibling.read_text()).get("group") == root.parent.name:
                group = root.parent.name
                break
        except (OSError, ValueError):
            continue
    created = datetime.datetime.fromtimestamp(root.stat().st_mtime).isoformat(timespec="seconds")
    return {"group": group, "name": root.name, "created": created}
