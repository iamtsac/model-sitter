import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..layout import RUN_INFO


class MigrationError(RuntimeError):
    """A run cannot be migrated as planned (e.g. a target path already exists)."""


@dataclass
class Action:
    kind: str                      # move | write | delete | keep
    src: str                       # path relative to the run folder
    dst: str = ""                  # move: target; write: description; keep: reason
    payload: Optional[dict] = None  # write: entries merged into run.json
    defaults: Optional[dict] = None  # write: entries used only when the file did not exist


@dataclass
class Plan:
    """What one run needs, in execution order. Planning only reads the disk; `apply` performs the actions."""
    run_dir: Path
    actions: List[Action] = field(default_factory=list)
    title: str = ""
    _moved: set = field(default_factory=set)
    _targets: set = field(default_factory=set)

    def move(self, src: str, dst: str) -> None:
        if not (self.run_dir / src).exists():
            raise MigrationError(f"{src} vanished while planning")
        if (self.run_dir / dst).exists() or dst in self._targets:
            raise MigrationError(f"cannot move {src} -> {dst}: the target already exists")
        self._moved.add(src.rstrip("/"))
        self._targets.add(dst)
        self.actions.append(Action("move", _slash(self.run_dir, src), _slash(self.run_dir, dst, src_is=src)))

    def delete(self, path: str) -> None:
        self._moved.add(path.rstrip("/"))
        self.actions.append(Action("delete", _slash(self.run_dir, path)))

    def delete_if_emptied(self, folder: str) -> None:
        """Plans the removal of `folder` when every entry in it is moved away by this plan."""
        kids = [f"{folder}/{p.name}" for p in (self.run_dir / folder).iterdir()]
        if all(k in self._moved for k in kids):
            self.delete(folder)

    def write(self, payload: dict, defaults: Optional[dict] = None) -> None:
        shown = ", ".join(f"{k}={v}" for k, v in payload.items())
        self.actions.append(Action("write", RUN_INFO, shown, payload=payload, defaults=defaults))

    def warn(self, message: str) -> None:
        self.actions.append(Action("warn", "warning", message))

    def keep(self, path: str, reason: str) -> None:
        self.actions.append(Action("keep", _slash(self.run_dir, path), reason))

    def apply(self) -> None:
        for a in self.actions:
            if a.kind == "move":
                dst = self.run_dir / a.dst
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(self.run_dir / a.src), str(dst))
            elif a.kind == "delete":
                p = self.run_dir / a.src
                if p.is_dir():
                    p.rmdir()
                else:
                    p.unlink()
            elif a.kind == "write":
                path = self.run_dir / RUN_INFO
                data = json.loads(path.read_text()) if path.exists() else dict(a.defaults or {})
                data.update(a.payload)
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, indent=2))
                tmp.replace(path)


def _slash(run_dir: Path, rel: str, src_is: Optional[str] = None) -> str:
    """Folders print with a trailing slash. For a move target the kind follows its source."""
    probe = src_is if src_is is not None else rel
    return rel.rstrip("/") + ("/" if (run_dir / probe).is_dir() else "")
