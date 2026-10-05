import json
import os
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .backends import Backend, PickleBackend, TorchBackend, KNOWN_SUFFIXES, backend_for_suffix, resolve_backend

__all__ = ["CheckpointManager", "Backend", "PickleBackend", "TorchBackend"]


class CheckpointManager:
    """Saves/loads checkpoints of anything exposing `state_dict()` / `load_state_dict()` (or lists of them)."""

    def __init__(self, fpath: Path, logger: logging.Logger, backend: Union[str, Backend] = "auto",
                 keep_last: Optional[int] = None):
        if keep_last is not None and keep_last < 1:
            raise ValueError("keep_last must be >= 1 (or None to keep every checkpoint)")
        self.fpath = Path(fpath)
        self.logger = logger
        self.backend = resolve_backend(backend)
        self.keep_last = keep_last

    def save(
        self,
        step: int,
        monitor_state: Optional[dict] = None,
        is_best: bool = False,
        **modules,
    ) -> Path:
        _path = self.fpath / f"checkpoint_{step}"
        _path.mkdir(parents=True, exist_ok=True)

        cp = dict(step=step)
        for k, v in modules.items():
            if v is not None: cp.update(self._create_dict(v, k))

        if monitor_state:
            cp['monitor_state'] = monitor_state

        ckpt_file = f"checkpoint.{self.backend.suffix}"
        self._atomic(lambda tmp: self.backend.save(cp, tmp), _path / ckpt_file)
        self.logger.debug(f"Checkpoint saved: checkpoint_{step}")

        if is_best:
            best_dir = self.fpath / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            for old in best_dir.glob("checkpoint.*"):
                if old.name != ckpt_file: old.unlink()
            self._atomic(lambda tmp: shutil.copyfile(_path / ckpt_file, tmp), best_dir / ckpt_file)

            if monitor_state and 'best_metrics' in monitor_state:
                with open(best_dir / "best_metrics.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, **monitor_state['best_metrics']}) + "\n")

        self._prune()
        return _path

    @staticmethod
    def _atomic(write, target: Path) -> None:
        """`write(tmp)` then rename, so a crash mid-write never leaves a truncated checkpoint behind."""
        tmp = target.with_name(target.name + ".tmp")
        try:
            write(tmp)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    def _prune(self) -> None:
        """Deletes the oldest `checkpoint_N` folders beyond `keep_last`; `best_model/` is never touched."""
        if self.keep_last is None:
            return
        for _, old in self._saved_steps()[:-self.keep_last]:
            shutil.rmtree(old)

    def _saved_steps(self):
        """`[(step, folder)]` of the `checkpoint_<N>` folders, oldest first."""
        return sorted((int(p.name.split("_")[-1]), p) for p in self.fpath.glob("checkpoint_*")
                      if p.is_dir() and p.name.split("_")[-1].isdigit())

    def load(self, step: Union[int, str], **modules) -> Tuple[dict, Path]:
        if isinstance(step, str):
            if step.lower() == "latest":
                steps = self._saved_steps()
                if not steps:
                    raise FileNotFoundError(f"No checkpoint_<N> folders in {self.fpath}")
                ckpt_folder = steps[-1][1].absolute()
                self.logger.info(f"Loading the latest checkpoint ({ckpt_folder.name})...")
            elif step.lower() == "best":
                ckpt_folder = self.fpath.absolute() / "best_model"
                if not ckpt_folder.exists():
                    self.logger.error(f"No best checkpoint found at {ckpt_folder}")
                    raise FileNotFoundError("Best checkpoint missing.")
                self.logger.info("Loading the absolute best model weights...")
            else:
                potential_path = Path(step)
                if potential_path.exists() and potential_path.is_dir():
                    ckpt_folder = potential_path
                    self.logger.info(f"Loading checkpoint from explicit path {ckpt_folder}")
                else:
                    ckpt_folder = self.fpath.absolute() / f"checkpoint_{step}"
        else:
            ckpt_folder = self.fpath.absolute() / f"checkpoint_{step}"

        checkpoint = self._restore(ckpt_folder, modules)
        return checkpoint, ckpt_folder

    def _restore(self, folder: Path, modules: Dict[str, Any]) -> dict:
        if not folder.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {folder}")
        file = next((folder / f"checkpoint.{s}" for s in KNOWN_SUFFIXES if (folder / f"checkpoint.{s}").exists()), None)
        if file is None:
            raise FileNotFoundError(f"No checkpoint.{{{','.join(KNOWN_SUFFIXES)}}} in {folder}")
        backend = self.backend if file.suffix[1:] == self.backend.suffix else backend_for_suffix(file.suffix[1:])

        live = {k: v for k, v in modules.items() if v is not None}
        cp = backend.load(file, live)
        for key, obj in live.items():
            expected = ([f"{key}_{i}" for i in range(1, len(obj) + 1)]
                        if isinstance(obj, (list, tuple)) and obj and all(self._is_stateful(o) for o in obj) else [key])
            missing = [k for k in expected if k not in cp]
            if missing:
                raise KeyError(f"{missing} not in {file} (it holds {sorted(cp)}); was it saved with different names?")

        out = {k: v for k, v in cp.items() if k not in live and not self._is_state_of(k, live)}
        for key, obj in live.items():
            if isinstance(obj, (list, tuple)) and obj and all(self._is_stateful(o) for o in obj):
                for i, o in enumerate(obj, start=1):
                    o.load_state_dict(cp[f"{key}_{i}"])
                out[key] = obj
            elif self._is_stateful(obj):
                obj.load_state_dict(cp[key])
                out[key] = obj
            elif isinstance(obj, dict) and any(self._is_stateful(v) for v in obj.values()):
                saved = cp[key]
                for k, v in obj.items():
                    if self._is_stateful(v): v.load_state_dict(saved[k])
                out[key] = {**saved, **{k: v for k, v in obj.items() if self._is_stateful(v)}}
            else:  # plain data: the saved value replaces the template
                out[key] = cp[key]
        out["step"] = cp.get("step")
        return out

    @staticmethod
    def _is_state_of(key: str, live: Dict[str, Any]) -> bool:
        return any(isinstance(o, (list, tuple)) and key.startswith(f"{k}_") for k, o in live.items())

    @staticmethod
    def _is_stateful(o: object) -> bool:
        return hasattr(o, "state_dict") and hasattr(o, "load_state_dict")

    @staticmethod
    def _create_dict(obj: object, key: str) -> Dict[str, Any]:
        """Stateful objects (and lists of them) are stored as their state_dict, anything else
        (dicts, numpy/jax pytrees, scalars, rng keys...) is stored as is and comes back from `load`."""
        is_stateful = CheckpointManager._is_stateful
        if is_stateful(obj):
            return {key: obj.state_dict()}
        if isinstance(obj, (list, tuple)) and obj and all(is_stateful(o) for o in obj):
            return {f'{key}_{i}': o.state_dict() for i, o in enumerate(obj, start=1)}
        if isinstance(obj, dict) and any(is_stateful(v) for v in obj.values()):
            # e.g. {"gen": gen, "disc": disc, "step_count": 3}: states for the stateful ones, the rest as is
            return {key: {k: (v.state_dict() if is_stateful(v) else v) for k, v in obj.items()}}
        return {key: obj}
