import sys
import json
import math
import logging
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Callable

class ExperimentEnvironment:
    """ Helper class that manages the local experiment directories and paths """
    def __init__(self, fpath: str, overwrite: bool, resume: bool, eval_mode: bool, logger: logging.Logger,
                 confirm: Optional[Callable[[str], bool]] = None):
        self.fpath = Path(fpath)
        self.overwrite = overwrite and not eval_mode
        self.resume = resume and not eval_mode
        self.eval_mode = eval_mode
        self.logger = logger
        self.confirm = confirm or self._ask_user

        self.work_dir = self.fpath / "evaluation" if self.eval_mode else self.fpath

        self._setup_directory()

    def _setup_directory(self) -> None:
        if self.eval_mode and not self.fpath.exists():
            raise FileNotFoundError(f"eval_mode needs an existing run, but {self.fpath} does not exist")

        if self.overwrite and self.fpath.exists() and not (self.resume or self.eval_mode):
            msg = (f"Overwrite is set to True for '{self.fpath}'. This will completely DESTROY existing local stats "
                   "and delete this run's WandB lineage from the cloud (other runs in its group are untouched).")
            if not self.confirm(msg):
                print("Aborted by user.")
                sys.exit(0)

        if self.fpath.exists():
            if not (self.resume or self.eval_mode or self.overwrite):
                mtime = self.fpath.stat().st_mtime
                ts = datetime.datetime.fromtimestamp(mtime).strftime('%Y_%m_%d__%H_%M_%S')
                archived_path = self.fpath.with_name(f"{self.fpath.name}_{ts}")
                n = 1
                while archived_path.exists():  # two reruns inside one second
                    archived_path = self.fpath.with_name(f"{self.fpath.name}_{ts}_{n}")
                    n += 1
                self.fpath.rename(archived_path)
                self.logger.info(f"Archived previous run to: {archived_path}")

        self.fpath.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ask_user(msg: str) -> bool:
        print(f"\n⚠️ WARNING: {msg}")
        return input("Proceed? [y/N]: ").strip().lower() == 'y'

class HistoryManager:

    def __init__(self, work_dir: Path, overwrite: bool, eval_mode: bool, logger: logging.Logger):
        self.work_dir = work_dir
        self.logger = logger

        filename = "eval_stats.jsonl" if eval_mode else "stats.jsonl"
        self.stats_file = self.work_dir / filename

        self._stage_buffer: Dict[str, Any] = {}

        if overwrite and self.stats_file.exists():
            self.stats_file.unlink()

        best_metrics_file = self.work_dir / "best_model/best_metrics.jsonl"
        if overwrite and best_metrics_file.exists():
            best_metrics_file.unlink()

    def stage_metrics(self, stats: dict) -> Dict[str, Any]:
        flat_stats = {k: self._to_scalar(k, v) for k, v in self._flatten_dict(stats).items()}
        self._stage_buffer.update(flat_stats)
        return flat_stats

    def flush_buffer_to_jsonl(self, step: int) -> Dict[str, Any]:
        entry = {"step": step}
        for k, v in self._stage_buffer.items():
            entry[k] = v

        with open(self.stats_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self._stage_buffer.clear()
        return entry

    def recover_state(self, safe_step: int, overwrite: bool) -> None:
        if overwrite:
            return []

        valid_rows = []
        if self.stats_file.exists():
            valid_lines = []
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    row = json.loads(line)
                    if row.get("step", 0) <= safe_step:
                        valid_lines.append(line)
                        valid_rows.append(row)

            temp_file = self.stats_file.with_suffix('.jsonl.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.writelines(valid_lines)
            temp_file.replace(self.stats_file)

        self.logger.info(f"History safely truncated to step {safe_step} on disk.")
        return valid_rows

    @staticmethod
    def _to_scalar(key: str, v: Any) -> Any:
        """Arrays/tensors/jax values holding one element become python numbers, bigger ones are refused."""
        if hasattr(v, "item"):
            n = v.numel() if hasattr(v, "numel") else getattr(v, "size", 1)
            if n != 1:
                raise ValueError(f"add_stats({key}=...) needs a scalar, got shape {tuple(getattr(v, 'shape', ()))}; "
                                 "reduce it first (e.g. .mean()) or log it as an analysis table")
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            return None  # NaN/inf are not valid JSON
        return v

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(HistoryManager._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
