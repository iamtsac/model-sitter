import csv
import numbers
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union, Any

from .config_yaml import dump_config

class ArtifactManager:
    def __init__(self, work_dir: Path, logger: logging.Logger):
        self.work_dir = work_dir
        self.logger = logger
        self._obj: Any = None
        self._extra: Dict[str, Any] = {}

    def save_config(self, config: Any) -> None:
        """Dicts merge into one mapping, any other config (e.g. a dataclass) is kept as the main object.
        With both present config.yaml holds `{config: <object>, extra: {<dict values>}}`."""
        if isinstance(config, dict):
            self._extra.update(config)
        else:
            self._obj = config
        if self._obj is None:
            doc = self._extra
        elif not self._extra:
            doc = self._obj
        else:
            doc = {"config": self._obj, "extra": self._extra}
        with open(self.work_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            f.write(dump_config(doc))

    def _step_dir(self, kind: str, step: int) -> Path:
        """`<run>/<kind>/step_<step>/`, kind being images, videos or analysis."""
        d = self.work_dir / kind / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_csv(self, name: str, data: Dict[str, List], step: int) -> Optional[Path]:
        keys, values = list(data.keys()), list(data.values())
        if not values: return None

        csv_path = self._step_dir("analysis", step) / f"{name}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(list(zip(*values)))
        return csv_path

    def save_media(self, name: str, obj: Any, step: int, **kwargs) -> Optional[Path]:
        """ Mainly for plots and images through matplotlib, seaborn or PIL """
        file_path = self._step_dir("images", step) / name

        if hasattr(obj, 'savefig'): obj.savefig(file_path, **kwargs)
        elif hasattr(obj, 'save'): obj.save(file_path, **kwargs)
        else:
            self.logger.error(f"Object {name} has no .save() or .savefig() method.")
            return None
        return file_path

    def save_video(self, name: str, video: Union[str, Path, np.ndarray], step: int, **imageio_kwargs) -> Optional[Path]:
        target_path = self._step_dir("videos", step) / name
        if target_path.suffix != '.mp4': 
            target_path = target_path.with_suffix('.mp4')

        if isinstance(video, (str, Path)):
            source_path = Path(video)
            if not source_path.exists():
                self.logger.error(f"Video file {source_path} not found.")
                return None
            if source_path.absolute() != target_path.absolute():
                shutil.copy(source_path, target_path)
        else:
            try:
                import imageio
            except ImportError as e:
                raise ImportError("Saving video arrays needs imageio: pip install 'model-sitter[media]'") from e
            if 'fps' not in imageio_kwargs: imageio_kwargs["fps"] = 30
            if 'quality' not in imageio_kwargs: imageio_kwargs["quality"] = 8
            imageio.mimwrite(target_path, video, **imageio_kwargs)

        return target_path

    @staticmethod
    def _sanitize_config(data: Any) -> Any:
        if isinstance(data, dict): return {k: ArtifactManager._sanitize_config(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)): return [ArtifactManager._sanitize_config(v) for v in data]
        if isinstance(data, (numbers.Number, np.ndarray, bool, type(None))): return data
        return str(data)
