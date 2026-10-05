"""Serialization backends for checkpoints. Only `TorchBackend` needs torch, and it imports it lazily."""
import pickle
from pathlib import Path
from typing import Any, Dict, Protocol, Union, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    suffix: str

    def save(self, obj: Dict[str, Any], path: Path) -> None: ...
    def load(self, path: Path, like: Dict[str, Any]) -> Dict[str, Any]:
        """`like` maps the names the caller wants restored to the live objects (used e.g. to infer a device)."""
        ...


class PickleBackend:
    suffix = "pkl"

    def save(self, obj: Dict[str, Any], path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path, like: Dict[str, Any]) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return pickle.load(f)


class TorchBackend:
    suffix = "pt"

    def __init__(self):
        try:
            import torch
        except ImportError as e:
            raise ImportError("The torch checkpoint backend needs torch: pip install 'model-sitter[torch]'") from e
        self.torch = torch

    def save(self, obj: Dict[str, Any], path: Path) -> None:
        self.torch.save(obj, path)

    def load(self, path: Path, like: Dict[str, Any]) -> Dict[str, Any]:
        # Checkpoints hold optimizer/monitor state, not just tensors, so full unpickling is required.
        return self.torch.load(path, map_location=self._device(like), weights_only=False)

    @staticmethod
    def _device(like: Dict[str, Any]):
        """Device of the first parameter found among the objects being restored, or None."""
        for obj in like.values():
            for o in (obj if isinstance(obj, (list, tuple)) else [obj]):
                params = getattr(o, "parameters", None)
                if callable(params):
                    p = next(iter(params()), None)
                    if p is not None and hasattr(p, "device"):
                        return p.device
        return None


_REGISTRY = {"pickle": PickleBackend, "torch": TorchBackend}
KNOWN_SUFFIXES = tuple(b.suffix for b in _REGISTRY.values())


def resolve_backend(backend: Union[str, Backend] = "auto") -> Backend:
    if not isinstance(backend, str):
        return backend
    if backend == "auto":
        try:
            return TorchBackend()
        except ImportError:
            return PickleBackend()
    if backend not in _REGISTRY:
        raise ValueError(f"Unknown checkpoint backend {backend!r}, use one of {['auto', *_REGISTRY]} or a Backend instance")
    return _REGISTRY[backend]()


def backend_for_suffix(suffix: str) -> Backend:
    return {b.suffix: b for b in _REGISTRY.values()}[suffix]()
