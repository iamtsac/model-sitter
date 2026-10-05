from .sitter import Sitter, SitterConfig
from .groups import load_group_stats, list_group_runs
from .config_yaml import dump_config, load_config
from .checkpointing import Backend, PickleBackend, TorchBackend

# Deprecated aliases from before the rename; kept for one release.
ModelLogger = Sitter
LoggerConfig = SitterConfig

__version__ = "0.1.0"
__all__ = [
    "Sitter", "SitterConfig", "ModelLogger", "LoggerConfig", "load_group_stats", "list_group_runs", "dump_config", "load_config",
    "Backend", "PickleBackend", "TorchBackend",
]
