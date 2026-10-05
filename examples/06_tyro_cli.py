"""SitterConfig is a plain dataclass, so tyro (or any dataclass CLI) can nest it in your own config.
   python 06_tyro_cli.py --lr 0.01 --logger.group seeds_run --logger.name test_s0 --logger.keep-last 3"""
from dataclasses import dataclass, field
import tyro
from mosi import SitterConfig

@dataclass
class Config:
    lr: float = 1e-3
    steps: int = 10
    logger: SitterConfig = field(default_factory=lambda: SitterConfig(fpath="runs", checkpoint_backend="pickle"))

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    with cfg.logger.build() as sitter:
        sitter.add_config(cfg)                  # lossless YAML of the whole config, nested dataclass included
        for _ in range(cfg.steps):
            sitter.add_stats(loss=cfg.lr)
            sitter.step()
