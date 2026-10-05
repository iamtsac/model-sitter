"""Several seeds of one experiment: runs/seeds_run/test_s0, test_s1, ... and one W&B group.
Add --wandb to sync (needs `pip install model-sitter[wandb]` and WANDB_PROJECT or project=...)."""
import sys
import numpy as np
from mosi import Sitter, load_group_stats

use_wandb = "--wandb" in sys.argv
for seed in range(3):
    with Sitter("runs", group="seeds_run", name=f"test_s{seed}", job_type="train", tags=["demo"],
                     use_wandb=use_wandb, checkpoint_backend="pickle", log_level="WARNING",
                     confirm=lambda msg: True) as sitter:
        sitter.add_config(seed=seed, lr=0.1)           # the seed is just config, so you can filter on it in W&B
        rng = np.random.default_rng(seed)
        for step in range(20):
            sitter.add_stats(loss=1 / (step + 1) + 0.01 * rng.normal())
            sitter.step()

stats = load_group_stats("runs/seeds_run")          # {"test_s0": [rows], ...}
final = [rows[-1]["loss"] for rows in stats.values()]
print(f"final loss over seeds: {np.mean(final):.4f} +- {np.std(final):.4f}")
