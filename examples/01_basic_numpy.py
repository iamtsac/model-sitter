"""Smallest useful loop: stats, config and checkpoints, no torch and no wandb."""
import numpy as np
from mosi import Sitter

rng = np.random.default_rng(0)
x = rng.normal(size=(256, 1)); y = 3 * x + 1
w, b = np.zeros((1, 1)), np.zeros(1)

class Params:                       # anything with state_dict / load_state_dict is checkpointed as its state
    def state_dict(self): return {"w": w, "b": b}
    def load_state_dict(self, d): w[...], b[...] = d["w"], d["b"]

with Sitter("runs/basic", checkpoint_backend="pickle", keep_last=3) as sitter:
    sitter.add_config(lr=0.1, steps=50)
    params = Params()
    for step in range(50):
        err = x @ w + b - y
        w -= 0.1 * (x.T @ err) / len(x); b -= 0.1 * err.mean(0)
        sitter.add_stats(loss=float((err ** 2).mean()))
        sitter.checkpoint(params=params)
        sitter.step()

# later: restore the live object in place
with Sitter("runs/basic", eval_mode=True, checkpoint_backend="pickle") as sitter:
    sitter.load_checkpoint("latest", params=params)
    print("restored w:", w.ravel())
