"""Images, plots, videos and tables land in images/, videos/ and analysis/ (one step_<N> folder each)."""
import numpy as np
from PIL import Image
from mosi import Sitter

with Sitter("runs/artifacts", checkpoint_backend="pickle", log_level="WARNING") as sitter:
    for step in range(1, 21):
        sitter.add_stats(loss=1 / step)
        if step % 10 == 0:
            img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8"))
            sitter.add_image("sample.png", img)                          # anything with .save(path); plots: .savefig
            sitter.add_analysis("weights", {"idx": [0, 1, 2], "val": [0.1, 0.2, 0.3]})   # CSV (+ W&B table)
            # sitter.add_video("rollout", frames_uint8_THWC, fps=15)     # needs model-sitter[media]
        sitter.step()
