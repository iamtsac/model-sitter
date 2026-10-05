"""The monitor tracks the best model (best_model/) and tells your loop when to stop."""
from mosi import ModelLogger

class Model:
    def __init__(self): self.v = 0
    def state_dict(self): return {"v": self.v}
    def load_state_dict(self, d): self.v = d["v"]

val_losses = [1.0, 0.7, 0.5, 0.55, 0.6, 0.65, 0.7, 0.3]   # stops before the late improvement
model = Model()
with ModelLogger("runs/early", checkpoint_backend="pickle", log_level="WARNING") as log:
    log.enable_monitor(decrease_keys=["val/loss"], patience=3, log_every=5)
    for step, val in enumerate(val_losses, start=1):
        model.v = step
        log.add_stats(val={"loss": val})
        log.checkpoint(model=model)        # metrics default to the stats staged for this step
        log.step()
        if log.should_stop:
            print(f"early stop at step {step}")
            break

with ModelLogger("runs/early", eval_mode=True, checkpoint_backend="pickle") as log:
    log.load_checkpoint("best", model=model)
    print("best model came from step", model.v)
