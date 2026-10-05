"""PyTorch: model + optimizer + scheduler + EMA in one call, then resume after an interruption."""
import sys
import torch
from mosi import ModelLogger

resume = "--resume" in sys.argv
model = torch.nn.Linear(4, 1)
opt = torch.optim.Adam(model.parameters(), 1e-2)
sched = torch.optim.lr_scheduler.StepLR(opt, 10)
ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99))

log = ModelLogger("runs/torch", resume=resume, checkpoint_backend="torch", keep_last=2)
start = 0
if resume:
    # live objects are restored in place; the returned dict holds the step and anything non-stateful
    start = log.load_checkpoint("latest", model=model, optimizer=opt, lr_scheduler=sched, ema=ema)["step"]

x = torch.randn(256, 4); y = x.sum(1, keepdim=True)
for step in range(start, start + 20):
    loss = torch.nn.functional.mse_loss(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step(); ema.update_parameters(model)
    log.add_stats(loss=loss, lr=sched.get_last_lr()[0])   # tensors are fine, they become floats
    log.checkpoint(model=model, optimizer=opt, lr_scheduler=sched, ema=ema)
    log.step()
log.finish()
