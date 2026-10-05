"""JAX/optax: params and optimizer state are plain pytrees, stored as is with the pickle backend."""
import jax, jax.numpy as jnp, optax
from mosi import Sitter

x = jnp.linspace(-1, 1, 64)[:, None]; y = 3 * x + 1
params = {"w": jnp.zeros((1, 1)), "b": jnp.zeros(1)}
tx = optax.adam(0.1); opt_state = tx.init(params)
loss_fn = lambda p: jnp.mean((x @ p["w"] + p["b"] - y) ** 2)

with Sitter("runs/jax", checkpoint_backend="pickle") as sitter:
    for _ in range(30):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        sitter.add_stats(loss=loss)                                  # jax scalars are fine
        sitter.checkpoint(params=params, opt_state=opt_state)
        sitter.step()

with Sitter("runs/jax", eval_mode=True, checkpoint_backend="pickle") as sitter:
    out = sitter.load_checkpoint("latest")                           # plain values come back in the dict
    print({k: v.ravel() for k, v in out["params"].items()})
