"""Checkpointing for non-torch frameworks, through plain pytrees / pickleable estimators."""
import numpy as np
import pytest
from mosi import ModelLogger


def test_jax_training_state_roundtrip(tmp_path):
    jax = pytest.importorskip("jax")
    optax = pytest.importorskip("optax")
    import jax.numpy as jnp

    x = jnp.linspace(-1, 1, 32)[:, None]; y = 3 * x + 1
    params = {"w": jnp.zeros((1, 1)), "b": jnp.zeros(1)}
    tx = optax.adam(0.1); opt_state = tx.init(params)
    loss_fn = lambda p: jnp.mean((x @ p["w"] + p["b"] - y) ** 2)

    log = ModelLogger(tmp_path / "r", checkpoint_backend="pickle")
    for _ in range(30):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        log.add_stats(loss=loss)  # a jax scalar straight into the stats
        log.checkpoint(params=params, opt_state=opt_state, rng=jax.random.key_data(jax.random.key(0)))
        log.step()

    out = ModelLogger(tmp_path / "r", eval_mode=True, checkpoint_backend="pickle").load_checkpoint(30)
    assert np.allclose(out["params"]["w"], params["w"]) and np.allclose(out["params"]["b"], params["b"])
    assert jax.tree_util.tree_structure(out["opt_state"]) == jax.tree_util.tree_structure(opt_state)
    assert np.isfinite(float(loss))
    assert (tmp_path / "r" / "stats.jsonl").read_text().count("\n") == 30


def test_sklearn_estimator_roundtrip(tmp_path):
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3)); y = X @ [1.0, -2.0, 0.5]
    model = sk.SGDRegressor(random_state=0)
    log = ModelLogger(tmp_path / "r", checkpoint_backend="pickle")
    for _ in range(5):
        model.partial_fit(X, y)
        log.add_stats(score=model.score(X, y)); log.checkpoint(model=model); log.step()
    out = ModelLogger(tmp_path / "r", eval_mode=True, checkpoint_backend="pickle").load_checkpoint(5)
    assert np.allclose(out["model"].coef_, model.coef_)
