import logging
import numbers
from typing import Optional, List, Dict, Union, Any


class _ModelMonitor:

    def __init__(
        self,
        logger: logging.Logger,
        increase_keys: Optional[Union[List[str], Dict[str, str]]] = None,
        decrease_keys: Optional[Union[List[str], Dict[str, str]]] = None,
        tolerance: float = 0.1,
        patience: int = 1,
        log_every: int = 1,
    ):
        self.logger = logger
        self._increase_keys = self._prepare_keys(increase_keys)
        self._decrease_keys = self._prepare_keys(decrease_keys)

        # Performance Monitor
        self._tolerance = tolerance
        self._patience = patience
        self._wait = 0
        self._log_every = max(1, log_every)
        self._n_best = 0
        self.should_stop = False

        # State
        self.best_step = -1
        self.best_metrics = {}


    def __call__(self, step: int, metrics: dict) -> bool:
        metrics = self._prepare_metrics(metrics)
        present_inc = [k for k in self._increase_keys if k in metrics]
        present_dec = [k for k in self._decrease_keys if k in metrics]

        if not len(present_inc) and not len(present_dec):
            return False

        if self.best_step == -1:
            self._update_state(step, metrics, present_inc, present_dec)
            self._log_new_best(step)
            return True

        total_relative_improvement = 0.0
        eps = 1e-08

        for k in present_inc:
            best_val = self.best_metrics.get(k, metrics[k])
            total_relative_improvement += (metrics[k] - best_val) / (abs(best_val) + eps)

        for k in present_dec:
            best_val = self.best_metrics.get(k, metrics[k])
            total_relative_improvement += (best_val - metrics[k]) / (abs(best_val) + eps)

        if total_relative_improvement > 0.0:
            self._update_state(step, metrics, present_inc, present_dec)
            self._wait = 0  # Reset early stopping clock
            self.should_stop = False

            self._log_new_best(step, total_relative_improvement)
            return True
        else:
            self._wait += 1

            # Check for catastrophic degradation using `tolerance`
            if total_relative_improvement < -self._tolerance:
                self.logger.warning(
                    f"⚠️ [Monitor] Catastrophic degradation at step {step}! "
                    f"Score dropped by {abs(total_relative_improvement)*100:.2f}% "
                    f"(Tolerance: {self._tolerance*100:.2f}%)"
                )
            else:
                self.logger.debug(f"[Monitor] Step {step}: No improvement. (Wait: {self._wait}/{self._patience})")

            if self._patience > 0 and self._wait >= self._patience:
                if not self.should_stop:
                    self.logger.warning(f"🛑 [Monitor] Early stopping triggered at step {step}!")
                self.should_stop = True

            return False

    def state(self) -> dict:
        return {
            'best_step': self.best_step,
            'best_metrics': self.best_metrics,
            'wait': self._wait
        }

    def load_state(self, state: dict) -> None:
        self.best_step = state.get('best_step', -1)
        self.best_metrics = state.get('best_metrics', {})
        self._wait = state.get('wait', 0)
        self.should_stop = self._patience > 0 and self._wait >= self._patience

        if self.best_step != -1:
            self.logger.info(
                f"Monitor state restored: Best Step = {self.best_step}, "
                f"Wait = {self._wait}/{'Infinite' if self._patience == -1 else self._patience}"
            )

    def _update_state(self, step: int, metrics: dict, inc: list, dec: list):
        self.best_step = step
        self.best_metrics = {k: metrics[k] for k in inc + dec}

    def _log_new_best(self, step: int, improvement: Optional[float] = None):
        self._n_best += 1
        if (self._n_best - 1) % self._log_every:
            return
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in self.best_metrics.items()])
        gain = f", +{improvement*100:.2f}%" if improvement is not None else ""
        self.logger.info(f"🏆 Step {step}: New Best Model! ({metric_str}{gain})")

    @staticmethod
    def _flatten_key_paths(obj: Any, prefix: str = '', separator: str = '/') -> List[str]:
        paths = []
        if isinstance(obj, str):
            paths.append(f"{prefix}{separator}{obj}" if prefix else obj)
        elif isinstance(obj, list):
            for item in obj:
                paths.extend(_ModelMonitor._flatten_key_paths(item, prefix, separator))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}{separator}{k}" if prefix else k
                if v is None:
                    paths.append(new_prefix)
                else:
                    paths.extend(_ModelMonitor._flatten_key_paths(v, new_prefix, separator))
        return paths

    @staticmethod
    def _prepare_keys(keys: Union[List[str], Dict, None]) -> List[str]:
        if keys is None:
            return []

        if isinstance(keys, list) and all(isinstance(x, str) for x in keys):
            return keys

        return _ModelMonitor._flatten_key_paths(keys)

    @staticmethod
    def _prepare_metrics(metrics: dict, prefix: str = '', separator: str = '/') -> Dict[str, float]:
        flattened = {}
        for k, v in metrics.items():
            new_key = f"{prefix}{separator}{k}" if prefix else k

            if isinstance(v, dict):
                flattened.update(_ModelMonitor._prepare_metrics(v, new_key, separator))
            elif isinstance(v, numbers.Number):
                flattened[new_key] = v
        return flattened
