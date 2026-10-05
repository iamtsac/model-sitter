# Changelog

## 0.1.0
- Standalone package extracted from butane's logger (core needs only numpy + pyyaml).
- Run groups: `group` + `name` give `<fpath>/<group>/<name>` on disk and the W&B group / run name.
- Framework-free checkpoints: anything with `state_dict()` / `load_state_dict()` or plain data; torch and pickle backends, `keep_last`, atomic writes, `"latest"` / `"best"`.
- Artifacts split into `images/`, `videos/`, `analysis/` per step.
- Monitor with `should_stop`, `log_every`; `LoggerConfig` dataclass (tyro friendly); context manager; `run.log`.
