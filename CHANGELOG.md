# Changelog

## Unreleased
- Renamed `ModelLogger` to `Sitter` and `LoggerConfig` to `SitterConfig` (both in `mosi.sitter`). The old names remain as deprecated aliases. On-disk run format is unchanged.

## 0.1.0
- Provenance: `env.json` (command line, python, host, git commit / branch / dirty / untracked) and `git_diff.patch`; resumes add `env.resume_<k>.json`.
- Group membership lives in each run's `run.json` instead of a shared `group.json` (no race between parallel runs).
- Standalone package extracted from butane's logger (core needs only numpy + pyyaml).
- Run groups: `group` + `name` give `<fpath>/<group>/<name>` on disk and the W&B group / run name.
- Framework-free checkpoints: anything with `state_dict()` / `load_state_dict()` or plain data; torch and pickle backends, `keep_last`, atomic writes, `"latest"` / `"best"`.
- Artifacts split into `images/`, `videos/`, `analysis/` per step.
- Monitor with `should_stop`, `log_every`; `LoggerConfig` dataclass (tyro friendly); context manager; `run.log`.
