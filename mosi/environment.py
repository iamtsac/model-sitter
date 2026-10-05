"""Run provenance: how the script was launched and which code state it ran on. Never raises."""
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAX_DIFF_BYTES = 1_000_000
MAX_UNTRACKED = 50


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _script_dir() -> Path:
    """Directory of the running script (`sys.argv[0]`), else the cwd (`-c`, `-m`, REPL, notebooks)."""
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and Path(argv0).is_file():
        return Path(argv0).resolve().parent
    return Path.cwd()


def _git_info(start: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """`(info, diff)` for the repo containing `start`; `(None, None)` outside a repo or without git."""
    root = _git(start, "rev-parse", "--show-toplevel")
    commit = _git(start, "rev-parse", "HEAD")
    if root is None or commit is None:
        return None, None
    root_path = Path(root.strip())
    status = (_git(root_path, "status", "--porcelain") or "").splitlines()
    untracked: List[str] = [l[3:] for l in status if l.startswith("??")]
    diff = _git(root_path, "diff", "HEAD") or ""
    info: Dict[str, Any] = {
        "root": str(root_path),
        "commit": commit.strip(),
        "branch": (_git(root_path, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "dirty": bool(status),
        "untracked": untracked[:MAX_UNTRACKED],
    }
    if len(diff.encode()) > MAX_DIFF_BYTES:
        info["diff_truncated"] = True
        diff = diff.encode()[:MAX_DIFF_BYTES].decode(errors="ignore")
    return info, diff or None


def capture_environment() -> Tuple[Dict[str, Any], Optional[str]]:
    """`(env, diff)`: a json-able description of the launch and the git state of the script's repo."""
    env: Dict[str, Any] = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": {"argv": list(sys.argv), "cwd": os.getcwd()},
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "host": {"name": socket.gethostname(), "platform": platform.platform()},
    }
    diff = None
    try:
        env["git"], diff = _git_info(_script_dir())
    except Exception:  # provenance must never break a run
        env["git"] = None
    return env, diff


def write_environment(run_dir: Path, keep_previous: bool) -> List[Path]:
    """Writes `env.json` (+ `git_diff.patch` when the tree is dirty). With `keep_previous` an existing
    `env.json` is left alone and the new capture goes to `env.resume_<k>.json` / `git_diff.resume_<k>.patch`."""
    env, diff = capture_environment()
    tag = ""
    if keep_previous and (run_dir / "env.json").exists():
        k = 1
        while (run_dir / f"env.resume_{k}.json").exists():
            k += 1
        tag = f".resume_{k}"
    written = []
    env_path = run_dir / f"env{tag}.json"
    env_path.write_text(json.dumps(env, indent=2), encoding="utf-8")
    written.append(env_path)
    diff_path = run_dir / f"git_diff{tag}.patch"
    if diff:
        diff_path.write_text(diff, encoding="utf-8")
        written.append(diff_path)
    elif diff_path.exists():
        diff_path.unlink()  # clean tree now: don't leave a stale patch from an earlier session
    return written
