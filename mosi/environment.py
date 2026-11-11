import datetime
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_DIFF_BYTES = 1_000_000
MAX_UNTRACKED = 50
PACKAGES = ("torch", "numpy", "jax", "jaxlib", "wandb", "imageio", "tyro", "pyyaml")


def _git(root: Path, *args: str, env: Optional[Dict[str, str]] = None, timeout: int = 5) -> Optional[str]:
    """stdout of a git command, None on failure. `--no-optional-locks` keeps read-only commands from
    refreshing (rewriting) the repo's real index."""
    try:
        out = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True, text=True,
                             timeout=timeout, env={**os.environ, **(env or {})})
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _script_dir() -> Path:
    """Directory of the running script (`sys.argv[0]`), else the cwd (`-c`, `-m`, REPL, notebooks)."""
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and Path(argv0).is_file():
        return Path(argv0).resolve().parent
    return Path.cwd()


def _diff_with_untracked(root: Path, untracked: List[str]) -> str:
    """`git diff HEAD` including the `untracked` files. They are marked intent-to-add (`git add -N`) in a
    temporary copy of HEAD's index (GIT_INDEX_FILE), so the repo's real index is never touched."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        if _git(root, "read-tree", "HEAD", env=env) is None:
            return _git(root, "diff", "HEAD", timeout=30) or ""
        if untracked:
            _git(root, "add", "-N", "--", *untracked, env=env)
        return _git(root, "diff", "HEAD", env=env, timeout=30) or ""


def _git_info(start: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """`(info, diff)` for the repo containing `start`; `(None, None)` outside a repo or without git."""
    root = _git(start, "rev-parse", "--show-toplevel")
    commit = _git(start, "rev-parse", "HEAD")
    if root is None or commit is None:
        return None, None
    root_path = Path(root.strip())
    status = (_git(root_path, "status", "--porcelain") or "").splitlines()
    untracked: List[str] = [f for f in (_git(root_path, "ls-files", "--others", "--exclude-standard") or "").splitlines() if f]
    diff = _diff_with_untracked(root_path, untracked[:MAX_UNTRACKED])
    diff_bytes = len(diff.encode())
    info: Dict[str, Any] = {
        "root": str(root_path),
        "commit": commit.strip(),
        "branch": (_git(root_path, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "dirty": bool(status),
        "untracked": untracked[:MAX_UNTRACKED],
        "diff_bytes": diff_bytes,
        "diff_truncated": diff_bytes > MAX_DIFF_BYTES,
    }
    if info["diff_truncated"]:
        diff = diff.encode()[:MAX_DIFF_BYTES].decode(errors="ignore")
    return info, diff or None


def _gpu_info() -> Optional[Dict[str, Any]]:
    """GPU names, driver and CUDA version: from torch.cuda when torch is importable and sees a GPU
    (the driver then still comes from nvidia-smi), else from nvidia-smi, else None."""
    names: List[str] = []
    cuda: Optional[str] = None
    source = None
    try:
        import torch
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            cuda, source = torch.version.cuda, "torch"
    except ImportError:
        pass
    query = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    rows = [[c.strip() for c in line.split(",")] for line in (query or "").splitlines() if line.strip()]
    driver = rows[0][1] if rows else None
    if source is None:
        if not rows:
            return None
        names, source = [r[0] for r in rows], "nvidia-smi"
        match = re.search(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]) or "")
        cuda = match.group(1) if match else None
    return {"names": names, "driver": driver, "cuda": cuda, "source": source}


def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _package_versions() -> Dict[str, str]:
    """Versions of the installed packages that matter for reproducing a run (installed, not imported)."""
    versions: Dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    import mosi
    versions["model-sitter"] = mosi.__version__
    return versions


def _pixi_info(root: Path) -> Optional[Dict[str, Any]]:
    """sha256 of `<root>/pixi.lock` and the active `PIXI_ENVIRONMENT_NAME`, None when neither exists."""
    lock = root / "pixi.lock"
    name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if not lock.is_file() and name is None:
        return None
    return {"lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
            "environment": name}


def capture_environment(logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, Any], Optional[str]]:
    """`(env, diff)`: a json-able description of the launch, the git state of the script's repo, the GPU,
    package versions and pixi lock. A part that fails is None and logs a WARNING, it never breaks the run."""
    warn: Callable[[str], None] = (logger or logging.getLogger("mosi")).warning
    env: Dict[str, Any] = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": {"argv": list(sys.argv), "cwd": os.getcwd()},
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "host": {"name": socket.gethostname(), "platform": platform.platform()},
    }
    diff = None
    script_dir = _script_dir()
    try:
        env["git"], diff = _git_info(script_dir)
    except Exception as e:
        env["git"] = None
        warn(f"Provenance: could not capture the git state: {e!r}")
    for key, capture in (("gpu", _gpu_info), ("packages", _package_versions)):
        try:
            env[key] = capture()
        except Exception as e:
            env[key] = None
            warn(f"Provenance: could not capture {key}: {e!r}")
    try:
        root = Path(env["git"]["root"]) if env["git"] else script_dir
        env["pixi"] = _pixi_info(root)
    except Exception as e:
        env["pixi"] = None
        warn(f"Provenance: could not capture the pixi environment: {e!r}")
    return env, diff


def write_environment(launch_dir: Path, keep_previous: bool, logger: Optional[logging.Logger] = None) -> List[Path]:
    """Writes `env.json` (+ `git_diff.patch` when the tree is dirty) into `launch_dir`. With `keep_previous`
    an existing `env.json` is left alone and the new capture goes to `resume_<k>/env.json` / `resume_<k>/git_diff.patch`."""
    env, diff = capture_environment(logger)
    target = launch_dir
    if keep_previous and (launch_dir / "env.json").exists():
        k = 1
        while (launch_dir / f"resume_{k}").exists():
            k += 1
        target = launch_dir / f"resume_{k}"
    target.mkdir(parents=True, exist_ok=True)
    written = []
    env_path = target / "env.json"
    env_path.write_text(json.dumps(env, indent=2), encoding="utf-8")
    written.append(env_path)
    diff_path = target / "git_diff.patch"
    if diff:
        diff_path.write_text(diff, encoding="utf-8")
        written.append(diff_path)
    elif diff_path.exists():
        diff_path.unlink()  # clean tree now: don't leave a stale patch from an earlier session
    return written
