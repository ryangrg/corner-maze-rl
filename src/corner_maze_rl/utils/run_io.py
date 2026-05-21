"""Seed management, run-config writing, and run-environment capture.

Ported from legacy ``src/rl/seed_utils.py`` with additions:
  * ``capture_git_sha`` — records the repo commit so a run can be tied to code.
  * ``hash_dataset`` — fingerprints the yoked-data parquets so runs trained on
    a different dataset version are distinguishable.
  * ``save_run_config`` now auto-fills ``git_sha`` and ``dataset_hash`` if
    those fields aren't supplied explicitly.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from stable_baselines3.common.utils import set_random_seed as _sb3_seed
except ImportError:  # pragma: no cover — SB3 is optional
    _sb3_seed = None


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_global_seed(seed: int, using_cuda: bool = False) -> None:
    """Seed all RNG sources: random, numpy, torch (and CUDA if requested).

    Uses stable_baselines3's seeder when available (handles SB3-internal RNGs),
    otherwise falls back to manual seeding of random/numpy/torch.
    """
    if _sb3_seed is not None:
        _sb3_seed(seed, using_cuda=using_cuda)
        return

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if using_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_seed() -> int:
    """Generate a random 6-digit seed in [100_000, 999_999].

    Uses ``os.urandom`` so the result is independent of any prior
    ``set_global_seed`` call.
    """
    return int.from_bytes(os.urandom(4), "big") % 900_000 + 100_000


# ---------------------------------------------------------------------------
# Git / dataset fingerprinting
# ---------------------------------------------------------------------------

def capture_git_sha(repo_dir: str | os.PathLike[str] | None = None) -> str | None:
    """Return the current git HEAD sha, or ``None`` if not in a repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    sha = out.stdout.strip()
    return sha or None


def capture_git_dirty(repo_dir: str | os.PathLike[str] | None = None) -> bool | None:
    """Return True iff the working tree has uncommitted changes; None on error."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return bool(out.stdout.strip())


def hash_dataset(paths: Iterable[str | os.PathLike[str]]) -> str:
    """Hash the byte content of one or more dataset files.

    Returns a short (12-char) sha256 hex digest of the concatenated file bytes,
    in the order given. Raises FileNotFoundError if any path is missing.
    """
    h = hashlib.sha256()
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"hash_dataset: not a file: {path}")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Run-environment capture (deps + hardware)
# ---------------------------------------------------------------------------

def capture_pip_freeze() -> list[str] | None:
    """Return `pip freeze` output as a list of `pkg==ver` strings, or None.

    Captures the full installed-package set for the active interpreter so a
    run can be reproduced bit-for-bit. ~200 lines on a typical env.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return [line for line in out.stdout.splitlines() if line.strip()]


def capture_environment() -> dict[str, Any]:
    """Snapshot the runtime environment for reproducibility.

    Returns a dict with python version, OS/arch, torch + CUDA info if
    available, and `pip freeze`. Safe to call without torch installed.
    """
    env: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform":       platform.platform(),
        "machine":        platform.machine(),
        "processor":      platform.processor() or None,
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
        env["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass
    env["pip_freeze"] = capture_pip_freeze()
    return env


# ---------------------------------------------------------------------------
# Run config persistence
# ---------------------------------------------------------------------------

def save_run_config(
    save_dir: str | os.PathLike[str],
    seed: int,
    *,
    hyperparams: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    dataset_paths: Iterable[str | os.PathLike[str]] | None = None,
    repo_dir: str | os.PathLike[str] | None = None,
    capture_env: bool = True,
    filename: str = "run_config.json",
) -> str:
    """Write a JSON run config to *save_dir*/*filename*.

    Always includes: ``seed``, ``timestamp`` (UTC ISO-8601), ``cwd``,
    ``git_sha`` (or null), ``git_dirty`` (or null).

    Optional: ``hyperparams`` (model + training settings), ``extra`` (catch-all
    for run-specific keys), ``dataset_hash`` (auto-computed from
    ``dataset_paths`` if provided and ``extra`` doesn't already supply one),
    ``environment`` (deps + hardware snapshot; toggle with ``capture_env``).

    The directory is created if it doesn't exist. Returns the path written.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "seed": int(seed),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "git_sha": capture_git_sha(repo_dir),
        "git_dirty": capture_git_dirty(repo_dir),
    }
    if hyperparams is not None:
        config["hyperparams"] = hyperparams
    if dataset_paths is not None and (extra is None or "dataset_hash" not in extra):
        config["dataset_hash"] = hash_dataset(dataset_paths)
    if capture_env:
        config["environment"] = capture_environment()
    if extra is not None:
        config.update(extra)

    out_path = save_path / filename
    with out_path.open("w") as f:
        json.dump(config, f, indent=2, default=str)
    return str(out_path)


def load_run_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a run_config.json back as a dict."""
    with open(path) as f:
        return json.load(f)
