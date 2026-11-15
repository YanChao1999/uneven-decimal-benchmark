"""Resolve paths relative to the repository root (not the caller's cwd)."""

from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Walk parents until we find ``pyproject.toml`` or ``.git``."""
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return cur


def resolve_repo_path(path: str | Path, *, root: Path | None = None) -> Path:
    """Absolute paths pass through; relative paths are rooted at the repo."""
    p = Path(path)
    if p.is_absolute():
        return p
    base = root or repo_root()
    return (base / p).resolve()
