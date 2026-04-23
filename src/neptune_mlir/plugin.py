"""Helpers for locating the HTile dialect plugin shared library."""

import os
from pathlib import Path

DEFAULT_PLUGIN_ENV = "NEPTUNE_MLIR_PLUGIN"
PLUGIN_PATTERNS = ("libHTileDialectPlugin.*",)


def _find_in_dir(directory: Path) -> Path | None:
    for pattern in PLUGIN_PATTERNS:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _candidate_dirs() -> list[Path]:
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parents[1] if len(pkg_dir.parents) >= 2 else pkg_dir
    return [
        pkg_dir / "_native",
        repo_root / "build",
        Path.cwd() / "build",
    ]


def find_plugin_path(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Resolve the HTile dialect plugin path.

    Resolution order:
    1. Explicit path argument.
    2. NEPTUNE_MLIR_PLUGIN environment variable.
    3. Installed package directory: neptune_mlir/_native.
    4. Common development build dirs: <repo>/build and <cwd>/build.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    env_path = os.environ.get(DEFAULT_PLUGIN_ENV)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    for directory in _candidate_dirs():
        if not directory.is_dir():
            continue
        if (match := _find_in_dir(directory)) is not None:
            return match.resolve()

    return None
