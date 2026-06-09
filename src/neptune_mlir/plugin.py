"""Helpers for locating Neptune MLIR native plugin shared libraries."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

NATIVE_DIR_ENV = "NEPTUNE_MLIR_NATIVE_DIR"


@dataclass(frozen=True)
class NeptunePlugins:
    loop_transform: Path
    ta_dialect: Path
    htile_dialect: Path

    def dialect_plugin_args(self) -> list[str]:
        return [
            f"--load-dialect-plugin={self.loop_transform}",
            f"--load-dialect-plugin={self.ta_dialect}",
            f"--load-dialect-plugin={self.htile_dialect}",
        ]

    def htile_pass_plugin_args(self) -> list[str]:
        return [f"--load-pass-plugin={self.htile_dialect}"]


def _native_library_names(target: str) -> tuple[str, ...]:
    suffix = _dynamic_library_suffix()
    if sys.platform == "win32":
        return (f"{target}{suffix}", f"lib{target}{suffix}")
    return (f"lib{target}{suffix}",)


def _native_library_name(target: str) -> str:
    return _native_library_names(target)[0]


def _dynamic_library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _find_native_library(directory: Path, target: str) -> Path | None:
    for name in _native_library_names(target):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _find_plugin_set(directory: Path) -> NeptunePlugins | None:
    loop_transform = _find_native_library(directory, "LoopTransform")
    ta_dialect = _find_native_library(directory, "TADialect")
    htile_dialect = _find_native_library(directory, "HTileDialect")
    if loop_transform is None or ta_dialect is None or htile_dialect is None:
        return None
    return NeptunePlugins(
        loop_transform=loop_transform,
        ta_dialect=ta_dialect,
        htile_dialect=htile_dialect,
    )


def _append_candidate(candidates: list[Path], directory: Path) -> None:
    directory = directory.expanduser()
    if directory not in candidates:
        candidates.append(directory)


def _append_build_dir_candidates(candidates: list[Path], build_dir: Path) -> None:
    _append_candidate(candidates, build_dir)
    if not build_dir.is_dir():
        return
    for child in sorted(build_dir.iterdir()):
        if child.is_dir():
            _append_candidate(candidates, child)


def _candidate_dirs() -> list[Path]:
    candidates: list[Path] = []

    env_dir = os.environ.get(NATIVE_DIR_ENV)
    if env_dir:
        _append_candidate(candidates, Path(env_dir))

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parents[1] if len(pkg_dir.parents) >= 2 else pkg_dir
    _append_candidate(candidates, pkg_dir / "_native")
    _append_build_dir_candidates(candidates, repo_root / "build")
    _append_build_dir_candidates(candidates, Path.cwd() / "build")

    return candidates


def find_neptune_plugins() -> NeptunePlugins | None:
    """Resolve the Neptune MLIR native plugin set.

    Resolution order:
    1. NEPTUNE_MLIR_NATIVE_DIR, containing all three plugin libraries.
    2. Installed package directory: neptune_mlir/_native.
    3. Common editable/development build dirs: <repo>/build and children.
    4. Current working directory build dirs: <cwd>/build and children.
    """
    for directory in _candidate_dirs():
        if not directory.is_dir():
            continue
        if plugins := _find_plugin_set(directory):
            return plugins
    return None


def find_plugin_path() -> Path | None:
    """Resolve the HTile dialect plugin path for existing translator entry points."""
    plugins = find_neptune_plugins()
    if plugins is None:
        return None
    return plugins.htile_dialect
