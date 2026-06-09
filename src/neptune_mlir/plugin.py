"""Helpers for locating Neptune MLIR native plugin shared libraries."""

import importlib.util
import os
import sys
from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path
from warnings import warn

LIB_DIR_ENV_VAR = "NEPTUNE_MLIR_LIB_DIR"
DIST = distribution("neptune-mlir")
_NATIVE_EXTENSION_MODULE = "neptune_mlir._neptuneMlir"
_NATIVE_EXTENSION_TARGET = "_neptuneMlir"


def _get_lib_dir_env() -> Path | None:
    env_value = os.environ.get(LIB_DIR_ENV_VAR)
    return Path(env_value).expanduser() if env_value else None


def _find_file_in_dist(name_stem: str) -> Path | None:
    candidates = []
    for file in DIST.files or []:
        # No suffix detection. EXTENSION_SUFFIXES has been unreliable
        # (doesn't contain .dylib on macOS, for example)
        if name_stem in file.name:
            candidates.append(Path(DIST.locate_file(file)).resolve())  # type: ignore
    if len(candidates) > 1:
        warn(
            f"Multiple candidates found for file {name_stem} in distribution {DIST.metadata['Name']}: "
            f"{candidates}. Unable to resolve file."
        )
    return candidates[0] if len(candidates) == 1 else None


def _find_file_in_directory(dir: Path, name_stem: str) -> Path | None:
    candidates = [
        path
        for suffix in [".so", ".dylib", ".dll"]
        for path in dir.glob(f"*{name_stem}*{suffix}")  # Allow for a prefix (often "lib")
        if path.is_file()
    ]
    if len(candidates) > 1:
        warn(
            f"Multiple candidates found for file {name_stem} in directory {dir}: "
            f"{candidates}. Unable to resolve file."
        )
    return candidates[0] if len(candidates) == 1 else None


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


def _find_plugin_set(finder) -> NeptunePlugins | None:
    loop_transform = finder("LoopTransform")
    ta_dialect = finder("TADialect")
    htile_dialect = finder("HTileDialect")
    if loop_transform is None or ta_dialect is None or htile_dialect is None:
        return None
    return NeptunePlugins(
        loop_transform=loop_transform,
        ta_dialect=ta_dialect,
        htile_dialect=htile_dialect,
    )


def find_neptune_plugins() -> NeptunePlugins | None:
    """Resolve the Neptune MLIR native plugin set from installed distribution metadata."""
    if env_path := _get_lib_dir_env():
        return _find_plugin_set(lambda name: _find_file_in_directory(env_path, name))
    return _find_plugin_set(_find_file_in_dist)


def find_neptune_native_extension() -> Path | None:
    """Resolve the Neptune Python native extension used for MLIR registration."""
    if env_path := _get_lib_dir_env():
        extension_dir = env_path / "neptune_mlir" / "_mlir_libs"
        return _find_file_in_directory(extension_dir, _NATIVE_EXTENSION_TARGET)
    return _find_file_in_dist(_NATIVE_EXTENSION_TARGET)


def _load_neptune_native_extension():
    if module := sys.modules.get(_NATIVE_EXTENSION_MODULE):
        return module

    ext_path = find_neptune_native_extension()
    if ext_path is None:
        raise ImportError("failed to load Neptune MLIR native extension: no extension found")
    spec = importlib.util.spec_from_file_location(_NATIVE_EXTENSION_MODULE, ext_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"failed to load Neptune MLIR native extension: invalid spec for {ext_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NATIVE_EXTENSION_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except ImportError:
        sys.modules.pop(_NATIVE_EXTENSION_MODULE, None)
        raise
    return module


def register_htile_dialect(context, *, load: bool = True) -> None:
    """Register and optionally load the HTile dialect into an MLIR Python context."""
    _load_neptune_native_extension().register_htile_dialect(context, load)


def register_dialects(context, *, load: bool = True) -> None:
    """Register Neptune dialects available through the native Python extension."""
    register_htile_dialect(context, load=load)
