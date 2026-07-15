"""Helpers for locating native artifacts in the Neptune distribution."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from importlib.metadata import distribution
from pathlib import Path
from warnings import warn

NEPTUNE_OPT_ENV_VAR = "NEPTUNE_MLIR_OPT"
_DIST = distribution("neptune-mlir")
_NATIVE_EXTENSION_MODULE = "neptune_mlir._neptuneMlir"
_NATIVE_EXTENSION_TARGET = "_neptuneMlir"


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _find_unique_distribution_file(name: str, predicate) -> Path | None:
    candidates = [
        file_path.resolve()
        for file in _DIST.files or []
        if name in file.name and predicate(file_path := Path(_DIST.locate_file(file)))
    ]
    if len(candidates) > 1:
        warn(
            f"Multiple candidates found for {name} in distribution {_DIST.metadata['Name']}: "
            f"{candidates}. Unable to resolve one."
        )
    return candidates[0] if len(candidates) == 1 else None


def find_neptune_opt() -> Path | None:
    """Find neptune-opt from an explicit override, package metadata, or PATH."""
    if env_value := os.environ.get(NEPTUNE_OPT_ENV_VAR):
        candidate = Path(env_value).expanduser().resolve()
        return candidate if _is_executable(candidate) else None

    if candidate := _find_unique_distribution_file("neptune-opt", _is_executable):
        return candidate

    if candidate := shutil.which("neptune-opt"):
        return Path(candidate).resolve()
    return None


def _load_neptune_native_extension():
    if module := sys.modules.get(_NATIVE_EXTENSION_MODULE):
        return module

    ext_path = _find_unique_distribution_file(_NATIVE_EXTENSION_TARGET, Path.is_file)
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


def register_dialects(context, *, load: bool = True) -> None:
    """Register Neptune dialects available through the native extension."""
    _load_neptune_native_extension().register_htile_dialect(context, load)
