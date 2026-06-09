import importlib.machinery
from pathlib import Path

import pytest

from neptune_mlir.mlir_bindings import ir
from neptune_mlir.plugin import (
    LIB_DIR_ENV_VAR,
    NeptunePlugins,
    find_neptune_native_extension,
    find_neptune_plugins,
    register_htile_dialect,
)


def _plugin_library_name(stem: str) -> str:
    return f"lib{stem}{importlib.machinery.EXTENSION_SUFFIXES[0]}"


def _touch_plugins(directory: Path) -> None:
    for target in ("LoopTransform", "TADialect", "HTileDialect"):
        (directory / _plugin_library_name(target)).touch()


def _native_extension_name() -> str:
    return f"_neptuneMlir{importlib.machinery.EXTENSION_SUFFIXES[0]}"


def test_finds_complete_plugin_set_from_native_dir_env(tmp_path, monkeypatch) -> None:
    _touch_plugins(tmp_path)
    monkeypatch.setenv(LIB_DIR_ENV_VAR, str(tmp_path))

    plugins = find_neptune_plugins()

    assert plugins == NeptunePlugins(
        loop_transform=(tmp_path / _plugin_library_name("LoopTransform")).resolve(),
        ta_dialect=(tmp_path / _plugin_library_name("TADialect")).resolve(),
        htile_dialect=(tmp_path / _plugin_library_name("HTileDialect")).resolve(),
    )


def test_plugin_args_are_ordered_for_transform_pipeline(tmp_path) -> None:
    plugins = NeptunePlugins(
        loop_transform=tmp_path / _plugin_library_name("LoopTransform"),
        ta_dialect=tmp_path / _plugin_library_name("TADialect"),
        htile_dialect=tmp_path / _plugin_library_name("HTileDialect"),
    )

    assert plugins.dialect_plugin_args() == [
        f"--load-dialect-plugin={tmp_path / _plugin_library_name('LoopTransform')}",
        f"--load-dialect-plugin={tmp_path / _plugin_library_name('TADialect')}",
        f"--load-dialect-plugin={tmp_path / _plugin_library_name('HTileDialect')}",
    ]
    assert plugins.htile_pass_plugin_args() == [
        f"--load-pass-plugin={tmp_path / _plugin_library_name('HTileDialect')}"
    ]


def test_incomplete_plugin_set_is_ignored(tmp_path, monkeypatch) -> None:
    (tmp_path / _plugin_library_name("HTileDialect")).touch()
    monkeypatch.setenv(LIB_DIR_ENV_VAR, str(tmp_path))

    assert find_neptune_plugins() is None


def test_finds_native_extension_from_shared_candidate_roots(tmp_path, monkeypatch) -> None:
    _touch_plugins(tmp_path)
    extension_dir = tmp_path / "neptune_mlir" / "_mlir_libs"
    extension_dir.mkdir(parents=True)
    extension = extension_dir / _native_extension_name()
    extension.touch()
    monkeypatch.setenv(LIB_DIR_ENV_VAR, str(tmp_path))

    assert find_neptune_native_extension() == extension.resolve()


def test_register_htile_dialect_parses_custom_form_htile() -> None:
    if find_neptune_native_extension() is None:
        pytest.skip("Neptune MLIR Python native extension is required")

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        register_htile_dialect(ctx)
        module = ir.Module.parse(Path("test/python/data/flash_attention_htile.mlir").read_text())

    assert next(iter(module.body.operations)).name.value == "flash_attention_htile"
