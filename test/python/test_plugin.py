from pathlib import Path

from neptune_mlir import plugin
from neptune_mlir.plugin import (
    NATIVE_DIR_ENV,
    NeptunePlugins,
    _native_library_name,
    find_neptune_plugins,
    find_plugin_path,
)


def _touch_plugins(directory: Path) -> None:
    for target in ("LoopTransform", "TADialect", "HTileDialect"):
        (directory / _native_library_name(target)).touch()


def test_finds_complete_plugin_set_from_native_dir_env(tmp_path, monkeypatch) -> None:
    _touch_plugins(tmp_path)
    monkeypatch.setenv(NATIVE_DIR_ENV, str(tmp_path))

    plugins = find_neptune_plugins()

    assert plugins == NeptunePlugins(
        loop_transform=(tmp_path / _native_library_name("LoopTransform")).resolve(),
        ta_dialect=(tmp_path / _native_library_name("TADialect")).resolve(),
        htile_dialect=(tmp_path / _native_library_name("HTileDialect")).resolve(),
    )


def test_plugin_args_are_ordered_for_transform_pipeline(tmp_path) -> None:
    plugins = NeptunePlugins(
        loop_transform=tmp_path / _native_library_name("LoopTransform"),
        ta_dialect=tmp_path / _native_library_name("TADialect"),
        htile_dialect=tmp_path / _native_library_name("HTileDialect"),
    )

    assert plugins.dialect_plugin_args() == [
        f"--load-dialect-plugin={tmp_path / _native_library_name('LoopTransform')}",
        f"--load-dialect-plugin={tmp_path / _native_library_name('TADialect')}",
        f"--load-dialect-plugin={tmp_path / _native_library_name('HTileDialect')}",
    ]
    assert plugins.htile_pass_plugin_args() == [
        f"--load-pass-plugin={tmp_path / _native_library_name('HTileDialect')}"
    ]


def test_find_plugin_path_returns_htile_compat_path(tmp_path, monkeypatch) -> None:
    _touch_plugins(tmp_path)
    monkeypatch.setenv(NATIVE_DIR_ENV, str(tmp_path))

    assert find_plugin_path() == (tmp_path / _native_library_name("HTileDialect")).resolve()


def test_incomplete_plugin_set_is_ignored(tmp_path, monkeypatch) -> None:
    (tmp_path / _native_library_name("HTileDialect")).touch()
    monkeypatch.setattr(plugin, "_candidate_dirs", lambda: [tmp_path])

    assert find_neptune_plugins() is None
