from pathlib import Path

from neptune_mlir import dist


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_explicit_neptune_opt_override(tmp_path, monkeypatch) -> None:
    executable = _make_executable(tmp_path / "custom-neptune-opt")
    monkeypatch.setenv(dist.NEPTUNE_OPT_ENV_VAR, str(executable))

    assert dist.find_neptune_opt() == executable.resolve()


def test_invalid_explicit_override_does_not_fall_back(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(dist.NEPTUNE_OPT_ENV_VAR, str(tmp_path / "missing"))
    monkeypatch.setattr(dist, "_find_unique_distribution_file", lambda *_: Path("unexpected"))

    assert dist.find_neptune_opt() is None


def test_finds_neptune_opt_on_path(tmp_path, monkeypatch) -> None:
    executable = _make_executable(tmp_path / "neptune-opt")
    monkeypatch.delenv(dist.NEPTUNE_OPT_ENV_VAR, raising=False)
    monkeypatch.setattr(dist, "_find_unique_distribution_file", lambda *_: None)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert dist.find_neptune_opt() == executable.resolve()


def test_register_htile_dialect_parses_custom_form_htile() -> None:
    from mlir import ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        dist.register_dialects(ctx)
        module = ir.Module.parse(Path("test/python/data/causal_attention_htile.mlir").read_text())
    op_names = [op.operation.name for op in module.body.operations]
    assert op_names == ["func.func", "htile.kernel"]
