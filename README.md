# Neptune MLIR

Neptune MLIR provides an HTile MLIR dialect plugin and Python translators for
Triton, TileLang, and cuTile backends.

## Development

```bash
pip install -e .[dev]
cmake -S . -B build
cmake --build build
pytest -q
```
