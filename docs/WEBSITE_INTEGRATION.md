# Integrating OES into the academic website

The page lives in the website repository (`tuhingm/almamun`) at `oes/`:

| File | Purpose |
|---|---|
| `oes/index.html` | the workspace: model choice, data upload or guide example, variable operations, specification form, results, test suite |
| `oes/worker.js` | Web Worker that starts Pyodide 0.27.2, loads the engine and runs `oes_engine.run_json` |
| `oes/engine/oes_engine/*.py` | a copy of `oes-engine/oes_engine` from this repository |
| `oes/engine/catalog.json` | `oes_engine.catalog()`: models, fields, examples, Pyodide packages |
| `oes/engine/expected.json` | reference results of every guide example for the in-browser test suite |

## Updating the engine on the website

```bash
# in the website repository, with the Pyodide 0.27.2 package versions installed
pip install numpy==2.0.2 pandas==2.2.3 scipy==1.14.1 statsmodels==0.14.4 scikit-learn==1.6.1 xgboost==2.1.2
python3 tools/oes_sync.py /path/to/open-econometric-systems
```

Then bump `__version__` in `oes_engine/__init__.py` (it is used to bypass browser caches) and commit both repositories.

## Adding a model

1. Write the estimator in `oes-engine/oes_engine/` returning `core.result(...)`.
2. Register it in `MODELS` in `oes_engine/__init__.py` with its input fields and a guide example.
3. Add a reference test to `oes-engine/tests/test_models.py`.
4. Run `tools/oes_sync.py` in the website repository. The page builds its forms from the catalogue, so no HTML changes are needed.
