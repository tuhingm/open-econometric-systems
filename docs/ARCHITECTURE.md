# OES technical architecture

## Topology

Everything runs in the visitor's browser. The website (GitHub Pages, static files) serves the page, the engine's
Python files and the guide example datasets. A Web Worker (`oes/worker.js`) starts Pyodide 0.27.2 from the jsDelivr
CDN, loads NumPy, pandas, SciPy and statsmodels (plus scikit-learn and XGBoost for the machine-learning models, on
demand), writes the engine files into Pyodide's file system and calls `oes_engine.run_json`. No server is involved,
and uploaded files are read locally and never sent anywhere.

## Layers

1. **Presentation** (`almamun/oes/index.html`): model catalogue, CSV upload or guide example, variable operations,
   a form generated from the catalogue, results (summary tiles, tables, SVG charts, notes), downloads, a
   "reproduce in Python" snippet and the in-browser test suite.
2. **Worker** (`almamun/oes/worker.js`): Pyodide start-up, package loading, one estimation at a time.
3. **Engine** (`oes-engine/oes_engine`): `run(model_id, csv_text, spec)`; `prepare()` applies data operations;
   `timeseries.py`, `panel.py`, `causal.py` and `ml.py` hold the estimators; the `_*.py` files are the helper files
   published with the guides, used unchanged.
4. **Validation** (`oes-engine/tests`): reference comparisons, published results and simulations (see the README).

## Result format

```
{"model", "title", "nobs", "guide", "engine",
 "summary": [{"label", "value"}],
 "tables":  [{"title", "columns", "rows", "note"}],
 "charts":  [{"title", "x", "series": [{"name", "y"}], "band": {"lo", "hi"}, "kind": "line|bar|scatter"}],
 "notes":   [str]}
```
Errors in the data or specification return `{"error": message}`; the engine never raises to the page.

## Numerical standard for new methods

1. Recover known parameters from simulated data.
2. Compare with an independent reference implementation or a published example.
3. Test covariance options and small-sample corrections separately.
4. Fix random seeds; record the package versions (those of Pyodide) used for the reference values.

## Optional server

`oes-backend/` (FastAPI, OLS only) and `oes-frontend/oes-client.js` are kept for anyone who wants a server API; the
website does not use them.
