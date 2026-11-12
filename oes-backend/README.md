# Open Econometric Systems (OES)

OES is an independent, free, open-source econometric computation service designed to run from a browser-based research website. The website is a client; this Python service performs estimation.

## Status

**Bootstrap / Phase 0.** This repository scaffold includes the API contract, model catalogue, a working OLS estimator, tests, and deployment configuration. The catalogue is a roadmap, not a claim that every listed estimator has already been implemented. Each method becomes available only after its implementation and numerical tests are merged.

## Architecture

- Browser: HTML/CSS/JavaScript embedded in the existing academic website.
- API: FastAPI, JSON request/response contract.
- Computation: NumPy, SciPy, pandas, statsmodels, and independently validated specialist packages.
- Deployment: Render-compatible Python web service (or Docker-compatible host).
- Source control: GitHub.
- Results: JSON tables and diagnostics; downloadable artifacts can be added in later phases.

## Local development

```bash
cd oes-backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
uvicorn oes_api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## First implemented endpoint

`POST /api/v1/estimate/ols`

Example:

```json
{
  "data": {
    "y": [1.2, 2.1, 2.8, 4.2, 5.1],
    "x": [[1, 0.2], [1, 0.4], [1, 0.6], [1, 0.8], [1, 1.0]]
  },
  "variable_names": ["const", "x1"],
  "covariance": "nonrobust"
}
```

The caller supplies the design matrix explicitly. The intercept is not silently added. Responses include estimates, standard errors, t statistics, p values, confidence intervals, fit statistics, and residual diagnostics.

## Model implementation policy

1. No estimator is listed as production-ready until it has a documented specification, implementation, deterministic tests, reference comparisons, and numerical edge-case tests.
2. Keep estimators independent of HTTP request handling.
3. Preserve explicit options: deterministic terms, lag selection, covariance estimator, sample, optimizer, convergence tolerances, and missing-data treatment.
4. Return warnings and convergence status; never silently convert failed estimation into success.
5. Do not copy proprietary EViews source code, binaries, interfaces, or documentation text. Implement published methods independently.

## Deployment

Set the service root directory to `oes-backend`, install with `pip install .`, and start with:

```bash
uvicorn oes_api.main:app --host 0.0.0.0 --port $PORT
```

Set `OES_ALLOWED_ORIGINS` to the exact website origin(s), e.g. `https://tuhingm.github.io`. Configure the API base URL in the website JavaScript client.

## Roadmap

See [ARCHITECTURE.md](../docs/ARCHITECTURE.md) and [MODEL_CATALOG.md](../docs/MODEL_CATALOG.md).
