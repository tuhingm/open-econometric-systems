# Open Econometric Systems (OES)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://dashboard.render.com/blueprint/new?repo=https://github.com/tuhingm/open-econometric-systems)

A free, open-source, browser-accessible econometric platform.

## Current implementation

The initial backend release provides a FastAPI service with:
- OLS estimation
- Conventional covariance and heteroskedasticity-consistent HC0–HC3 covariance estimators
- Coefficient estimates, standard errors, test statistics, p-values and confidence intervals
- R-squared, adjusted R-squared, F-statistic, log-likelihood, AIC and BIC
- Fitted values and residuals
- Health and model-catalog endpoints

The broader econometric catalogue is a development roadmap. Methods not implemented in the API must not be treated as available estimators.

## Repository layout

- `oes-backend/`: Python/FastAPI estimation API
- `oes-frontend/`: browser API client
- `docs/`: architecture, model catalogue and website integration notes
- `.github/workflows/`: automated lint and test workflow
- `render.yaml`: Render deployment blueprint

## Run locally

From `oes-backend/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn oes_api.main:app --reload
```

The API is then available at `http://127.0.0.1:8000`; interactive API documentation is at `/docs`.

## Website integration

The website interface is hosted separately in the academic website at:
https://tuhingm.github.io/almamun/oes/

Before live estimation works, deploy the API and set `OES_ALLOWED_ORIGINS` to include `https://tuhingm.github.io`. Enter the deployed API base URL in the OES workspace.

## Status

The interface and initial OLS API are under development. No claim of full EViews parity is made; the project will expand method-by-method with tests and documented assumptions.
