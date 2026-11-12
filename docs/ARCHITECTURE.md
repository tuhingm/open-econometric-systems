# OES Technical Architecture

## Deployment topology

The existing static website remains on GitHub Pages. The Python computation service is deployed independently. Browser requests go from the website to the API over HTTPS. The browser must never receive server credentials or deployment secrets.

## Logical layers

1. **Presentation:** accessible HTML forms, model catalogue, dataset preview, results tables, Plotly visualizations.
2. **API gateway:** FastAPI routes, schema validation, CORS allowlist, request-size limits, execution time limits, structured errors.
3. **Application services:** dataset validation, sample selection, transformation pipeline, estimator dispatch, result serialization.
4. **Numerical engine:** pure Python estimator modules; no HTTP or UI dependencies.
5. **Validation:** unit tests, simulation-based tests, reference fixtures, property tests, cross-package comparisons.
6. **Operations:** structured logs, health checks, rate limits, job queue for expensive tasks, retention policy and resource limits.

## API contract

- `GET /health`: service health.
- `GET /api/v1/models`: machine-readable catalogue and implementation status.
- `POST /api/v1/estimate/{model_id}`: validated estimation request.
- Future: `POST /api/v1/jobs`, `GET /api/v1/jobs/{job_id}`, artifact download endpoints.

Requests should carry data, variable identifiers, sample selection, model-specific specification, covariance options and optimizer settings. Responses should include model metadata, estimates, covariance matrix, inference statistics, diagnostics, convergence state, warnings and reproducibility metadata.

## Estimator interface

Each estimator should expose:
- a typed request schema;
- a pure `fit(data, specification) -> Result` implementation;
- deterministic tests;
- a method note describing objective function and assumptions;
- reference fixtures with tolerances;
- explicit support/status metadata.

## Numerical validation standard

For each method:
1. Recover known parameters from simulated data.
2. Compare with an independent reference implementation or published numerical example.
3. Test small samples, collinearity, missing values, scaling and boundary conditions.
4. Test covariance estimators and finite-sample corrections separately.
5. Verify optimizer convergence, gradients and Hessians where applicable.
6. Record random seeds and package versions for stochastic procedures.

## Security and public service controls

The initial OLS endpoint is stateless and does not persist uploaded datasets. Before opening the service to unrestricted public traffic, add:
- request body and row/column limits;
- per-IP rate limiting and concurrency limits;
- execution timeouts and memory ceilings;
- queued workers for long-running Bayesian, GARCH, panel and DSGE jobs;
- strict CORS origins;
- safe file parsing and temporary-file cleanup;
- abuse monitoring and operational alerts.

CORS is not authentication or rate limiting. Do not treat it as a security boundary.

## Website integration

Load the OES client script from the academic website and configure the deployed API base URL. The interface should call `GET /api/v1/models` to render the live catalogue, then post selected specifications to the relevant endpoint. Display only models marked `implemented` as runnable; show planned models as roadmap entries.

## Repository strategy

The GitHub connector available for this session can add files to an existing repository but cannot create a new GitHub repository. This bootstrap is therefore staged under `oes-backend/` in the existing website repository on a separate branch. Once the user creates a dedicated `tuhingm/open-econometric-systems` repository, this directory can be moved there and deployed with `oes-backend` as its root directory.
