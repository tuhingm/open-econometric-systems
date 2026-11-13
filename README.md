# Open Econometric Systems (OES)

A free, open-source econometric workspace. It runs every model of the four research guides on
[tuhingm.github.io/almamun](https://tuhingm.github.io/almamun/) — time series, panel data, causal inference and
machine learning — plus Bayesian VARs and Markov-switching models, on the user's own data, **inside the browser**. Nothing is installed and the data never leave the
user's computer.

**Use it:** <https://tuhingm.github.io/almamun/oes/>

## Models (51)

| Group | Models |
|---|---|
| Time series | OLS with diagnostics · ARDL bounds test · NARDL · VAR · VECM · 2SLS and GMM · Local projections · Structural VAR (Cholesky, long-run, proxy) · Structural breaks (Bai-Perron) · Threshold regression (TAR / SETAR) · Smooth-transition regression (LSTAR / ESTAR) |
| Panel data | Pooled OLS · Fixed effects · Random effects · First differences · FE with Driscoll-Kraay errors · Difference GMM · Mean group / CCE-MG · PMG / MG / DFE · CS-ARDL · Panel local projections · Panel threshold regression (Hansen) |
| Causal inference | Difference-in-differences · Staggered DiD (Callaway-Sant'Anna) · Interrupted time series · IV (2SLS) · Regression discontinuity · Synthetic control · Generalized synthetic control |
| Machine learning | LASSO · Ridge · Elastic net · Random forest · XGBoost · Double ML (partially linear, interactive, IV) · Causal forest |
| Bayesian | Bayesian linear regression (exact or Gibbs) · BVAR with Minnesota, Normal-Wishart, Independent Normal-Wishart or Normal-diffuse priors and block exogeneity · FAVAR · Mean-adjusted (steady-state) BVAR · Panel VAR (OLS mean group, Bayesian pooled, Zellner-Hong, hierarchical, static factor, dynamic factor) · TVP-VAR · Stochastic-volatility VAR · Mixed-frequency BVAR · Threshold BVAR |
| Markov switching | Markov-switching regression · Markov-switching AR (Hamilton) · Markov-switching VAR (MSI, MSIH, MSIAH) · Markov-switching panel VAR |

The Bayesian VARs follow the ECB's BEAR toolbox (Dieppe, Legrand and van Roye 2016). The Bayesian and Markov-switching
groups have their own guides on the website, [Bayesian Guide](https://tuhingm.github.io/almamun/bayesian-guide/) and
[Markov-Switching Guide](https://tuhingm.github.io/almamun/markov-guide/): one page per model, with pre-estimation tests,
estimation, convergence and regime checks, diagnostics, robustness and text for the paper.

DSGE models are not included: they are solved and estimated in Dynare (see the DSGE Guide).

## How it works

```
oes-engine/oes_engine/     pure-Python estimators (NumPy, pandas, SciPy, statsmodels, scikit-learn, XGBoost)
        │  copied by tools/oes_sync.py in the website repository
        ▼
almamun/oes/               the web page; oes/worker.js runs the engine in Pyodide (Python compiled to WebAssembly)
```

The engine has one entry point, `oes_engine.run(model_id, csv_text, spec)`, which returns tables, summary
statistics, charts and notes in one format for every model. The same call works on a laptop:

```bash
git clone https://github.com/tuhingm/open-econometric-systems
pip install -e "open-econometric-systems/oes-engine[test]"
```

```python
import oes_engine
r = oes_engine.run("ardl", open("yield.csv").read(),
                   {"y": "yield_10y", "x": ["inflation", "gdp_growth", "unemployment"], "time": "date"})
print(r["summary"])
```

`oes_engine.catalog()` lists every model with its input fields and its guide example. Data operations (lags,
differences, logs, growth rates, cross-sectional demeaning, sample restriction) are passed as `spec["ops"]`.

## Testing

`oes-engine/tests/` (161 tests) checks that every model runs on its guide example and compares the
estimates with reference implementations and published results:

- OLS, VAR and VECM against statsmodels; 2SLS and GMM (all covariance options, Hansen J) against linearmodels;
- pooled OLS, fixed effects (one- and two-way; conventional, robust, clustered and Driscoll-Kraay errors), random
  effects and first differences against linearmodels, to its small-sample scaling;
- difference GMM against Arellano and Bond (1991); the ARDL bounds test and long-run coefficients against the guide;
- 2SLS (Card) and two-way fixed effects against linearmodels; DiD, double ML (PLR, IRM, IV), CS-ARDL and the causal
  forest on simulated data with known effects; LASSO and the XGBoost wrapper against scikit-learn and xgboost;
- Markov-switching regression and AR against Stata's `mswitch` manual and Hamilton (1989); the MS-VAR likelihood
  against statsmodels; BVARs against the OLS VAR (loose prior), the prior (tight prior) and a Monte Carlo marginal
  likelihood; TVP-, SV-, mixed-frequency, threshold, steady-state and panel VARs on simulated data with known features.

```bash
cd oes-engine
OES_SITE=/path/to/almamun python -m pytest -q      # OES_FAST=1 skips the slow models
```

The tests run on GitHub Actions with the package versions of Pyodide 0.27.2. The web page also has a
**Run the test suite** button that runs every model in the visitor's browser and compares the results with the
reference values in `almamun/oes/engine/expected.json`.

## Optional server API

`oes-backend/` is an earlier FastAPI service with an OLS endpoint (`render.yaml` deploys it to Render). The website
does not need it: all estimation runs in the browser.
