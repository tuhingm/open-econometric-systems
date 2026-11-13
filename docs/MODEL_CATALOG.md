# OES model catalogue

All 46 models below are implemented in `oes-engine` and run in the browser at
<https://tuhingm.github.io/almamun/oes/>. `oes_engine.catalog()` is the machine-readable version (input fields,
guide page, example dataset, Pyodide packages).

| id | Model | Guide page | Implementation | Reference used in the tests |
|---|---|---|---|---|
| `ols` | OLS with diagnostics | research-guide/ols | statsmodels | statsmodels |
| `ardl` | ARDL bounds test | research-guide/ardl | statsmodels ARDL/UECM | guide (F = 8.18, long run 1.98) |
| `nardl` | Nonlinear ARDL | research-guide/nardl | partial sums + UECM, Wald symmetry tests | partial-sum identities |
| `var` | VAR | research-guide/var | statsmodels | statsmodels |
| `vecm` | VECM | research-guide/vecm | statsmodels (Johansen) | statsmodels |
| `iv_gmm` | 2SLS and GMM | research-guide/gmm | NumPy | linearmodels IV2SLS / IVGMM |
| `lp` | Local projections | research-guide/lp | NumPy (HAC, lag augmentation, LP-IV) | statsmodels OLS and VAR impulse responses |
| `pols` | Pooled OLS | panel-data-guide/pols | NumPy | linearmodels PooledOLS |
| `fe` | Fixed effects | panel-data-guide/fe | NumPy (within, two-way) | linearmodels PanelOLS |
| `re` | Random effects | panel-data-guide/re | NumPy (Swamy-Arora) | linearmodels RandomEffects |
| `fd` | First differences | panel-data-guide/diff | NumPy | linearmodels PanelOLS on differences |
| `fedk` | FE, Driscoll-Kraay | panel-data-guide/fedk | NumPy | linearmodels (kernel) |
| `diff_gmm` | Difference GMM | panel-data-guide/gmm | guide helper `diff_gmm.py` | Arellano and Bond (1991) |
| `mg` | Mean group / CCE-MG | panel-data-guide/mg | guide helper `panel_estimators.py` | unit-by-unit OLS |
| `pmg` | PMG / MG / DFE | panel-data-guide/pmg | guide helper | convergence, Hausman |
| `csardl` | CS-ARDL | panel-data-guide/csardl | guide helper | simulated panel |
| `plp` | Panel local projections | panel-data-guide/plp | NumPy (within, clustered, Driscoll-Kraay) | linearmodels PanelOLS |
| `did` | Difference-in-differences | causal-inference-guide/did | guide helper `causal_tests.py` | simulation; TWFE vs linearmodels |
| `staggered` | Staggered DiD | causal-inference-guide/staggered | guide helper | runs on guide data |
| `its` | Interrupted time series | causal-inference-guide/its | guide helper | runs on guide data |
| `iv` | IV (2SLS) | causal-inference-guide/iv | guide helper | linearmodels IV2SLS |
| `rdd` | Regression discontinuity | causal-inference-guide/rdd | guide helper | guide helper `rd_estimate` |
| `scm` | Synthetic control | causal-inference-guide/scm | guide helper | weight constraints |
| `gsc` | Generalized synthetic control | causal-inference-guide/gsc | guide helper | runs on guide data |
| `lasso` | LASSO | ml-guide/lasso | scikit-learn | scikit-learn |
| `ridge` | Ridge | ml-guide/ridge | scikit-learn | runs on guide data |
| `enet` | Elastic net | ml-guide/enet | scikit-learn | runs on guide data |
| `rf` | Random forest | ml-guide/rf | scikit-learn | runs on guide data |
| `xgb` | XGBoost | ml-guide/xgb | xgboost (native API) | xgboost |
| `dml_plr` | Double ML, partially linear | ml-guide/dml-plr | guide cross-fitting | simulation |
| `dml_irm` | Double ML, interactive | ml-guide/dml-irm | guide cross-fitting | simulation |
| `dml_iv` | Double ML, partially linear IV | ml-guide/dml-iv | guide cross-fitting | simulation |
| `causal_forest` | Causal forest (DR-learner forest) | ml-guide/causal-forest | AIPW scores + random forest | simulation (heterogeneity found) |
| `bayes_reg` | Bayesian linear regression | bayesian-guide/bayes-reg | NIG conjugate or Gibbs | OLS limit, g-prior shrinkage g/(1+g) |
| `bvar` | BVAR (Minnesota, NW, INW, ND; block exogeneity) | bayesian-guide/bvar | BEAR formulas; Gibbs for INW/ND | OLS limit, prior limit, analytic posterior, Monte Carlo marginal likelihood |
| `favar` | FAVAR | bayesian-guide/favar | two-step PCA + BVAR (Bernanke, Boivin and Eliasz 2005) | principal components |
| `mabvar` | Mean-adjusted BVAR | bayesian-guide/steady-state | Villani (2009) Gibbs | simulated steady state; tight-prior limit |
| `pvar` | Panel VAR (6 estimators) | bayesian-guide/panel-var | BEAR chapter 4 | mean group, pooled OLS limit, simulated persistence |
| `tvpvar` | TVP-VAR | bayesian-guide/tvp-var | random-walk coefficients, Carter-Kohn | simulated break |
| `svvar` | SV-VAR | bayesian-guide/sv-var | Cogley-Sargent, KSC mixture | simulated volatility rise |
| `mfbvar` | Mixed-frequency BVAR | bayesian-guide/mixed-frequency | monthly state space, quarterly aggregation | simulated latent months |
| `tvar` | Threshold BVAR | bayesian-guide/threshold-var | NW marginal likelihood on a threshold/delay grid | simulated threshold and delay |
| `ms_reg` | Markov-switching regression | markov-guide/ms-regression | statsmodels | Stata mswitch manual |
| `ms_ar` | Markov-switching AR | markov-guide/ms-ar | statsmodels | Hamilton (1989) |
| `msvar` | Markov-switching VAR | markov-guide/ms-var | EM (Hamilton filter, Kim smoother) | statsmodels likelihood; simulation |
| `mspvar` | Markov-switching panel VAR | markov-guide/ms-panel-var | EM with a common regime path | simulation |

Notes
- The causal forest is a doubly robust forest, not EconML's `CausalForestDML` (EconML does not run in Pyodide).
- The double-ML learners use lighter settings than the guides (100 trees by default) so they run in the browser;
  the number of trees, folds and repetitions can be raised in the form.
- XGBoost is called through its native `train` API because `XGBRegressor` 2.1 is not compatible with the
  scikit-learn 1.6 shipped with Pyodide 0.27.
