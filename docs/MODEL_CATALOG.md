# OES Extended Model Catalogue

Statuses: **implemented** means an executable, tested endpoint exists; **planned** means scoped but not yet executable. The API catalogue is authoritative for live availability.

| Domain | Methods in scope |
|---|---|
| Linear and nonlinear regression | OLS, WLS, GLS, feasible GLS, robust covariance, LAD, nonlinear least squares |
| Instrumental variables and GMM | 2SLS, LIML, k-class, 3SLS, one-step/two-step GMM, continuously updated GMM |
| Time series | AR, MA, ARMA, ARIMA, ARFIMA, ARMAX, distributed lag, ARDL, UECM, NARDL |
| Unit roots and breaks | ADF, PP, KPSS, DF-GLS, Zivot-Andrews, Perron-type break tests |
| Cointegration | Engle-Granger, Johansen, FMOLS, DOLS, CCR, bounds testing |
| Multivariate systems | VAR, SVAR, VECM, Bayesian VAR, TVP-VAR, impulse responses, forecast error variance decomposition |
| Volatility | ARCH, GARCH, GJR-GARCH, EGARCH, TGARCH, APARCH, component GARCH, multivariate GARCH |
| Panel data | pooled OLS, FE, RE, between, first differences, panel GLS, panel IV |
| Dynamic panels | Difference GMM, System GMM, Arellano-Bond, Blundell-Bond |
| Heterogeneous panels | MG, PMG, DFE, CCE, CCEMG, AMG, CS-ARDL |
| Panel tests | LLC, IPS, Fisher-ADF/PP, CIPS, Pedroni, Kao, Westerlund, Pesaran CD |
| Limited dependent variables | binary/ordered Logit and Probit, Tobit, Heckman selection |
| Count and duration | Poisson, negative binomial, zero-inflated count, survival/duration models |
| Quantile methods | classical quantile regression, panel quantile, quantile ARDL |
| Nonlinear and regime models | threshold regression, TAR, SETAR, STAR, Markov switching, smooth transition |
| State-space | Kalman filter, smoother, local level/trend, time-varying parameter models |
| Factor methods | PCA, static factor models, dynamic factor models |
| Bayesian | Bayesian regression, BVAR, Bayesian panel models, MCMC diagnostics |
| Causal inference | DiD, event study, synthetic control, treatment effects, IV causal models |
| Machine learning | ridge, LASSO, elastic net, random forest, gradient boosting, XGBoost, support vector regression, neural networks |
| Structural macro | DSGE solution, likelihood construction, Bayesian estimation, impulse responses and stochastic simulation |
| Forecasting | static/dynamic, recursive, rolling, expanding-window, forecast combination and evaluation |

This is the intended extended scope, not a statement that all methods are already implemented. The initial bootstrap implements OLS only.
