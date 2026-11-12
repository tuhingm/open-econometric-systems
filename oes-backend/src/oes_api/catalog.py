"""Machine-readable model catalogue and implementation status.

A catalogue entry is not an availability claim. Only entries marked implemented
are executable through the public API.
"""

MODEL_CATALOG = [
    {"id": "ols", "family": "linear", "name": "Ordinary Least Squares", "status": "implemented", "endpoint": "/api/v1/estimate/ols"},
    {"id": "wls", "family": "linear", "name": "Weighted Least Squares", "status": "planned"},
    {"id": "gls", "family": "linear", "name": "Generalized Least Squares", "status": "planned"},
    {"id": "iv_2sls", "family": "iv_gmm", "name": "Two-Stage Least Squares", "status": "planned"},
    {"id": "gmm", "family": "iv_gmm", "name": "Generalized Method of Moments", "status": "planned"},
    {"id": "arima", "family": "time_series", "name": "ARIMA", "status": "planned"},
    {"id": "ardl", "family": "time_series", "name": "ARDL / UECM", "status": "planned"},
    {"id": "nardl", "family": "time_series", "name": "Nonlinear ARDL", "status": "planned"},
    {"id": "vecm", "family": "multivariate", "name": "Vector Error Correction Model", "status": "planned"},
    {"id": "var", "family": "multivariate", "name": "Vector Autoregression", "status": "planned"},
    {"id": "svar", "family": "multivariate", "name": "Structural VAR", "status": "planned"},
    {"id": "bvar", "family": "bayesian", "name": "Bayesian VAR", "status": "planned"},
    {"id": "garch", "family": "volatility", "name": "GARCH family", "status": "planned"},
    {"id": "panel_fe", "family": "panel", "name": "Fixed Effects", "status": "planned"},
    {"id": "panel_re", "family": "panel", "name": "Random Effects", "status": "planned"},
    {"id": "system_gmm", "family": "panel", "name": "System GMM", "status": "planned"},
    {"id": "pmg_ardl", "family": "panel", "name": "Pooled Mean Group ARDL", "status": "planned"},
    {"id": "ccemg", "family": "panel", "name": "Common Correlated Effects MG", "status": "planned"},
    {"id": "logit", "family": "limited_dependent", "name": "Logit", "status": "planned"},
    {"id": "probit", "family": "limited_dependent", "name": "Probit", "status": "planned"},
    {"id": "tobit", "family": "limited_dependent", "name": "Tobit", "status": "planned"},
    {"id": "quantile", "family": "quantile", "name": "Quantile Regression", "status": "planned"},
    {"id": "kalman", "family": "state_space", "name": "State Space / Kalman Filter", "status": "planned"},
    {"id": "pca", "family": "factor", "name": "Principal Components", "status": "planned"},
    {"id": "threshold", "family": "nonlinear", "name": "Threshold Regression", "status": "planned"},
    {"id": "did", "family": "causal", "name": "Difference-in-Differences", "status": "planned"},
    {"id": "random_forest", "family": "machine_learning", "name": "Random Forest", "status": "planned"},
    {"id": "xgboost", "family": "machine_learning", "name": "Gradient Boosting / XGBoost", "status": "planned"},
    {"id": "dsge", "family": "structural", "name": "DSGE Solution and Estimation", "status": "planned"},
]

def list_models():
    return MODEL_CATALOG
