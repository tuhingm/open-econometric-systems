import numpy as np
import statsmodels.api as sm

def fit_ols(y, x, variable_names=None, covariance="nonrobust", confidence_level=0.95):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.ndim != 1 or x.ndim != 2:
        raise ValueError("y must be one-dimensional and x must be a two-dimensional matrix.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("The number of rows in x must equal the number of observations in y.")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Input data must contain only finite numeric values.")
    if x.shape[0] <= x.shape[1]:
        raise ValueError("OLS requires more observations than regressors.")
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("The design matrix is rank deficient; coefficients are not uniquely identified.")
    if variable_names is not None and len(variable_names) != x.shape[1]:
        raise ValueError("variable_names must have one name per design-matrix column.")

    cov_type = "nonrobust" if covariance == "nonrobust" else covariance
    result = sm.OLS(y, x, hasconst=None).fit(cov_type=cov_type)
    ci = result.conf_int(alpha=1-confidence_level)
    names = variable_names or [f"x{i+1}" for i in range(x.shape[1])]

    return {
        "model": "OLS",
        "nobs": int(result.nobs),
        "df_model": float(result.df_model),
        "df_resid": float(result.df_resid),
        "r_squared": float(result.rsquared),
        "adjusted_r_squared": float(result.rsquared_adj),
        "f_statistic": None if result.fvalue is None else float(result.fvalue),
        "f_pvalue": None if result.f_pvalue is None else float(result.f_pvalue),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "covariance": covariance,
        "confidence_level": confidence_level,
        "coefficients": [
            {
                "name": names[i],
                "estimate": float(result.params[i]),
                "std_error": float(result.bse[i]),
                "statistic": float(result.tvalues[i]),
                "p_value": float(result.pvalues[i]),
                "ci_lower": float(ci[i, 0]),
                "ci_upper": float(ci[i, 1]),
            }
            for i in range(len(names))
        ],
        "residuals": [float(v) for v in result.resid],
        "fitted": [float(v) for v in result.fittedvalues],
        "warnings": [],
    }
