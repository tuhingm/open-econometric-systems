"""Difference GMM (Arellano and Bond, 1991) for dynamic panel models, written out in plain NumPy for teaching.

    result = diff_gmm(df, unit="firm", time="year", y="n", lags_y=2, x=["w", "k"], gmm_lags=(2, None), collapse=False)

Model:  y(i,t) = a1*y(i,t-1) + ... + b'x(i,t) + year effects + fixed effect(i) + error(i,t)

The equation is first-differenced to remove the fixed effect. The differenced lags of y are instrumented with levels of y
dated t-2 and earlier (gmm_lags gives the first and last lag used; None means all available). The x variables are
treated as strictly exogenous and instrument themselves. Year dummies are always included.

Reported: one-step estimates with standard errors robust to heteroskedasticity and within-unit correlation, the
Arellano-Bond tests for first- and second-order serial correlation in the differenced residuals, and the Hansen J test
from the two-step estimator. With collapse=True one instrument column is used per lag instead of one per lag and period.
The function reproduces the standard published results for the Arellano-Bond employment data (see the 2SLS / GMM guide).
"""
import numpy as np
import pandas as pd
from scipy import stats


def diff_gmm(df, unit, time, y, lags_y=1, x=(), gmm_lags=(2, None), collapse=False):
    x = list(x)
    d = df[[unit, time, y] + x].dropna(subset=[unit, time, y]).sort_values([unit, time])
    first_lag, last_lag = gmm_lags
    rows = []                                   # one entry per usable differenced observation
    for i, g in d.groupby(unit):
        level = dict(zip(g[time], g[y]))
        xs = {t: v for t, v in zip(g[time], g[x].values)}
        for t in g[time]:
            need = [t - k for k in range(lags_y + 2)]            # y(t), ..., y(t - lags_y - 1) must all exist
            if not all(s in level for s in need) or (t - 1) not in xs:
                continue
            if x and (np.isnan(xs[t]).any() or np.isnan(xs[t - 1]).any()):
                continue
            dy = level[t] - level[t - 1]
            dlags = [level[t - k] - level[t - k - 1] for k in range(1, lags_y + 1)]
            dx = list(xs[t] - xs[t - 1])
            inst = {}                                            # GMM-style instruments: levels of y dated t-2 and earlier
            k = first_lag
            while (t - k) in level and (last_lag is None or k <= last_lag):
                inst[k if collapse else (t, k)] = level[t - k]
                k += 1
            rows.append((i, t, dy, dlags + dx, inst))
    years = sorted({r[1] for r in rows})
    inst_cols = sorted({c for r in rows for c in r[4]}, key=str)
    n_obs, col = len(rows), {c: j for j, c in enumerate(inst_cols)}
    Y = np.array([r[2] for r in rows])
    D = np.array([[1.0 if r[1] == s else 0.0 for s in years] for r in rows])          # year dummies
    X = np.column_stack([np.array([r[3] for r in rows]), D])
    Zg = np.zeros((n_obs, len(inst_cols)))
    for j, r in enumerate(rows):
        for c, v in r[4].items():
            Zg[j, col[c]] = v
    Z = np.column_stack([Zg, np.array([r[3][lags_y:] for r in rows]).reshape(n_obs, len(x)), D])
    names = [f"L{k}.{y}" for k in range(1, lags_y + 1)] + x + [f"year_{s}" for s in years]

    units = np.array([r[0] for r in rows]); times = np.array([r[1] for r in rows])
    groups = [np.where(units == i)[0] for i in pd.unique(units)]
    ZHZ = np.zeros((Z.shape[1], Z.shape[1]))
    for idx in groups:                                           # one-step weighting matrix
        t_ = times[idx]
        H = 2.0 * np.eye(len(idx)) - (np.abs(t_[:, None] - t_[None, :]) == 1)
        ZHZ += Z[idx].T @ H @ Z[idx]
    W1 = np.linalg.pinv(ZHZ)
    XZ = X.T @ Z
    A = np.linalg.inv(XZ @ W1 @ XZ.T)
    beta = A @ XZ @ W1 @ (Z.T @ Y)
    e = Y - X @ beta
    S = sum(np.outer(Z[idx].T @ e[idx], Z[idx].T @ e[idx]) for idx in groups)
    V = A @ XZ @ W1 @ S @ W1 @ XZ.T @ A                           # robust (clustered) variance of the one-step estimator
    se = np.sqrt(np.diag(V))

    W2 = np.linalg.pinv(S)                                       # two-step estimator, used for the Hansen test
    beta2 = np.linalg.inv(XZ @ W2 @ XZ.T) @ XZ @ W2 @ (Z.T @ Y)
    e2 = Y - X @ beta2
    hansen = float((Z.T @ e2) @ W2 @ (Z.T @ e2))
    n_inst = int(np.linalg.matrix_rank(Z))
    df_h = n_inst - X.shape[1]

    def ar_test(m):                                              # Arellano-Bond test of order-m serial correlation
        w = np.zeros(n_obs)
        pos = {(u, t): j for j, (u, t) in enumerate(zip(units, times))}
        for j, (u, t) in enumerate(zip(units, times)):
            if (u, t - m) in pos:
                w[j] = e[pos[(u, t - m)]]
        num = w @ e
        wX = w @ X
        var = (sum((w[idx] @ e[idx]) ** 2 for idx in groups)
               - 2 * wX @ A @ XZ @ W1 @ sum((Z[idx].T @ e[idx]) * (e[idx] @ w[idx]) for idx in groups)
               + wX @ V @ wX)
        z = num / np.sqrt(var)
        return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))

    table = pd.DataFrame({"coef": beta, "std err": se, "z": beta / se, "p-value": 2 * (1 - stats.norm.cdf(np.abs(beta / se)))}, index=names)
    return {"table": table, "nobs": n_obs, "units": len(groups), "instruments": n_inst,
            "ar1": ar_test(1), "ar2": ar_test(2), "hansen": (hansen, df_h, float(1 - stats.chi2.cdf(hansen, df_h)))}
