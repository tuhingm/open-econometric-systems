"""Heterogeneous panel estimators for long (macro) panels, written out in NumPy for teaching.

    mean_group(df, unit, time, y, x)                 Pesaran-Smith mean group (MG): one OLS per unit, slopes averaged
    cce_mg(df, unit, time, y, x)                     Pesaran (2006) common correlated effects mean group (CCE-MG)
    ardl_ec(df, unit, time, y, x, method="pmg")      panel ARDL(1,1) in error-correction form: "pmg" (Pesaran, Shin and
                                                     Smith 1999), "mg" (mean group) or "dfe" (dynamic fixed effects)
    cs_ardl(df, unit, time, y, x, pt=None)           Chudik et al. (2016) cross-sectionally augmented ARDL(1,1), MG
    hausman(b1, V1, b2, V2)                          Hausman test of b1 (efficient under H0) against b2 (consistent)

Data are in long format with one row per unit and period. Every function returns a dict with a coefficient table
(pandas DataFrame with coef, std err, z, p-value) and the unit-by-unit estimates where relevant.
Mean-group standard errors are the nonparametric ones of Pesaran and Smith: the standard deviation of the unit
estimates divided by the square root of the number of units.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _table(b, se, names):
    b, se = np.asarray(b, float), np.asarray(se, float)
    z = b / se
    return pd.DataFrame({"coef": b, "std err": se, "z": z, "p-value": 2 * stats.norm.sf(np.abs(z))}, index=names)


def _ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return b, e


def _mg(B, names):
    B = np.asarray(B)
    return _table(B.mean(0), B.std(0, ddof=1) / np.sqrt(len(B)), names)


def _units(df, unit, time, cols):
    d = df.sort_values([unit, time])
    return [(u, g[cols].reset_index(drop=True)) for u, g in d.groupby(unit, sort=False)]


def mean_group(df, unit, time, y, x, trend=False):
    x = list(x); B = []; ids = []
    for u, g in _units(df, unit, time, [y] + x):
        g = g.dropna()
        if len(g) < len(x) + 4:
            continue
        X = np.column_stack([np.ones(len(g))] + ([np.arange(len(g))] if trend else []) + [g[x].to_numpy(float)])
        b, _ = _ols(g[y].to_numpy(float), X)
        B.append(b[-len(x):]); ids.append(u)
    B = np.array(B)
    return {"table": _mg(B, x), "units": pd.DataFrame(B, index=ids, columns=x), "n": len(ids)}


def cce_mg(df, unit, time, y, x):
    x = list(x)
    avg = df.groupby(time)[[y] + x].mean().add_suffix("_bar")
    d = df.join(avg, on=time)
    B = []; ids = []
    bars = [c + "_bar" for c in [y] + x]
    for u, g in _units(d, unit, time, [y] + x + bars):
        g = g.dropna()
        if len(g) < 2 * len(x) + 5:
            continue
        X = np.column_stack([np.ones(len(g)), g[x].to_numpy(float), g[bars].to_numpy(float)])
        b, _ = _ols(g[y].to_numpy(float), X)
        B.append(b[1:1 + len(x)]); ids.append(u)
    B = np.array(B)
    return {"table": _mg(B, x), "units": pd.DataFrame(B, index=ids, columns=x), "n": len(ids)}


def _ec_data(df, unit, time, y, x):
    """Per unit: dy, y(-1), X(-1), W = [1, dX] for the ARDL(1,1) error-correction form."""
    out = []
    for u, g in _units(df, unit, time, [y] + x):
        g = g.dropna()
        Y, X = g[y].to_numpy(float), g[x].to_numpy(float)
        if len(Y) < len(x) * 2 + 6:
            continue
        dy, yl, Xl, dX = np.diff(Y), Y[:-1], X[:-1], np.diff(X, axis=0)
        W = np.column_stack([np.ones(len(dy)), dX])
        out.append((u, dy, yl, Xl, W))
    return out


def ardl_ec(df, unit, time, y, x, method="pmg", tol=1e-9, maxit=500):
    """Delta y_it = phi_i (y_i,t-1 - theta' x_i,t-1) + c_i + gamma_i' Delta x_it + e_it."""
    x = list(x); k = len(x)
    U = _ec_data(df, unit, time, y, x)
    # unit-by-unit OLS of the error-correction regression (used by MG and to start PMG)
    per = []
    for u, dy, yl, Xl, W in U:
        Z = np.column_stack([yl, Xl, W])
        b, e = _ols(dy, Z)
        phi, bx = b[0], b[1:1 + k]
        per.append(dict(unit=u, phi=phi, theta=-bx / phi, sr=b[2 + k:], s2=e @ e / (len(dy) - Z.shape[1])))
    if method == "mg":
        th = np.array([p["theta"] for p in per]); ph = np.array([p["phi"] for p in per]); sr = np.array([p["sr"] for p in per])
        return {"long_run": _mg(th, x), "phi": _mg(ph[:, None], ["ec"]),
                "short_run": _mg(sr, ["D." + v for v in x]), "units": pd.DataFrame(th, index=[p["unit"] for p in per], columns=x),
                "phi_units": ph, "n": len(per), "theta": th.mean(0), "V": np.cov(th.T, ddof=1).reshape(k, k) / len(th)}
    if method == "dfe":
        rows_y, rows_X, ids = [], [], []
        for j, (u, dy, yl, Xl, W) in enumerate(U):
            rows_y.append(dy - dy.mean()); Z = np.column_stack([yl, Xl, W[:, 1:]]); rows_X.append(Z - Z.mean(0)); ids += [j] * len(dy)
        Y, Z, ids = np.concatenate(rows_y), np.vstack(rows_X), np.array(ids)
        b, e = _ols(Y, Z)
        XtXi = np.linalg.inv(Z.T @ Z)
        meat = sum(np.outer(Z[ids == g].T @ e[ids == g], Z[ids == g].T @ e[ids == g]) for g in np.unique(ids))
        Vb = XtXi @ meat @ XtXi * len(U) / (len(U) - 1)
        phi = b[0]; theta = -b[1:1 + k] / phi
        J = np.column_stack([b[1:1 + k] / phi ** 2, -np.eye(k) / phi])        # delta method for theta = -b_x / phi
        Vt = J @ Vb[:1 + k, :1 + k] @ J.T
        return {"long_run": _table(theta, np.sqrt(np.diag(Vt)), x), "phi": _table([phi], [np.sqrt(Vb[0, 0])], ["ec"]),
                "short_run": _table(b[1 + k:], np.sqrt(np.diag(Vb)[1 + k:]), ["D." + v for v in x]), "n": len(U), "theta": theta, "V": Vt}
    # PMG: back-substitution algorithm of Pesaran, Shin and Smith (1999)
    theta = np.mean([p["theta"] for p in per], axis=0)
    Hs = [np.eye(len(dy)) - W @ np.linalg.pinv(W.T @ W) @ W.T for u, dy, yl, Xl, W in U]
    for it in range(maxit):
        phis, s2s = [], []
        for (u, dy, yl, Xl, W), H in zip(U, Hs):
            ec = yl - Xl @ theta
            a = H @ ec
            phi = (a @ (H @ dy)) / (a @ a)
            r = H @ (dy - phi * ec)
            phis.append(phi); s2s.append(r @ r / len(dy))
        A = np.zeros((k, k)); c = np.zeros(k)
        for (u, dy, yl, Xl, W), H, phi, s2 in zip(U, Hs, phis, s2s):
            HX = H @ Xl
            A += phi ** 2 / s2 * Xl.T @ HX
            c += phi / s2 * HX.T @ (dy - phi * yl)
        new = -np.linalg.solve(A, c)
        if np.max(np.abs(new - theta)) < tol:
            theta = new
            break
        theta = new
    V = np.linalg.inv(A)
    sr = []
    for (u, dy, yl, Xl, W), phi in zip(U, phis):
        g, _ = _ols(dy - phi * (yl - Xl @ theta), W)
        sr.append(g[1:])
    return {"long_run": _table(theta, np.sqrt(np.diag(V)), x), "phi": _mg(np.array(phis)[:, None], ["ec"]),
            "short_run": _mg(np.array(sr), ["D." + v for v in x]), "phi_units": np.array(phis), "n": len(U),
            "iterations": it + 1, "theta": theta, "V": V}


def cs_ardl(df, unit, time, y, x, pt=None):
    """CS-ARDL(1,1): y_it = c_i + lam_i y_i,t-1 + b0_i' x_it + b1_i' x_i,t-1 + sum_l d_il' zbar_t-l + e_it,
    with zbar = cross-sectional averages of (y, x) and l = 0..pt. Long run theta_i = (b0_i + b1_i) / (1 - lam_i)."""
    x = list(x); k = len(x)
    T = df[time].nunique()
    pt = int(np.floor(T ** (1 / 3))) if pt is None else pt
    zb = df.groupby(time)[[y] + x].mean().sort_index()
    lagged = pd.concat({f"zb{l}_{c}": zb[c].shift(l) for l in range(pt + 1) for c in [y] + x}, axis=1)
    d = df.join(lagged, on=time)
    zc = list(lagged.columns)
    TH, PH, ids = [], [], []
    for u, g in _units(d, unit, time, [y] + x + zc):
        Y, X = g[y].to_numpy(float), g[x].to_numpy(float)
        R = np.column_stack([np.ones(len(Y) - 1), Y[:-1], X[1:], X[:-1], g[zc].to_numpy(float)[1:]])
        ok = np.isfinite(R).all(1) & np.isfinite(Y[1:])
        if ok.sum() < R.shape[1] + 3:
            continue
        b, _ = _ols(Y[1:][ok], R[ok])
        lam = b[1]
        TH.append((b[2:2 + k] + b[2 + k:2 + 2 * k]) / (1 - lam)); PH.append(lam - 1); ids.append(u)
    TH, PH = np.array(TH), np.array(PH)
    return {"long_run": _mg(TH, x), "phi": _mg(PH[:, None], ["ec"]), "units": pd.DataFrame(TH, index=ids, columns=x),
            "phi_units": PH, "n": len(ids), "pt": pt}


def hausman(b1, V1, b2, V2):
    d = np.asarray(b2) - np.asarray(b1)
    Vd = np.asarray(V2) - np.asarray(V1)
    H = float(d @ np.linalg.pinv(Vd) @ d)
    return H, len(d), float(stats.chi2.sf(H, len(d)))
