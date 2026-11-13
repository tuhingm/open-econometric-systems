"""Structural VAR: identification by a recursive (Cholesky) ordering, by long-run restrictions (Blanchard-Quah 1989),
or by an external instrument (proxy-SVAR; Stock and Watson 2012, Mertens and Ravn 2013, Gertler and Karadi 2015).

The reduced-form VAR is estimated by least squares. Each identification scheme turns the reduced-form residual
covariance into an impact matrix B (u_t = B e_t), and the impulse responses are Psi_h B, where Psi_h are the moving-average
coefficients of the VAR. Confidence bands come from a wild (Rademacher) residual bootstrap, which is valid under
heteroskedasticity and, for the proxy-SVAR, resamples the instrument with the same signs as the residuals.
"""
import numpy as np
import pandas as pd

from .core import OESError, as_list, chart, distinct, f, int_option, need, result, stat, table, test_row, tests_table
from .timeseries import _series_frame

IDENT = {"cholesky": "Recursive (Cholesky)", "longrun": "Long-run restrictions (Blanchard-Quah)", "proxy": "External instrument (proxy-SVAR)"}


def _lags_matrix(Y, p):
    T, n = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - j:T - j] for j in range(1, p + 1)])
    return Y[p:], X


def _fit(Y, p):
    Yt, X = _lags_matrix(Y, p)
    Bc = np.linalg.lstsq(X, Yt, rcond=None)[0]
    U = Yt - X @ Bc
    n = Y.shape[1]
    A = [Bc[1 + j * n:1 + (j + 1) * n, :].T for j in range(p)]
    return Bc, A, U


def _ma(A, H):
    n = A[0].shape[0]
    Psi = [np.eye(n)]
    for h in range(1, H + 1):
        Psi.append(sum(Psi[h - j] @ A[j - 1] for j in range(1, min(h, len(A)) + 1)))
    return np.array(Psi)


def _impact(method, A, U, Sigma, s, Z, size):
    """Impact matrix (n x n) for the recursive and long-run schemes; the n-vector of the shock's impact for the proxy."""
    n = Sigma.shape[0]
    if method == "cholesky":
        return np.linalg.cholesky(Sigma)
    if method == "longrun":
        Ainf = np.eye(n) - sum(A)
        C = np.linalg.inv(Ainf)
        L = np.linalg.cholesky(C @ Sigma @ C.T)
        return Ainf @ L
    ok = ~np.isnan(Z)
    z = Z[ok] - Z[ok].mean()
    u = U[ok] - U[ok].mean(axis=0)
    cov = z @ u / len(z)
    b = cov / cov[s]
    return b * size


def _irf(method, A, U, Sigma, s, Z, size, H):
    Psi = _ma(A, H)
    B = _impact(method, A, U, Sigma, s, Z, size)
    if method == "proxy":
        return Psi @ B, B, Psi
    return (Psi @ B[:, s]), B, Psi


def _simulate(Y, Bc, p, E):
    T, n = Y.shape
    Ys = np.zeros_like(Y)
    Ys[:p] = Y[:p]
    for t in range(p, T):
        x = np.concatenate([[1.0]] + [Ys[t - j] for j in range(1, p + 1)])
        Ys[t] = x @ Bc + E[t - p]
    return Ys


def svar(df, spec):
    from statsmodels.tsa.api import VAR
    names = as_list(need(spec, "vars", "variables (at least two, in the order of the identification)"))
    if len(names) < 2:
        raise OESError("A structural VAR needs at least two variables.")
    method = spec.get("identification", "cholesky") or "cholesky"
    if method not in IDENT:
        raise OESError("Identification must be cholesky, longrun or proxy.")
    inst = as_list(spec.get("instrument"))
    shock = spec.get("shock") or names[0]
    if shock not in names:
        raise OESError("The shock variable must be one of the VAR variables.")
    if method == "proxy" and len(inst) != 1:
        raise OESError("The proxy-SVAR needs exactly one instrument column.")
    distinct(names, inst)
    d, labels = _series_frame(df, spec, names + (inst if method == "proxy" else []))
    data = d[names].astype(float)
    Zfull = d[inst[0]].astype(float).to_numpy() if method == "proxy" else None
    if spec.get("transform", "levels") == "diff":
        data = data.diff()
        keep = data.notna().all(axis=1).to_numpy()
        data = data[keep]
        Zfull = Zfull[keep] if Zfull is not None else None
        labels = [l for l, k in zip(labels, keep) if k]
    n = len(names)
    s = names.index(shock)
    maxl = int_option(spec, "max_lags", 8, 1, 24, "Maximum lag")
    ic = spec.get("ic", "aic")
    H = int_option(spec, "horizon", 16, 4, 48, "Horizon")
    B = int_option(spec, "boot", 200, 50, 1000, "Bootstrap replications")
    level = float(spec.get("level", 95) or 95)
    if level not in (68.0, 90.0, 95.0):
        raise OESError("The band level must be 68, 90 or 95.")
    size = float(spec.get("size", 1) or 1)
    maxl = min(maxl, max(1, (len(data) - 10) // (n + 1)))
    so = VAR(data.reset_index(drop=True)).select_order(maxl)
    p = spec.get("lags")
    p = int_option(spec, "lags", 2, 1, 24, "VAR lag order") if p not in (None, "", "auto") else max(1, int(getattr(so, ic)))
    Y = data.to_numpy(float)
    T = len(Y)
    if T < 40 + p * n:
        raise OESError("Too few observations for the chosen number of variables and lags.")
    Bc, A, U = _fit(Y, p)
    k = 1 + n * p
    Sigma = U.T @ U / (len(U) - k)
    Zr = Zfull[p:] if Zfull is not None else None
    if method == "proxy":
        if np.isnan(Zr).all() or (~np.isnan(Zr)).sum() < 30:
            raise OESError("The instrument needs at least 30 non-missing observations in the sample.")
    if method == "longrun":
        roots = np.abs(np.linalg.eigvals(_companion(A)))
        if roots.max() >= 0.9999:
            raise OESError("Long-run identification needs a stationary VAR (every root inside the unit circle). Use first differences or growth rates.")
    irf, Bimp, Psi = _irf(method, A, U, Sigma, s, Zr, size, H)
    # wild bootstrap
    rng = np.random.default_rng(int(spec.get("seed", 12345) or 12345))
    draws = np.empty((B, H + 1, n))
    good = 0
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=len(U))
        Ys = _simulate(Y, Bc, p, U * w[:, None])
        try:
            Bc_b, A_b, U_b = _fit(Ys, p)
            S_b = U_b.T @ U_b / (len(U_b) - k)
            Zb = Zr * w if Zr is not None else None
            r, _, _ = _irf(method, A_b, U_b, S_b, s, Zb, size, H)
        except np.linalg.LinAlgError:
            continue
        draws[good] = r
        good += 1
    if good < 20:
        raise OESError("The bootstrap failed too often. Reduce the number of lags or change the identification.")
    draws = draws[:good]
    lo_q, hi_q = (100 - level) / 2, 100 - (100 - level) / 2
    lo, hi = np.percentile(draws, lo_q, axis=0), np.percentile(draws, hi_q, axis=0)
    x = list(range(H + 1))
    charts = [chart(f"Response of {nm} to a {shock} shock", x, [("response", irf[:, i])], "horizon", nm, band=(lo[:, i], hi[:, i])) for i, nm in enumerate(names)]
    rows = [[nm, h, f(irf[h, i]), f(lo[h, i]), f(hi[h, i])] for i, nm in enumerate(names) for h in range(H + 1)]
    tabs = [table(f"Impulse responses to the {shock} shock", ["Variable", "Horizon", "Response", f"{level:g}% band low", f"{level:g}% band high"], rows,
                  f"{IDENT[method]}. Wild residual bootstrap with {good} replications; percentile bands.")]
    # impact matrix
    if method == "proxy":
        tabs.append(table("Impact of the shock on each variable (period 0)", ["Variable", "Impact"], [[nm, f(Bimp[i])] for i, nm in enumerate(names)],
                          f"The shock is scaled to raise {shock} by {f(size, 4)} on impact."))
    else:
        tabs.append(table("Structural impact matrix B (u = B e)", ["Variable"] + [f"shock {nm}" for nm in names],
                          [[nm] + [f(Bimp[i, j]) for j in range(n)] for i, nm in enumerate(names)],
                          "Column j is the contemporaneous effect of structural shock j on each variable."))
    if method == "longrun":
        Cm = np.linalg.inv(np.eye(n) - sum(A)) @ Bimp
        tabs.append(table("Long-run impact matrix", ["Variable"] + [f"shock {nm}" for nm in names],
                          [[nm] + [f(Cm[i, j]) for j in range(n)] for i, nm in enumerate(names)],
                          "Zeros above the diagonal are the identifying restrictions: shock j has no long-run effect on variables ordered before it."))
    # variance decomposition (recursive and long-run schemes)
    if method != "proxy":
        hs = sorted({1, min(4, H), min(8, H), H})
        Pb = np.einsum("hij,jk->hik", Psi, Bimp)
        tot = np.cumsum((Pb ** 2).sum(axis=2), axis=0)
        own = np.cumsum(Pb[:, :, s] ** 2, axis=0)
        tabs.append(table(f"Share of forecast-error variance explained by the {shock} shock (%)", ["Variable"] + [f"h = {h}" for h in hs],
                          [[nm] + [f(100 * own[h - 1, i] / tot[h - 1, i], 2) for h in hs] for i, nm in enumerate(names)],
                          "Computed from the point estimates."))
    sel_rows = [[lag] + [f(so.ics[c][lag], 4) for c in ("aic", "bic", "hqic", "fpe")] for lag in range(len(so.ics["aic"]))]
    tabs.append(table("Lag-order selection", ["Lags", "AIC", "BIC", "HQIC", "FPE"], sel_rows, f"Used: {p} lag(s)" + (" (chosen by " + ic.upper() + ")." if spec.get("lags") in (None, "", "auto") else ".")))
    diag = []
    roots = np.abs(np.linalg.eigvals(_companion(A)))
    diag.append(["Stability: largest root modulus", f(roots.max(), 4), None, "stable if below 1 (a level VAR with unit roots is near 1)"])
    summary = [stat("Observations", len(U)), stat("Lags", p), stat("Variables", n), stat("Identification", IDENT[method]), stat("Shock", shock),
               stat("Bootstrap draws", good)]
    notes = [f"Identification: {IDENT[method]}."]
    if method == "cholesky":
        notes.append("The variable order is the identifying assumption: a variable ordered earlier does not respond to later shocks within the period. Reorder the variables to check robustness.")
    elif method == "longrun":
        notes.append("Shock j has no permanent effect on the variables ordered before j. In the classic application, output growth is ordered first: the first shock is the supply shock, the second a demand shock with only transitory effects on output.")
    else:
        Zv = Zr[~np.isnan(Zr)]
        uv = U[~np.isnan(Zr), s]
        zc, uc = Zv - Zv.mean(), uv - uv.mean()
        slope = (zc @ uc) / (zc @ zc)
        e = uc - slope * zc
        se = np.sqrt((zc ** 2 * e ** 2).sum()) / (zc @ zc)
        Fr = (slope / se) ** 2
        r2 = (zc @ uc) ** 2 / ((zc @ zc) * (uc @ uc))
        diag.append(["First stage: robust F", f(Fr, 3), None, "instrument relevance; below 10 signals a weak instrument"])
        diag.append(["First stage: R-squared", f(r2, 4), None, f"{len(Zv)} observations with a non-missing instrument"])
        summary.append(stat("First-stage F", Fr, 2))
        notes.append("The instrument must be correlated with the shock (relevance, tested above) and uncorrelated with the other structural shocks (exogeneity, which cannot be tested). The responses are scaled to a shock that raises " + shock + f" by {f(size, 4)} on impact.")
        if Fr < 10:
            notes.append("Warning: the first-stage F is below 10, so the instrument is weak and the bands are unreliable.")
    tabs.append(tests_table(diag, "Instrument and stability" if method == "proxy" else "Stability"))
    if spec.get("transform", "levels") == "diff":
        notes.append("The variables were first-differenced; responses are of the changes. Cumulate them for the level.")
    return result("svar", f"Structural VAR ({IDENT[method]})", len(U), summary, tabs, charts, notes)


def _companion(A):
    n, p = A[0].shape[0], len(A)
    top = np.hstack(A)
    if p == 1:
        return top
    return np.vstack([top, np.hstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))])])
