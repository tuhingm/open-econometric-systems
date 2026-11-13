"""Multiple structural breaks (Bai and Perron 1998, 2003): breaks in the mean or in a regression, at unknown dates.

For a trimming fraction eps every regime has at least h = eps * T observations. The global least-squares partition for
m = 1, ..., M breaks is found by dynamic programming over the table of segment sums of squared residuals (Bai and Perron
2003, section 3). The number of breaks is chosen by the sup-F(l) and UDmax tests, the sequential sup-F(l+1 | l) test, or an
information criterion (BIC or Liu-Wu-Zidek). Critical values were simulated for this engine (see _bp_cv.py); they are
asymptotic, for the pure structural change model in which all coefficients change.
"""
import numpy as np
import pandas as pd

from . import _bp_cv
from .core import OESError, as_list, chart, distinct, f, int_option, need, result, stat, table, tests_table
from .timeseries import _series_frame

TRIMS = {0.10: 8, 0.15: 5, 0.20: 3, 0.25: 2}


def ssr_table(y, X, h):
    """S[i, j] = SSR of the regression of y on X over observations i..j (inclusive); inf where the segment is shorter than h."""
    T, q = X.shape
    S = np.full((T, T), np.inf)
    Cxx = np.concatenate([np.zeros((1, q, q)), np.cumsum(X[:, :, None] * X[:, None, :], axis=0)])
    Cxy = np.concatenate([np.zeros((1, q)), np.cumsum(X * y[:, None], axis=0)])
    Cyy = np.concatenate([[0.0], np.cumsum(y * y)])
    for i in range(T - h + 1):
        j = np.arange(i + h - 1, T)
        Sxx = Cxx[j + 1] - Cxx[i]
        Sxy = Cxy[j + 1] - Cxy[i]
        try:
            beta = np.linalg.solve(Sxx, Sxy[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            beta = np.einsum("jab,jb->ja", np.linalg.pinv(Sxx), Sxy)
        S[i, j] = np.maximum(Cyy[j + 1] - Cyy[i] - (beta * Sxy).sum(axis=1), 0.0)
    return S


def partitions(S, T, h, M):
    """Optimal break sets for m = 0..M: returns (ssr list, breaks list). A break b means a new regime starts at observation b."""
    opt = np.full((M + 1, T + 1), np.inf)
    arg = np.zeros((M + 1, T + 1), dtype=int)
    for t in range(h, T + 1):
        opt[0, t] = S[0, t - 1]
    for m in range(1, M + 1):
        for t in range((m + 1) * h, T + 1):
            s = np.arange(m * h, t - h + 1)
            cand = opt[m - 1, s] + S[s, t - 1]
            k = int(np.argmin(cand))
            opt[m, t], arg[m, t] = cand[k], s[k]
    ssr, brk = [opt[0, T]], [[]]
    for m in range(1, M + 1):
        if not np.isfinite(opt[m, T]):
            break
        b, t = [], T
        for mm in range(m, 0, -1):
            t = arg[mm, t]
            b.append(int(t))
        ssr.append(float(opt[m, T]))
        brk.append(sorted(b))
    return ssr, brk


def best_extra_break(S, brk, T, h):
    """Smallest SSR obtainable by adding one break inside one of the segments defined by brk; returns (reduction, new break)."""
    edges = [0] + list(brk) + [T]
    best, where = 0.0, None
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2 * h:
            continue
        tau = np.arange(a + h, b - h + 1)
        gain = S[a, b - 1] - (S[a, tau - 1] + S[tau, b - 1])
        k = int(np.argmax(gain))
        if gain[k] > best:
            best, where = float(gain[k]), int(tau[k])
    return best, where


def sup_f(ssr, T, q, M):
    return [((T - (l + 1) * q) / (l * q)) * (ssr[0] - ssr[l]) / ssr[l] for l in range(1, M + 1)]


def _critical(kind, q, eps, l=None):
    tab = _bp_cv.CV[(min(q, _bp_cv.QMAX), eps)]
    if kind == "supf":
        return tab["supf"][l - 1]
    if kind == "udmax":
        return tab["udmax"]
    return tab["seq"][l]


def breaks(df, spec):
    y = need(spec, "y", "dependent variable")
    xs = as_list(spec.get("x"))
    distinct([y], xs)
    d, labels = _series_frame(df, spec, [y] + xs)
    T = len(d)
    q = 1 + len(xs)
    if q > _bp_cv.QMAX:
        raise OESError(f"Break tests are tabulated for up to {_bp_cv.QMAX - 1} changing regressors plus the constant.")
    eps = float(spec.get("trim", 0.15) or 0.15)
    if eps not in TRIMS:
        raise OESError("The trimming fraction must be 0.10, 0.15, 0.20 or 0.25.")
    M = min(int_option(spec, "max_breaks", TRIMS[eps], 1, TRIMS[eps], f"Maximum number of breaks (at most {TRIMS[eps]} at this trimming)"), TRIMS[eps])
    sel = spec.get("select", "bic") or "bic"
    if sel not in ("bic", "lwz", "sequential", "udmax"):
        raise OESError("Selection must be bic, lwz, sequential or udmax.")
    if T < 40:
        raise OESError("Break tests need at least 40 observations.")
    if T > 800:
        raise OESError("This tool handles up to 800 observations; aggregate the data first.")
    h = max(int(np.floor(eps * T)), q + 1)
    M = min(M, T // h - 1)
    if M < 1:
        raise OESError("The sample is too short for this trimming fraction and number of regressors.")
    yv = d[y].to_numpy(float)
    X = np.column_stack([np.ones(T)] + [d[c].to_numpy(float) for c in xs])
    S = ssr_table(yv, X, h)
    ssr, brk = partitions(S, T, h, M)
    M = len(ssr) - 1
    F = sup_f(ssr, T, q, M)
    # sequential tests
    seq = []
    for l in range(M):
        gain, where = best_extra_break(S, brk[l], T, h)
        new = ssr[l] - gain
        seq.append((gain / (new / (T - (l + 2) * q)), where))
    # information criteria
    bic, lwz = [], []
    for m in range(M + 1):
        pstar = (m + 1) * q + m
        bic.append(np.log(ssr[m] / T) + pstar * np.log(T) / T)
        lwz.append(np.log(ssr[m] / (T - pstar)) + pstar * 0.299 * np.log(T) ** 2.1 / T)
    rows = []
    for l in range(1, M + 1):
        c5 = _critical("supf", q, eps, l)
        rows.append([f"supF({l}) : 0 versus {l}", f(F[l - 1], 3), f(c5, 3), "Reject" if F[l - 1] > c5 else "Do not reject"])
    ud = max(F)
    cud = _critical("udmax", q, eps)
    rows.append(["UDmax: 0 versus an unknown number (1 to " + str(M) + ")", f(ud, 3), f(cud, 3), "Reject" if ud > cud else "Do not reject"])
    tabs = []
    tabs.append(table("Tests of no break against l breaks", ["Test", "Statistic", "5% critical value", "Decision"], rows,
                      "Critical values are simulated for this engine (asymptotic, all coefficients change, trimming " + f"{eps:g}" + ")."))
    rows = []
    n_seq = 0
    for l in range(M):
        c5 = _critical("seq", q, eps, l)
        rej = seq[l][0] > c5
        rows.append([f"supF({l + 1} | {l}) : {l} versus {l + 1}", f(seq[l][0], 3), f(c5, 3), "Reject" if rej else "Do not reject"])
    tabs.append(table("Sequential tests: l breaks against l + 1", ["Test", "Statistic", "5% critical value", "Decision"], rows,
                      "Start at l = 0 and stop at the first non-rejection; that l is the sequential estimate."))
    seq_n = 0
    for l in range(M):
        if seq[l][0] > _critical("seq", q, eps, l):
            seq_n = l + 1
        else:
            break
    n_bic, n_lwz = int(np.argmin(bic)), int(np.argmin(lwz))
    n_ud = (int(np.argmax(F)) + 1) if ud > cud else 0
    # udmax: choose the number of breaks by the sequential procedure when rejecting; otherwise 0
    chosen = {"bic": n_bic, "lwz": n_lwz, "sequential": seq_n, "udmax": max(n_ud, 0) if ud > cud else 0}[sel]
    if sel == "udmax" and ud > cud:
        chosen = seq_n if seq_n >= 1 else 1
    tabs.append(table("Information criteria by number of breaks", ["Breaks", "SSR", "BIC", "LWZ"],
                      [[m, f(ssr[m], 4), f(bic[m], 4), f(lwz[m], 4)] for m in range(M + 1)],
                      f"BIC chooses {n_bic}, LWZ chooses {n_lwz}, the sequential test chooses {seq_n}. Shown below: {chosen} break(s), chosen by "
                      + {"bic": "BIC", "lwz": "LWZ", "sequential": "the sequential test", "udmax": "UDmax then the sequential test"}[sel] + "."))
    # selected model
    bs = brk[chosen]
    edges = [0] + bs + [T]
    names = ["const"] + xs
    reg_rows, betas, sigma2 = [], [], ssr[chosen] / (T - (chosen + 1) * q)
    fitted = np.empty(T)
    from statsmodels.stats.sandwich_covariance import cov_hac
    for r, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        Xa, ya = X[a:b], yv[a:b]
        beta = np.linalg.lstsq(Xa, ya, rcond=None)[0]
        e = ya - Xa @ beta
        fitted[a:b] = Xa @ beta
        nl = max(int(np.floor(4 * (len(ya) / 100) ** (2 / 9))), 1)
        try:
            XtXi = np.linalg.inv(Xa.T @ Xa)
            V = XtXi @ (Xa * e[:, None]).T @ (Xa * e[:, None]) @ XtXi
            for L in range(1, nl + 1):
                w = 1 - L / (nl + 1)
                G = (Xa[L:] * e[L:, None]).T @ (Xa[:-L] * e[:-L, None])
                V = V + w * XtXi @ (G + G.T) @ XtXi
            se = np.sqrt(np.maximum(np.diag(V), 0))
        except np.linalg.LinAlgError:
            se = np.full(q, np.nan)
        betas.append(beta)
        reg_rows.append([r + 1, labels[a], labels[b - 1], b - a] + [f"{f(beta[k], 4)} ({f(se[k], 4)})" for k in range(q)])
    tabs.append(table(f"Regimes of the model with {chosen} break(s)", ["Regime", "From", "To", "Observations"] + names, reg_rows,
                      "Coefficients with Newey-West standard errors in parentheses, estimated separately in each regime."))
    # break dates with confidence intervals (Bai 1997: shrinking-shift limit, homoskedastic version)
    if chosen:
        crit = {90: _bp_cv.ARGMAX[0.90], 95: _bp_cv.ARGMAX[0.95]}
        rows = []
        for i, bd in enumerate(bs):
            a, b = edges[i], edges[i + 2]
            Q = X[a:b].T @ X[a:b] / (b - a)
            dl = betas[i + 1] - betas[i]
            res = np.concatenate([yv[a:bd] - X[a:bd] @ betas[i], yv[bd:b] - X[bd:b] @ betas[i + 1]])
            w = X[a:b] * res[:, None]
            nl = max(int(np.floor(4 * ((b - a) / 100) ** (2 / 9))), 1)
            Om = w.T @ w / (b - a)
            for L in range(1, nl + 1):
                G = w[L:].T @ w[:-L] / (b - a)
                Om = Om + (1 - L / (nl + 1)) * (G + G.T)
            num = float(dl @ Q @ dl) ** 2
            den = float(dl @ Om @ dl)
            scale = num / den if den > 0 else 0.0
            hw95 = crit[95] / scale if scale > 0 else np.nan
            hw90 = crit[90] / scale if scale > 0 else np.nan
            lo, hi = int(max(np.floor(bd - hw95), 0)), int(min(np.ceil(bd + hw95), T - 1))
            rows.append([i + 1, labels[bd], f"{labels[lo]} to {labels[hi]}", f(hw90, 1), f(hw95, 1)])
        tabs.append(table("Estimated break dates", ["Break", "First observation of the new regime", "95% confidence interval", "90% half-width (obs.)", "95% half-width (obs.)"], rows,
                          "Intervals follow Bai (1997) and Bai and Perron (2003): the limiting distribution of the break date under a shrinking shift, with a Newey-West long-run variance so that serially correlated errors widen the interval. They are wide when the shift is small relative to the noise."))
    x = list(labels)
    charts = [chart(f"{y} and the fitted regimes", x, [(y, yv), ("fitted", fitted)], "", y),
              chart("Information criteria by number of breaks", list(range(M + 1)), [("BIC", bic), ("LWZ", lwz)], "breaks", "criterion")]
    summary = [stat("Observations", T), stat("Regressors changing (incl. constant)", q), stat("Trimming", eps, 2), stat("Minimum regime length", h),
               stat("Maximum breaks", M), stat("Breaks selected", chosen), stat("BIC chooses", n_bic), stat("Sequential chooses", seq_n)]
    notes = ["All coefficients are allowed to change at every break (pure structural change). Regimes must contain at least " + str(h) + " observations.",
             "The break dates are the global least-squares minimisers, found by dynamic programming. Different selection rules can give different numbers of breaks; BIC tends to be cautious, and the sequential test is the most common in applied work.",
             "The tests assume serially uncorrelated errors with a common variance across regimes. If the errors are autocorrelated, add lags of the dependent variable as regressors or interpret the tests with care."]
    return result("bp", f"Structural breaks: {y}", T, summary, tabs, charts, notes)
