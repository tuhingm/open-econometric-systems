"""Threshold regression for time series (Tong's TAR and SETAR; Hansen 1996, 1997, 2000): the coefficients change abruptly when a
threshold variable crosses an estimated threshold.

y_t = x_t' b1 * 1(s_t <= g) + x_t' b2 * 1(s_t > g) + e_t,   s_t = y_{t-d} (SETAR) or another variable lagged d periods.

With observations sorted by the threshold variable, every regime is a contiguous block, so the sum of squared residuals for all
candidate thresholds comes from cumulative moments. The threshold is the minimiser. The null of linearity is tested with Hansen's
(1996) sup-F statistic and a fixed-regressor bootstrap, because the threshold is not identified under the null. The confidence
interval for the threshold is the no-rejection region of the likelihood-ratio statistic (Hansen 2000); a second threshold is
searched with the first held fixed and then the first is refined (Bai 1997).
"""
import numpy as np
import pandas as pd

from .core import (OESError, as_list, chart, check_columns, f, int_option, need, numeric, result, stat, table, time_order)

C95 = -2 * np.log(1 - np.sqrt(0.95))      # 7.352: 95% critical value of the LR statistic for the threshold (Hansen 2000)
C90 = -2 * np.log(1 - np.sqrt(0.90))


def _cum(y, X):
    Cxx = np.concatenate([np.zeros((1,) + (X.shape[1],) * 2), np.cumsum(X[:, :, None] * X[:, None, :], axis=0)])
    Cxy = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X * y[:, None], axis=0)])
    Cyy = np.concatenate([[0.0], np.cumsum(y * y)])
    return Cxx, Cxy, Cyy


def _seg(C, i, j):
    """SSR of the regression over sorted observations i..j-1 (half-open), for arrays of i and j."""
    Cxx, Cxy, Cyy = C
    i, j = np.broadcast_arrays(np.asarray(i), np.asarray(j))
    Sxx, Sxy = Cxx[j] - Cxx[i], Cxy[j] - Cxy[i]
    try:
        beta = np.linalg.solve(Sxx, Sxy[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.einsum("...ab,...b->...a", np.linalg.pinv(Sxx), Sxy)
    return np.maximum(Cyy[j] - Cyy[i] - (beta * Sxy).sum(axis=-1), 0.0)


def _valid_splits(sq, h):
    """Sorted positions k (first k observations in the lower regime) that keep h observations per regime and split distinct values."""
    T = len(sq)
    k = np.arange(h, T - h + 1)
    return k[sq[k - 1] < sq[k]]


def _scan(y, X, order, sq, ks):
    C = _cum(y[order], X[order])
    T = len(y)
    return _seg(C, 0, ks) + _seg(C, ks, T), C


def _second(C, sq, k1, h, T):
    """Best additional split of the three-regime model with the split at k1 fixed: returns (S, k2, all candidates)."""
    left = np.arange(h, k1 - h + 1)
    left = left[sq[left - 1] < sq[left]]
    right = np.arange(k1 + h, T - h + 1)
    right = right[sq[right - 1] < sq[right]]
    out = []
    if len(left):
        out.append((left, _seg(C, 0, left) + _seg(C, left, k1) + _seg(C, k1, T)))
    if len(right):
        out.append((right, _seg(C, 0, k1) + _seg(C, k1, right) + _seg(C, right, T)))
    if not out:
        return np.inf, None, np.array([], int), np.array([])
    ks = np.concatenate([o[0] for o in out])
    S = np.concatenate([o[1] for o in out])
    m = int(np.argmin(S))
    return float(S[m]), int(ks[m]), ks, S


def _hac(Z, e, lags=None):
    T = len(e)
    if lags is None:
        lags = max(int(np.floor(4 * (T / 100) ** (2 / 9))), 1)
    ZtZi = np.linalg.pinv(Z.T @ Z)
    sc = Z * e[:, None]
    S = sc.T @ sc
    for L in range(1, lags + 1):
        G = sc[L:].T @ sc[:-L]
        S = S + (1 - L / (lags + 1)) * (G + G.T)
    return ZtZi @ S @ ZtZi


def tar(df, spec):
    y = need(spec, "y", "dependent variable")
    xs = as_list(spec.get("x"))
    p = int_option(spec, "ar_lags", 1, 0, 12, "Lags of the dependent variable")
    qv = spec.get("threshold") or None
    if y in xs or qv == y:
        qv = None if qv == y else qv
        if y in xs:
            raise OESError("The dependent variable cannot also be a regressor; use the lags option for its lags.")
    if p == 0 and not xs:
        raise OESError("Add lags of the dependent variable or other regressors.")
    d0 = int_option(spec, "delay", 1, 0 if qv else 1, 12, "Delay")
    search = spec.get("search_delay") in (True, "yes", "true", "1")
    dmax = int_option(spec, "max_delay", 4, 1, 12, "Largest delay") if search else d0
    delays = list(range(1 if not qv else 0, dmax + 1)) if search else [d0]
    trim = float(spec.get("trim", 0.15) or 0.15)
    if not 0.05 <= trim <= 0.4:
        raise OESError("Trimming must be between 0.05 and 0.40.")
    B = int_option(spec, "boot", 199, 0, 2000, "Bootstrap replications")
    nthr = int_option(spec, "max_thresholds", 2, 1, 2, "Maximum thresholds")
    seed = int_option(spec, "seed", 12345, 0, 10**9, "Seed")
    cols = list(dict.fromkeys([y] + xs + ([qv] if qv else [])))
    dframe = df
    tcol = spec.get("time")
    if tcol:
        check_columns(df, [tcol])
        dframe = time_order(df, tcol)
    d = numeric(dframe, cols, keep=[tcol] if tcol else [])
    labels = d[tcol].astype(str).tolist() if tcol else [str(i + 1) for i in range(len(d))]
    yy = d[y].to_numpy(float)
    n = len(yy)
    start = max(p, max(delays))
    if n - start < 30:
        raise OESError("Too few observations for a threshold model after taking lags.")
    rows = np.arange(start, n)
    T = len(rows)
    names = ["const"] + [f"{y}(-{j})" for j in range(1, p + 1)] + xs
    X = np.column_stack([np.ones(T)] + [yy[rows - j] for j in range(1, p + 1)] + [d[c].to_numpy(float)[rows] for c in xs])
    Y = yy[rows]
    qsrc = d[qv].to_numpy(float) if qv else yy
    qname = qv or y
    Q = {dl: qsrc[rows - dl] for dl in delays}
    lab = [labels[i] for i in rows]
    q = X.shape[1]
    h = max(int(np.ceil(trim * T)), q + 2)
    if T < 2 * h + 2:
        raise OESError("Too few observations for this trimming; lower the trimming or the number of regressors.")
    orders = {dl: np.argsort(Q[dl], kind="mergesort") for dl in delays}
    sqs = {dl: Q[dl][orders[dl]] for dl in delays}
    kss = {dl: _valid_splits(sqs[dl], h) for dl in delays}
    for dl in delays:
        if len(kss[dl]) == 0:
            raise OESError("The threshold variable has too few distinct values for this trimming.")
    # linear fit
    b0 = np.linalg.lstsq(X, Y, rcond=None)[0]
    e0 = Y - X @ b0
    S0 = float(e0 @ e0)

    def best(yv):
        res = {}
        for dl in delays:
            S, _ = _scan(yv, X, orders[dl], sqs[dl], kss[dl])
            m = int(np.argmin(S))
            res[dl] = (float(S[m]), int(kss[dl][m]), S)
        dl = min(res, key=lambda k: res[k][0])
        return dl, res

    dl, res = best(Y)
    S1, k1, Sgrid = res[dl]
    s2_1 = S1 / (T - 2 * q)
    F1 = (S0 - S1) / s2_1
    order, sq = orders[dl], sqs[dl]
    g1 = float(sq[k1 - 1])
    tests = []
    rng = np.random.default_rng(seed)
    p1 = None
    cv1 = {}
    if B:
        draws = rng.standard_normal((B, T))
        Fb = np.empty(B)
        for b in range(B):
            yb = e0 * draws[b]
            ub = yb - X @ np.linalg.lstsq(X, yb, rcond=None)[0]
            S0b = float(ub @ ub)
            _, rb = best(yb)
            S1b = min(v[0] for v in rb.values())
            Fb[b] = (S0b - S1b) / (S1b / (T - 2 * q))
        p1 = float((1 + (Fb >= F1).sum()) / (B + 1))
        cv1 = {10: float(np.quantile(Fb, 0.90)), 5: float(np.quantile(Fb, 0.95)), 1: float(np.quantile(Fb, 0.99))}
    tests.append(["Linearity against one threshold (sup-F)", f(F1, 3), f(p1, 4) if B else None, f(cv1.get(5), 3) if B else None,
                  f"{B} bootstrap draws" if B else "bootstrap off"])
    # second threshold
    nfound, ks_final = 1, [k1]
    p2 = F2 = None
    k2 = None
    C = _cum(Y[order], X[order])
    if nthr == 2 and T >= 3 * h:
        S2, k2, k2s, S2all = _second(C, sq, k1, h, T)
        if k2 is not None:
            s2_2 = S2 / (T - 3 * q)
            F2 = (S1 - S2) / s2_2
            if B:
                e1 = np.empty(T)
                Z1 = np.zeros((T, 2 * q))
                reg = (np.arange(T) >= k1)
                eo = np.empty(T)
                for j in (0, 1):
                    mk = reg == j
                    Xo = X[order][mk]
                    bj = np.linalg.lstsq(Xo, Y[order][mk], rcond=None)[0]
                    eo[mk] = Y[order][mk] - Xo @ bj
                e1[order] = eo
                F2b = np.empty(B)
                for b in range(B):
                    yb = e1 * draws[b]
                    Sb, Cb = _scan(yb, X, order, sq, kss[dl])
                    m = int(np.argmin(Sb))
                    kb = int(kss[dl][m])
                    Sb1 = float(Sb[m])
                    Cb = _cum(yb[order], X[order])
                    Sb2 = _second(Cb, sq, kb, h, T)[0]
                    F2b[b] = (Sb1 - Sb2) / (Sb2 / (T - 3 * q)) if np.isfinite(Sb2) else 0.0
                p2 = float((1 + (F2b >= F2).sum()) / (B + 1))
            tests.append(["One threshold against two (sup-F)", f(F2, 3), f(p2, 4) if B else None,
                          f(float(np.quantile(F2b, 0.95)), 3) if B else None, f"first threshold re-estimated in each of {B} draws" if B else "bootstrap off"])
            if B and p2 < 0.05 or (not B and F2 > 12):
                nfound = 2
    # choose the model shown: two thresholds only when the second is supported
    if nfound == 2:
        ka, kb_ = sorted([k1, k2])
        # refinement: re-search the first split given the second (Bai 1997)
        ks_final = [ka, kb_]
        for it in range(2):
            a, b_ = ks_final
            ca = np.arange(h, b_ - h + 1)
            ca = ca[sq[ca - 1] < sq[ca]]
            Sa = _seg(C, 0, ca) + _seg(C, ca, b_) + _seg(C, b_, T)
            a = int(ca[int(np.argmin(Sa))])
            cb = np.arange(a + h, T - h + 1)
            cb = cb[sq[cb - 1] < sq[cb]]
            Sb_ = _seg(C, 0, a) + _seg(C, a, cb) + _seg(C, cb, T)
            b_ = int(cb[int(np.argmin(Sb_))])
            ks_final = [a, b_]
        a, b_ = ks_final
        S_final = float(_seg(C, 0, a) + _seg(C, a, b_) + _seg(C, b_, T))
    else:
        S_final = S1
    R = len(ks_final) + 1
    gs = [float(sq[k - 1]) for k in ks_final]
    # confidence intervals from the likelihood-ratio statistic, one threshold at a time with the others fixed
    s2 = S_final / (T - R * q)
    ci_rows, LRs = [], []
    for i, kk in enumerate(ks_final):
        other = [k for j, k in enumerate(ks_final) if j != i]
        lo_b = max([k for k in other if k < kk], default=0)
        hi_b = min([k for k in other if k > kk], default=T)
        cand = np.arange(lo_b + h, hi_b - h + 1)
        cand = cand[sq[cand - 1] < sq[cand]]
        if len(cand) == 0:
            cand = np.array([kk])
        Sc = np.empty(len(cand))
        for t_, c_ in enumerate(cand):
            ks_ = sorted(other + [int(c_)])
            e_ = [0] + ks_ + [T]
            Sc[t_] = sum(float(_seg(C, e_[m], e_[m + 1])) for m in range(len(e_) - 1))
        LR = (Sc - S_final) / s2
        inside = cand[LR <= C95]
        lo_ci, hi_ci = float(sq[inside.min() - 1]), float(sq[inside.max() - 1])
        LRs.append((cand, LR))
        ci_rows.append([i + 1, f(gs[i], 4), f"{f(lo_ci, 4)} to {f(hi_ci, 4)}", int(kk), T - int(kk) if R == 2 else None])
    # regimes, coefficients and standard errors
    edges = [0] + ks_final + [T]
    regime_of_sorted = np.zeros(T, int)
    for r in range(R):
        regime_of_sorted[edges[r]:edges[r + 1]] = r
    reg = np.empty(T, int)
    reg[order] = regime_of_sorted
    Z = np.hstack([X * (reg == r)[:, None] for r in range(R)])
    bz = np.linalg.lstsq(Z, Y, rcond=None)[0]
    ez = Y - Z @ bz
    V = _hac(Z, ez)
    se = np.sqrt(np.maximum(np.diag(V), 0))
    coef_rows = []
    for k_, nm in enumerate(names):
        row = [nm]
        for r in range(R):
            j = r * q + k_
            row.append(f"{f(bz[j], 4)} ({f(se[j], 4)})")
        coef_rows.append(row)
    rule = []
    for r in range(R):
        if R == 1:
            rule.append("")
        elif r == 0:
            rule.append(f"{qname}(-{dl}) <= {f(gs[0], 3)}")
        elif r == R - 1:
            rule.append(f"{qname}(-{dl}) > {f(gs[-1], 3)}")
        else:
            rule.append(f"{f(gs[r - 1], 3)} < {qname}(-{dl}) <= {f(gs[r], 3)}")
    reg_rows = []
    for r in range(R):
        nobs = int((reg == r).sum())
        ar = [bz[r * q + 1 + j] for j in range(p)]
        sa = float(np.sum(ar)) if p else None
        mean_ = bz[r * q] / (1 - sa) if (p and not xs and abs(sa) < 0.999) else None
        reg_rows.append([r + 1, rule[r], nobs, f(nobs / T, 3), f(sa, 4) if p else None, f(mean_, 4) if mean_ is not None else None])
    tabs = [table("Tests for a threshold", ["Test", "Statistic", "Bootstrap p-value", "5% critical value", "Detail"], tests,
                  "The threshold is not identified under the null of linearity, so the F statistic does not follow an F distribution. The p-value comes from Hansen's (1996) fixed-regressor bootstrap."
                  + (f" The delay is searched over {delays[0]} to {delays[-1]} in the data and in every bootstrap draw." if len(delays) > 1 else "")),
            table("Estimated thresholds", ["Threshold", "Estimate", "95% confidence interval", "Observations at or below", "Observations above"],
                  ci_rows, "The interval collects every candidate threshold whose likelihood-ratio statistic is below 7.35 (Hansen 2000). It is often wide and can be asymmetric."),
            table(f"Coefficients by regime ({R} regime{'s' if R > 1 else ''})", ["Coefficient"] + [f"Regime {r + 1}" for r in range(R)], coef_rows,
                  "Newey-West (HAC) standard errors in parentheses, from one regression with the regime-specific columns."),
            table("The regimes", ["Regime", "Rule", "Observations", "Share", "Sum of AR coefficients", "Implied mean"], reg_rows,
                  "A sum of AR coefficients near or above 1 in a regime means the series is persistent or explosive there; the implied mean is c / (1 - sum) when the regime is stationary.")]
    # compare with the linear fit
    aic_lin = T * np.log(S0 / T) + 2 * q
    aic_th = T * np.log(S_final / T) + 2 * q * R + 2 * (R - 1)
    tabs.append(table("Fit against the linear model", ["Model", "SSR", "Residual std. dev.", "AIC"],
                      [["Linear", f(S0, 4), f(np.sqrt(S0 / (T - q)), 4), f(aic_lin, 3)],
                       [f"Threshold ({R} regimes)", f(S_final, 4), f(np.sqrt(S_final / (T - R * q)), 4), f(aic_th, 3)]],
                      "AIC counts the thresholds as parameters. A lower value favours the threshold model."))
    cand, LR = LRs[0]
    charts = [chart(f"Likelihood-ratio statistic for the threshold in {qname}(-{dl})", [float(sq[c - 1]) for c in cand],
                    [("LR(threshold)", LR), ("5% critical value", np.full(len(cand), C95))], qname, "LR")]
    if R == 3:
        cand2, LR2 = LRs[1]
        charts.append(chart(f"Likelihood-ratio statistic for the second threshold in {qname}(-{dl})", [float(sq[c - 1]) for c in cand2],
                            [("LR(threshold)", LR2), ("5% critical value", np.full(len(cand2), C95))], qname, "LR"))
    charts.append(chart(f"{qname}(-{dl}) and the estimated thresholds", lab, [(f"{qname}(-{dl})", Q[dl])] + [(f"threshold {i + 1}", np.full(T, g)) for i, g in enumerate(gs)], "", qname))
    summary = [stat("Observations", T), stat("Regressors per regime", q), stat("Threshold variable", f"{qname}(-{dl})"), stat("Delay", dl),
               stat("Trimming", trim, 2), stat("Sup-F against linearity", F1, 3)]
    if B:
        summary.append(stat("Bootstrap p (linearity)", p1, 4))
    summary += [stat("Thresholds found", R - 1)] + [stat(f"Threshold {i + 1}", g, 4) for i, g in enumerate(gs)]
    notes = ["All coefficients, including the constant, change across regimes. Each regime must hold at least " + str(h) + " observations.",
             "The sup-F test and the bootstrap assume errors that are not strongly autocorrelated or heteroskedastic; the coefficient standard errors are robust (HAC). Add lags of the dependent variable if the residuals are serially correlated.",
             "Regimes that hold only a few observations give imprecise coefficients. Check the observation counts before interpreting a regime."]
    if p1 is not None and p1 > 0.10:
        notes.append("The bootstrap does not reject linearity at 10%. The threshold estimate is then not reliable; the linear model is the safer choice.")
    return result("tar", f"Threshold regression: {y}", T, summary, tabs, charts, notes)
