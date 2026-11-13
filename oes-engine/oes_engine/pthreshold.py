"""Panel threshold regression (Hansen 1999): slopes that differ across regimes defined by a threshold variable.

y_it = mu_i + b1' x_it * 1(q_it <= g) + b2' x_it * 1(q_it > g) + c' z_it + e_it

The unit effects are removed by the within transformation, the threshold g is found by grid search over the sum of squared
residuals, the null of no threshold is tested with a bootstrap F statistic (the threshold is not identified under the null,
so the usual F distribution does not apply), and the confidence interval for g is the no-rejection region of the likelihood-
ratio statistic. A second threshold is searched with g1 fixed at its estimate (Bai 1997, Hansen 1999 section 3.3).
"""
import numpy as np
import pandas as pd

from . import panel as _panel
from .core import OESError, as_list, chart, coef_table, f, int_option, need, result, stat, table, tests_table


def _demean(A, u):
    return _panel._demean(A, u)


def _design(Xx, Zc, q, gs):
    r = np.searchsorted(np.asarray(gs, float), q, side="left")
    cols = [Xx * (r == j)[:, None] for j in range(len(gs) + 1)]
    if Zc.shape[1]:
        cols.append(Zc)
    return np.hstack(cols), r


def _qr(A):
    Q, R = np.linalg.qr(A)
    return Q


def pthresh(df, spec):
    y = need(spec, "y", "dependent variable")
    xs = as_list(need(spec, "x", "regime-dependent regressors"))
    ctrl = as_list(spec.get("controls"))
    qv = need(spec, "threshold", "threshold variable")
    for a, b, msg in ((xs, ctrl, "A regressor cannot be both regime-dependent and a control."),):
        if set(a) & set(b):
            raise OESError(msg)
    if y in xs or y in ctrl or y == qv:
        raise OESError("The dependent variable cannot also be a regressor or the threshold variable.")
    unit = need(spec, "unit", "unit identifier")
    time = need(spec, "time", "time identifier")
    if qv in ctrl:
        raise OESError("The threshold variable can be a regime-dependent regressor but not a control.")
    cols = list(dict.fromkeys([y] + xs + ctrl + [qv]))
    d, unit, time, N, T = _panel._panel(df, spec, cols)
    trim = float(spec.get("trim", 0.05) or 0.05)
    if not 0.01 <= trim <= 0.25:
        raise OESError("The trimming fraction must be between 0.01 and 0.25.")
    G = int_option(spec, "grid", 100, 20, 200, "Number of grid points")
    B = int_option(spec, "boot", 300, 50, 1000, "Bootstrap replications")
    max_thr = int_option(spec, "max_thresholds", 2, 1, 2, "Maximum number of thresholds")
    n = len(d)
    if n > 20000:
        raise OESError("This tool handles up to 20,000 observations.")
    u = d["_u"].to_numpy()
    yv = d[y].to_numpy(float)
    Xx = d[xs].to_numpy(float)
    Zc = d[ctrl].to_numpy(float) if ctrl else np.empty((n, 0))
    q = d[qv].to_numpy(float)
    kx, kc = len(xs), len(ctrl)
    if n - N < 10 * (2 * kx + kc + 1):
        raise OESError("Too few observations per unit for a threshold regression with this many regressors.")
    # candidate thresholds: quantiles of q between trim and 1 - trim
    cand = np.unique(np.quantile(q, np.linspace(trim, 1 - trim, G), method="lower"))
    cand = np.array([g for g in cand if (q <= g).sum() >= trim * n and (q > g).sum() >= trim * n])
    if len(cand) < 5:
        raise OESError("The threshold variable has too few distinct values for the chosen trimming fraction.")
    yd = _demean(yv[:, None], u)[:, 0]
    k1 = 2 * kx + kc
    if len(cand) * n * k1 > 4e7:
        raise OESError("The grid and the data are too large for the bootstrap in the browser; reduce the grid or the trimming range.")
    # linear (no-threshold) model
    A0 = _demean(np.hstack([Xx, Zc]), u)
    Q0 = _qr(A0)
    yy = float(yd @ yd)
    S0 = yy - float(np.sum((Q0.T @ yd) ** 2))
    # single-threshold grid
    Qs = np.empty((len(cand), n, k1))
    for i, g in enumerate(cand):
        A, _ = _design(Xx, Zc, q, [g])
        Qs[i] = _qr(_demean(A, u))
    S_grid = yy - np.sum((np.einsum("gnk,n->gk", Qs, yd)) ** 2, axis=1)
    i1 = int(np.argmin(S_grid))
    g1, S1 = float(cand[i1]), float(S_grid[i1])
    dfe = n - N
    s2_1 = S1 / dfe
    F1 = (S0 - S1) / s2_1
    LR = (S_grid - S1) / s2_1
    rng = np.random.default_rng(int(spec.get("seed", 12345) or 12345))
    # bootstrap of F1 under the linear model
    b0 = np.linalg.lstsq(A0, yd, rcond=None)[0]
    e0 = yd - A0 @ b0
    fit0 = A0 @ b0
    eta = rng.standard_normal((n, B))
    Eb = _demean(e0[:, None] * eta, u)
    Yb = fit0[:, None] + Eb
    yyb = (Yb ** 2).sum(axis=0)
    S0b = yyb - np.sum((Q0.T @ Yb) ** 2, axis=0)
    proj = (Qs.transpose(0, 2, 1).reshape(-1, n) @ Yb).reshape(len(cand), k1, B)
    S1b = (yyb[None, :] - np.sum(proj ** 2, axis=1)).min(axis=0)
    F1b = (S0b - S1b) / (S1b / dfe)
    p1 = float(np.mean(F1b >= F1))
    cv1 = {lvl: float(np.quantile(F1b, lvl)) for lvl in (0.90, 0.95, 0.99)}
    # confidence interval for the first threshold
    c95, c90 = -2 * np.log(1 - np.sqrt(0.95)), -2 * np.log(1 - np.sqrt(0.90))
    ci1 = cand[LR <= c95]
    # second threshold, with the first fixed at its estimate
    thr, S_final = [g1], S1
    p2 = F2 = cv2 = g2 = ci2 = LR2 = cand2 = None
    if max_thr == 2:
        cand2 = np.array([g for g in cand if g != g1 and _min_regime(q, sorted([g1, g]), trim * n)])
        if len(cand2) >= 5:
            Q2 = np.empty((len(cand2), n, 3 * kx + kc))
            for i, g in enumerate(cand2):
                A, _ = _design(Xx, Zc, q, sorted([g1, g]))
                Q2[i] = _qr(_demean(A, u))
            S2g = yy - np.sum((np.einsum("gnk,n->gk", Q2, yd)) ** 2, axis=1)
            j = int(np.argmin(S2g))
            g2, S2 = float(cand2[j]), float(S2g[j])
            s2_2 = S2 / dfe
            F2 = (S1 - S2) / s2_2
            LR2 = (S2g - S2) / s2_2
            ci2 = cand2[LR2 <= c95]
            A1 = _demean(_design(Xx, Zc, q, [g1])[0], u)
            b1 = np.linalg.lstsq(A1, yd, rcond=None)[0]
            e1 = yd - A1 @ b1
            Yb2 = (A1 @ b1)[:, None] + _demean(e1[:, None] * rng.standard_normal((n, B)), u)
            yyb2 = (Yb2 ** 2).sum(axis=0)
            Q1 = Qs[i1]
            S1b2 = yyb2 - np.sum((Q1.T @ Yb2) ** 2, axis=0)
            k2 = Q2.shape[2]
            proj2 = (Q2.transpose(0, 2, 1).reshape(-1, n) @ Yb2).reshape(len(cand2), k2, B)
            S2b = (yyb2[None, :] - np.sum(proj2 ** 2, axis=1)).min(axis=0)
            F2b = (S1b2 - S2b) / (S2b / dfe)
            p2 = float(np.mean(F2b >= F2))
            cv2 = {lvl: float(np.quantile(F2b, lvl)) for lvl in (0.90, 0.95, 0.99)}
    n_thr = 0
    if p1 < 0.05:
        n_thr = 1
        if p2 is not None and p2 < 0.05:
            n_thr = 2
    show = max(n_thr, 1)
    gs = sorted([g1] + ([g2] if (show == 2 and g2 is not None) else []))
    # final estimates with clustered standard errors
    A, r = _design(Xx, Zc, q, gs)
    Aw = _demean(A, u)
    b = np.linalg.lstsq(Aw, yd, rcond=None)[0]
    e = yd - Aw @ b
    V = _panel._sandwich(Aw, e, groups=u, time=d["_ti"].to_numpy(), kind="clustered", extra_df=N)
    se = np.sqrt(np.diag(V))
    R = len(gs) + 1
    labs = []
    for j in range(R):
        if R == 2:
            labs.append(f"{qv} <= {f(gs[0], 4)}" if j == 0 else f"{qv} > {f(gs[0], 4)}")
        else:
            labs.append(f"{qv} <= {f(gs[0], 4)}" if j == 0 else (f"{qv} > {f(gs[-1], 4)}" if j == R - 1 else f"{f(gs[j - 1], 4)} < {qv} <= {f(gs[j], 4)}"))
    names = [f"{x} | {labs[j]}" for j in range(R) for x in xs] + ctrl
    tabs = []
    rows = [["No threshold against one threshold", f(F1, 3), f(p1, 3), f(cv1[0.90], 2), f(cv1[0.95], 2), f(cv1[0.99], 2)]]
    if p2 is not None:
        rows.append(["One threshold against two (first threshold fixed)", f(F2, 3), f(p2, 3), f(cv2[0.90], 2), f(cv2[0.95], 2), f(cv2[0.99], 2)])
    tabs.append(table("Tests for the number of thresholds", ["Test", "F statistic", "Bootstrap p-value", "10% critical value", "5% critical value", "1% critical value"], rows,
                      f"Bootstrap with {B} replications under the model without the threshold (Hansen 1999). The threshold is not identified under the null, so the bootstrap replaces the F distribution."))
    thr_rows = [["Threshold 1", f(g1, 4), f"[{f(ci1.min(), 4)}, {f(ci1.max(), 4)}]", f"{100 * np.mean(q <= g1):.1f}% of observations at or below"]]
    if g2 is not None and show == 2:
        thr_rows.append(["Threshold 2", f(g2, 4), f"[{f(ci2.min(), 4)}, {f(ci2.max(), 4)}]", "found with threshold 1 fixed"])
    tabs.append(table("Estimated thresholds", ["Threshold", "Estimate", "95% confidence interval", "Note"], thr_rows,
                      f"The interval is the set of thresholds at which the likelihood-ratio statistic is below {f(c95, 2)} (Hansen 1999)."))
    tabs.append(coef_table(names, b, se, dist="t", df=dfe - len(b), title=f"Coefficients by regime ({R} regimes)",
                           note="Unit fixed effects removed; standard errors clustered by unit."))
    rg = [[labs[j], int((r == j).sum()), f(100 * np.mean(r == j), 1)] for j in range(R)]
    tabs.append(table("Observations by regime", ["Regime", "Observations", "Share (%)"], rg))
    charts = [chart(f"Likelihood-ratio statistic for the threshold in {qv}", [float(g) for g in cand], [("LR(threshold)", LR), ("5% critical value", np.full(len(cand), c95))], qv, "LR")]
    if cand2 is not None and LR2 is not None and show == 2:
        charts.append(chart(f"Likelihood-ratio statistic for the second threshold in {qv}", [float(g) for g in cand2], [("LR(threshold)", LR2), ("5% critical value", np.full(len(cand2), c95))], qv, "LR"))
    sig = n_thr
    summary = [stat("Observations", n), stat("Units", N), stat("Periods", T), stat("Threshold variable", qv), stat("Thresholds found", sig),
               stat("Threshold estimate", g1, 4), stat("Bootstrap p (F1)", p1, 3)]
    notes = ["The slopes of " + ", ".join(xs) + " differ across regimes; the controls (if any) have a common slope. Unit fixed effects are removed by the within transformation.",
             f"The grid has {len(cand)} candidate thresholds between the {trim:.0%} and {1 - trim:.0%} quantiles of {qv}; every regime holds at least {trim:.0%} of the observations."]
    if n_thr == 0:
        notes.append("The test does not reject a linear model at 5%. The estimates shown are the single-threshold fit and are descriptive: do not interpret a regime difference without evidence of a threshold.")
    if p2 is not None:
        notes.append("The second-threshold test and interval fix the first threshold at its estimate, which is valid because that estimate converges faster than the second.")
    notes.append("The F statistic and the confidence interval assume homoskedastic errors within units; the bootstrap draws errors with the unit's own residual variance structure. Use the clustered standard errors for the slopes.")
    return result("pthresh", f"Panel threshold regression: {y}", n, summary, tabs, charts, notes)


def _min_regime(q, gs, m):
    r = np.searchsorted(np.asarray(gs, float), q, side="left")
    return all((r == j).sum() >= m for j in range(len(gs) + 1))
