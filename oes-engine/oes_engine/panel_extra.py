"""Two-way fixed effects with cluster inference, dynamic panel threshold regression, panel cointegration tests,
panel quantile regression and spatial panel models (SAR, SEM, SDM). Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import _panel_tests
from .core import OESError, as_list, chart, distinct, f, int_option, float_option, need, result, stat, table, test_row, tests_table
from .panel import _panel


def _codes(d):
    return d["_u"].to_numpy(int), d["_ti"].to_numpy(int)


def _tw_demean(a, u, t, N, T, iters=200):
    """Remove unit and time means by alternating projections (one pass if balanced)."""
    a = np.asarray(a, float).copy()
    nu = np.bincount(u, minlength=N).astype(float); nt = np.bincount(t, minlength=T).astype(float)
    for _ in range(iters):
        old = a.copy()
        if a.ndim == 1:
            a -= (np.bincount(u, a, N) / nu)[u]; a -= (np.bincount(t, a, T) / nt)[t]
        else:
            for j in range(a.shape[1]):
                a[:, j] -= (np.bincount(u, a[:, j], N) / nu)[u]; a[:, j] -= (np.bincount(t, a[:, j], T) / nt)[t]
        if np.max(np.abs(a - old)) < 1e-11:
            break
    return a


def _cluster_v(X, e, groups, p_total):
    n, k = X.shape; XtXi = np.linalg.pinv(X.T @ X)
    G = int(groups.max()) + 1; S = np.zeros((k, k))
    sc = X * e[:, None]
    acc = np.zeros((G, k)); np.add.at(acc, groups, sc)
    S = acc.T @ acc
    return XtXi @ S @ XtXi * G / (G - 1) * (n - 1) / (n - p_total)


def _balanced(d, N, T, what):
    if len(d) != N * T:
        raise OESError(f"{what} needs a balanced panel (every unit observed in every period). This panel has {N} units, {T} periods and {len(d)} rows.")


# ---------------------------------------------------------------------------------------------------------------
# Two-way fixed effects with the choice of standard errors, wild cluster bootstrap and placebo experiment
# ---------------------------------------------------------------------------------------------------------------
def twfe(df, spec):
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs)
    n = len(d); k = len(xs); u, t = _codes(d)
    yv = d[y].to_numpy(float); Xv = d[xs].to_numpy(float)
    yt = _tw_demean(yv, u, t, N, T); Xt = _tw_demean(Xv, u, t, N, T)
    b = np.linalg.lstsq(Xt, yt, rcond=None)[0]; e = yt - Xt @ b
    p_full = k + (N - 1) + (T - 1) + 1
    XtXi = np.linalg.inv(Xt.T @ Xt)
    iid = XtXi * (e @ e) / (n - p_full)
    hc = XtXi @ ((Xt * e[:, None]).T @ (Xt * e[:, None])) @ XtXi * n / (n - p_full)
    Vu = _cluster_v(Xt, e, u, p_full); Vt = _cluster_v(Xt, e, t, p_full)
    Vut = _cluster_v(Xt, e, np.arange(n), p_full)
    V2 = Vu + Vt - Vut
    ses = {"Default (iid)": iid, "Robust (HC1)": hc, "Cluster by unit": Vu, "Cluster by time": Vt, "Two-way (unit, time)": V2}
    srow = []
    for nm, V in ses.items():
        sd = np.sqrt(np.clip(np.diag(V), 0, None))
        srow.append([nm] + [v for j in range(k) for v in (f(sd[j], 4), f(b[j] / sd[j], 2))])
    cols = ["Method"] + [c for x in xs for c in (f"s.e. {x}", f"t {x}")]
    # comparison with pooled and unit-effects-only estimates
    Xp = np.column_stack([np.ones(n), Xv]); bp = np.linalg.lstsq(Xp, yv, rcond=None)[0]
    ud = _tw_demean_1way(yv, u, N); Xud = np.column_stack([_tw_demean_1way(Xv[:, j], u, N) for j in range(k)])
    bu = np.linalg.lstsq(Xud, ud, rcond=None)[0]
    comp = [[x, f(bp[j + 1], 4), f(bu[j], 4), f(b[j], 4)] for j, x in enumerate(xs)]
    # variance shares
    vs = []
    for nm, a in zip([y] + xs, [yv] + [Xv[:, j] for j in range(k)]):
        tot = np.var(a)
        vs.append([nm, f(np.var(pd.Series(a).groupby(u).transform("mean")) / tot, 3), f(np.var(pd.Series(a).groupby(t).transform("mean")) / tot, 3),
                   f(np.var(_tw_demean(a, u, t, N, T)) / tot, 3)])
    # serial correlation of residuals within units
    ed = pd.DataFrame({"u": u, "e": e}); ed["l"] = ed.groupby("u").e.shift(1); m = ed.dropna()
    rho = float((m.e * m.l).sum() / (m.l ** 2).sum())
    # wild cluster bootstrap-t for the first regressor, null imposed
    R = int_option(spec, "boot", 499, 99, 2000, "Bootstrap draws"); rng = np.random.default_rng(int(spec.get("seed") or 0))
    def tstat(ys):
        bb = np.linalg.lstsq(Xt, ys, rcond=None)[0]; ee = _tw_demean(ys - Xt @ bb, u, t, N, T) if False else ys - Xt @ bb
        V = _cluster_v(Xt, ee, u, p_full); return bb[0], bb[0] / np.sqrt(V[0, 0])
    b_hat, t_hat = tstat(yt)
    X0 = Xt[:, 1:]
    if k > 1:
        b0 = np.linalg.lstsq(X0, yt, rcond=None)[0]; fit0 = X0 @ b0
    else:
        fit0 = np.zeros(n)
    e0 = yt - fit0; tb = []
    for _ in range(R):
        w = rng.choice([-1.0, 1.0], N)[u]
        ys = _tw_demean(fit0 + e0 * w, u, t, N, T)
        tb.append(tstat(ys)[1])
    tb = np.array(tb); p_boot = (np.abs(tb) >= abs(t_hat)).mean()
    p_norm = 2 * stats.norm.sf(abs(t_hat)); p_t = 2 * stats.t.sf(abs(t_hat), N - 1)
    # placebo experiment
    Rp = int_option(spec, "placebo", 200, 0, 1000, "Placebo draws"); rej = {"Default (iid)": 0, "Robust (HC1)": 0, "Cluster by unit": 0}
    prow = None
    if Rp > 0:
        for _ in range(Rp):
            xm = np.zeros((N, T)); uu = rng.standard_normal((N, T))
            for s in range(1, T):
                xm[:, s] = .8 * xm[:, s - 1] + uu[:, s]
            xf = _tw_demean(xm[u, t], u, t, N, T)
            bb = (xf @ yt) / (xf @ xf); ee = yt - bb * xf
            si = np.sqrt(ee @ ee / (n - N - T) / (xf @ xf)); sh = np.sqrt(((xf * ee) ** 2).sum()) / (xf @ xf)
            sc = np.bincount(u, xf * ee, N); sc_ = np.sqrt((sc ** 2).sum() * N / (N - 1)) / (xf @ xf)
            for nm, s_ in zip(rej, (si, sh, sc_)):
                rej[nm] += abs(bb / s_) > 1.96
        prow = [[nm, f(v / Rp, 3)] for nm, v in rej.items()]
    tabs = [table("Variance shares", ["Variable", "Between units", "Between periods", "Left after two-way demeaning"], vs),
            table("Coefficients under three sets of effects", ["Variable", "Pooled OLS", "Unit effects", "Unit + time effects"], comp),
            table("Standard errors for the two-way fixed-effects model", cols, srow,
                  f"Residual autocorrelation within units: {rho:.2f}. {N} unit clusters and {T} period clusters. Two-way clustering follows Cameron, Gelbach and Miller (2011)."),
            table(f"Wild cluster bootstrap for {xs[0]}", ["Item", "Value"],
                  [["Coefficient", f(b_hat, 4)], ["Cluster-robust t", f(t_hat, 3)], ["p-value, normal", f(p_norm, 4)], [f"p-value, t({N - 1})", f(p_t, 4)],
                   [f"Wild cluster bootstrap-t p-value ({R} draws)", f(p_boot, 4) if p_boot > 0 else f"< {1 / R:.4f}"],
                   ["Bootstrap 2.5% and 97.5% quantiles of t", f"{np.percentile(tb, 2.5):.2f}, {np.percentile(tb, 97.5):.2f}"]],
                  "Rademacher weights by unit, null imposed on the first regressor.")]
    if prow:
        tabs.append(table("Placebo experiment: rejection rate of a true 5% null", ["Standard error", "Share rejecting"], prow,
                          f"{Rp} persistent AR(1) (0.8) regressors unrelated to the outcome; a correct test rejects about 5% of the time."))
    ch = [chart("t-statistic: wild cluster bootstrap distribution", list(range(len(tb))), [("bootstrap t (sorted)", np.sort(tb))], "", "t", kind="line")]
    summary = [stat("Observations", n), stat("Units", N), stat("Periods", T), stat(f"Coefficient on {xs[0]}", float(b[0]), 4), stat("Cluster s.e.", float(np.sqrt(Vu[0, 0])), 4), stat("Bootstrap p", float(p_boot), 4)]
    notes = ["Unit and period effects are removed by demeaning (identical coefficients to dummy variables). For staggered treatment timing, two-way fixed effects can be biased by negative weights; use the event-study or synthetic DiD models."]
    return result("twfe", f"Two-way fixed effects with clustered inference: {y}", n, summary, tabs, ch, notes)


def _tw_demean_1way(a, u, N):
    a = np.asarray(a, float); nu = np.bincount(u, minlength=N).astype(float)
    return a - (np.bincount(u, a, N) / nu)[u]


# ---------------------------------------------------------------------------------------------------------------
# Dynamic panel threshold regression (first-difference IV, Seo-Shin style)
# ---------------------------------------------------------------------------------------------------------------
def _tsls(y, X, Z):
    Xh = Z @ (np.linalg.pinv(Z.T @ Z) @ (Z.T @ X))
    b = np.linalg.solve(Xh.T @ X, Xh.T @ y); return b, y - X @ b


def dpthresh(df, spec):
    y = need(spec, "y", "dependent variable"); xr = need(spec, "x", "regressor with the threshold effect"); ctrl = as_list(spec.get("controls"))
    qv = spec.get("tvar") or xr
    distinct([y], [xr], ctrl)
    d, unit, time, N, T = _panel(df, spec, [y, xr] + ctrl + ([qv] if qv not in [y, xr] + ctrl else []))
    if T < 6:
        raise OESError("The dynamic threshold model needs at least 6 periods per unit.")
    d = d.sort_values(["_u", "_ti"]).reset_index(drop=True)
    g = d.groupby("_u")
    def lag(v, L):
        out = g[v].shift(L); okc = (g["_ti"].shift(L) == d["_ti"] - L); return out.where(okc)
    d["dy"] = d[y] - lag(y, 1); d["dy_l1"] = lag(y, 1) - lag(y, 2); d["y_l2"] = lag(y, 2); d["y_l3"] = lag(y, 3)
    d["dx"] = d[xr] - lag(xr, 1); d["q"] = lag(qv, 1)
    for c in ctrl:
        d["d_" + c] = d[c] - lag(c, 1)
    need_cols = ["dy", "dy_l1", "dx", "q", "y_l2", "y_l3"] + ["d_" + c for c in ctrl]
    use = d.dropna(subset=need_cols).copy()
    if len(use) < 100:
        raise OESError("Too few usable observations after differencing and lags (needs consecutive periods).")
    ids = pd.factorize(use["_u"])[0]; G = ids.max() + 1
    yrs = pd.get_dummies(use["_ti"], drop_first=True).astype(float).to_numpy()
    q = use["q"].to_numpy(float); yy = use["dy"].to_numpy(float); X0 = use[["dy_l1"]].to_numpy(float); Xx = use[["dx"]].to_numpy(float)
    K = use[["d_" + c for c in ctrl]].to_numpy(float) if ctrl else np.zeros((len(use), 0)); Zn = use[["y_l2", "y_l3"]].to_numpy(float)
    nobs = len(use)
    # lagged dependent variable under three estimators
    dd = d.dropna(subset=[y]).copy(); dd["yl"] = lag(y, 1).reindex(dd.index); dd = dd.dropna(subset=["yl"])
    ols = np.polyfit(dd["yl"], dd[y], 1)[0]
    ym = dd[y] - dd.groupby("_u")[y].transform("mean"); xm = dd["yl"] - dd.groupby("_u")["yl"].transform("mean"); fe = float((xm @ ym) / (xm @ xm))
    Xl = np.column_stack([X0, Xx, K, yrs]); Zl = np.column_stack([Zn, Xx, K, yrs])
    b_lin, e_lin = _tsls(yy, Xl, Zl); ssr0 = float(e_lin @ e_lin)
    def fit(gam, y_=None):
        lo = (q <= gam).astype(float)[:, None]; hi = 1 - lo
        X = np.hstack([X0, Xx * lo, Xx * hi, K, yrs]); Z = np.hstack([Zn, Xx * lo, Xx * hi, K, yrs])
        b, e = _tsls(yy if y_ is None else y_, X, Z); return b, e, X, Z
    trim = float_option(spec, "trim", .15, "Trimming")
    if not 0.05 <= trim <= 0.4:
        raise OESError("Trimming must be between 0.05 and 0.4.")
    ng = int_option(spec, "grid", 71, 11, 101, "Grid points")
    grid = np.unique(np.quantile(q, np.linspace(trim, 1 - trim, ng)))
    ssr = np.array([float((lambda r: r[1] @ r[1])(fit(gm))) for gm in grid]); kk = int(np.argmin(ssr)); gam = float(grid[kk])
    sig2 = ssr[kk] / (nobs - Xl.shape[1] - 1)
    F_obs = (ssr0 - ssr[kk]) / sig2
    R = int_option(spec, "boot", 99, 19, 499, "Bootstrap draws"); rng = np.random.default_rng(int(spec.get("seed") or 0)); fitted = Xl @ b_lin; Fb = []
    for _ in range(R):
        w = rng.choice([-1.0, 1.0], G)[ids]; ys = fitted + e_lin * w
        s0 = float((lambda r: r[1] @ r[1])(_tsls(ys, Xl, Zl))); s1 = min(float((lambda r: r[1] @ r[1])(fit(gm, ys))) for gm in grid)
        Fb.append((s0 - s1) / (s1 / (nobs - Xl.shape[1] - 1)))
    Fb = np.array(Fb); pboot = float((Fb >= F_obs).mean())
    LR = (ssr - ssr[kk]) / sig2; ok = grid[LR <= 7.35]
    b, e, X, Z = fit(gam)
    Xh = Z @ (np.linalg.pinv(Z.T @ Z) @ (Z.T @ X)); A = np.linalg.inv(Xh.T @ Xh)
    acc = np.zeros((G, X.shape[1])); np.add.at(acc, ids, Xh * e[:, None]); V = A @ (acc.T @ acc) @ A * G / (G - 1)
    se = np.sqrt(np.diag(V)); nm = [f"lagged {y} (change)", f"{xr}, low regime", f"{xr}, high regime"] + [f"{c} (change)" for c in ctrl]
    keep = list(range(3 + len(ctrl)))
    ct = table("Estimates at the estimated threshold", ["Variable", "Coef.", "Cluster s.e.", "t"], [[nm[i], f(b[i], 4), f(se[i], 4), f(b[i] / se[i], 2)] for i in keep])
    dv = float(np.array([1, -1]) @ V[1:3, 1:3] @ np.array([1, -1]))
    gi = np.array([Zn[ids == i].T @ e[ids == i] for i in range(G)])
    try:
        Jv = float(gi.sum(0) @ np.linalg.inv(gi.T @ gi) @ gi.sum(0))
    except np.linalg.LinAlgError:
        Jv = float("nan")
    df_ = pd.DataFrame({"u": ids, "e": e}); df_["e2"] = df_.groupby("u").e.shift(2); m2 = df_.dropna()
    ar2 = float((m2.e * m2.e2).sum() / np.sqrt(((m2.e * m2.e2) ** 2).sum()))
    tests = [test_row("Regime difference in the slope of " + xr, (b[1] - b[2]) / np.sqrt(dv), 2 * stats.norm.sf(abs((b[1] - b[2]) / np.sqrt(dv))), f"difference {b[1] - b[2]:.3f}, s.e. {np.sqrt(dv):.3f}"),
             test_row(f"Threshold effect (sup-F, {R} bootstrap draws)", F_obs, pboot, "H0: no threshold; wild bootstrap by unit, regressors held fixed"),
             test_row("AR(2) in the differenced residuals", ar2, 2 * stats.norm.sf(abs(ar2)), "H0: no second-order autocorrelation"),
             test_row("Hansen J (lagged-level instruments, 1 df)", Jv, stats.chi2.sf(Jv, 1), "H0: instruments valid")]
    tabs = [table("Persistence under three estimators", ["Estimator", f"Coefficient on lagged {y}"], [["Pooled OLS", f(ols, 3)], ["Within (fixed effects)", f(fe, 3)], ["First-difference IV (linear)", f(float(b_lin[0]), 3)]],
                  "OLS is biased upward and the within estimator downward in short panels (Nickell bias); the IV estimate is the one used below."),
            table("Threshold", ["Item", "Value"], [["Estimated threshold", f(gam, 4)], ["Approximate 95% interval", f"[{ok.min():.3f}, {ok.max():.3f}]"],
                                                  ["Share of observations at or below it", f((q <= gam).mean(), 3)], ["IV sum of squares, linear model", f(ssr0, 2)], ["IV sum of squares, threshold model", f(ssr[kk], 2)],
                                                  [f"Bootstrap 90th, 95th, 99th percentile of F", ", ".join(f"{v:.2f}" for v in np.percentile(Fb, [90, 95, 99]))]],
                  f"Threshold variable: lagged {qv}. Search over {len(grid)} grid values between the {int(trim * 100)}th and {int((1 - trim) * 100)}th percentiles. Interval: values with LR below 7.35 (Hansen)."),
            ct, tests_table(tests)]
    ch = [chart("IV residual sum of squares by candidate threshold", grid, [("Threshold model", ssr), ("Linear model", [ssr0] * len(grid))], f"Candidate threshold (lagged {qv})", "Sum of squares")]
    summary = [stat("Observations", nobs), stat("Units", int(G)), stat("Threshold", gam, 4), stat("Bootstrap p", pboot, 3), stat("Regime slopes", f"{b[1]:.3f} / {b[2]:.3f}")]
    notes = ["First-difference instrumental variables: the lagged change is instrumented by the second and third lags of the level; the regressor and controls enter in differences and are treated as exogenous. The threshold variable is predetermined (lagged level). Year dummies are included."]
    return result("dpthresh", f"Dynamic panel threshold: {y}", nobs, summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Panel cointegration tests
# ---------------------------------------------------------------------------------------------------------------
def _ols_b(y, X):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    return b, e, np.sqrt(np.diag(XtXi) * (e @ e) / (len(y) - X.shape[1]))


def _adf_t(e, k=1):
    de = np.diff(e); n = len(de)
    cols = [e[k:-1]] + [de[k - j:n - j] for j in range(1, k + 1)]
    b, _, se = _ols_b(de[k:], np.column_stack(cols)); return b[0] / se[0]


def _group_adf(Y, Xs, k=1):
    ts = []
    for i in range(Y.shape[1]):
        X = np.column_stack([np.ones(len(Y))] + [x[:, i] for x in Xs]); ts.append(_adf_t(_ols_b(Y[:, i], X)[1], k))
    return float(np.mean(ts)), np.array(ts)


def _westerlund_gt(Y, Xs, k=0):
    out = []; T_ = len(Y)
    for i in range(Y.shape[1]):
        y = Y[:, i]; x = np.column_stack([a[:, i] for a in Xs]); dy = np.diff(y); idx = np.arange(k + 1, T_); dx = np.diff(x, axis=0)
        cols = [np.ones(len(idx)), y[idx - 1]] + [x[idx - 1, j] for j in range(x.shape[1])]
        cols += [dy[idx - 1 - L] for L in range(1, k + 1)]
        cols += [dx[idx - 1 - L, j] for L in range(0, k + 1) for j in range(x.shape[1])]
        b, _, se = _ols_b(dy[idx - 1], np.column_stack(cols)); out.append(b[1] / se[1])
    return float(np.mean(out)), np.array(out)


def pcoint(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs)
    _balanced(d, N, T, "Panel cointegration tests")
    if T < 12 or N < 5:
        raise OESError("Panel cointegration tests need at least 5 units and 12 periods.")
    names = d.drop_duplicates("_u").sort_values("_u")[unit].astype(str).tolist()
    Wl = {v: d.pivot(index="_ti", columns="_u", values=v).sort_index() for v in [y] + xs}
    for v in Wl: Wl[v].columns = names
    rows = []
    for v in [y] + xs:
        lev, dif = Wl[v], Wl[v].diff().dropna()
        c1, c2 = _panel_tests.cips_test(lev), _panel_tests.cips_test(dif); f1, f2 = _panel_tests.fisher_adf_test(lev, maxlag=1), _panel_tests.fisher_adf_test(dif, maxlag=1)
        g = lambda dct, k: (dct or {}).get(k)
        rows.append([v, g(c1, "stat"), g(c1, "p"), g(f1, "Z_p"), g(c2, "stat"), g(c2, "p"), g(f2, "Z_p")])
    Yv = Wl[y].to_numpy(float); Xl = [Wl[v].to_numpy(float) for v in xs]
    kao = _panel_tests.kao_test(d, unit, y, xs) or {}
    g_stat, g_i = _group_adf(Yv, Xl)
    R = int_option(spec, "sims", 200, 50, 1000, "Simulation replications"); rng = np.random.default_rng(int(spec.get("seed") or 0))
    def simulate(fn, demean=False):
        cs = (lambda A: A - A.mean(1, keepdims=True)) if demean else (lambda A: A)
        out = []
        for _ in range(R):
            Ys = np.cumsum(rng.standard_normal(Yv.shape), 0); Xss = [np.cumsum(rng.standard_normal(Yv.shape), 0) for _ in xs]
            out.append(fn(cs(Ys), [cs(a) for a in Xss]))
        return np.array(out)
    tests = [("Pedroni-type group ADF", lambda a, b: _group_adf(a, b)[0]), ("Westerlund Gt, no lags", lambda a, b: _westerlund_gt(a, b, 0)[0]), ("Westerlund Gt, 1 lag", lambda a, b: _westerlund_gt(a, b, 1)[0])]
    res = []
    for lab, fn in tests:
        obs = fn(Yv, Xl); sim = simulate(fn); cv = np.percentile(sim, [5, 10]); res.append([lab, f(obs, 3), f(cv[0], 3), f(cv[1], 3), f(((sim <= obs).sum() + 1) / (len(sim) + 1), 4)])
    # cross-sectional dependence of the residuals and cross-sectional demeaning
    E = np.column_stack([_ols_b(Yv[:, i], np.column_stack([np.ones(T)] + [a[:, i] for a in Xl]))[1] for i in range(N)])
    ed = pd.DataFrame(E).diff().dropna(); cd = _panel_tests.cd_test(ed)
    cs = lambda A: A - A.mean(1, keepdims=True)
    res2 = []
    for lab, fn in tests[:2]:
        obs = fn(cs(Yv), [cs(a) for a in Xl]); sim = simulate(fn, True); res2.append([lab + ", cross-section demeaned", f(obs, 3), f(np.percentile(sim, 5), 3), f(((sim <= obs).sum() + 1) / (len(sim) + 1), 4)])
    # DOLS mean group against fixed effects
    kx = len(xs)
    def dols(i, ll=1):
        yv = Yv[:, i]; x = np.column_stack([a[:, i] for a in Xl]); dx = np.diff(x, axis=0); idx = np.arange(ll + 1, T - ll)
        cols = [np.ones(len(idx))] + [x[idx, j] for j in range(kx)] + [dx[idx - 1 + s, j] for s in range(-ll + 1, ll + 1) for j in range(kx)]
        return _ols_b(yv[idx], np.column_stack(cols))[0][1:1 + kx]
    B = np.array([dols(i) for i in range(N)]); mg = B.mean(0); mg_se = B.std(0, ddof=1) / np.sqrt(N)
    ym = d[y] - d.groupby("_u")[y].transform("mean"); xm = d[xs] - d.groupby("_u")[xs].transform("mean")
    fe = sm.OLS(ym, xm).fit(cov_type="cluster", cov_kwds={"groups": d["_u"].to_numpy()})
    tabs = [table("Panel unit-root tests", ["Variable", "CIPS level", "p", "Fisher p", "CIPS difference", "p ", "Fisher p "], rows,
                  "CIPS allows common shocks; Fisher-ADF assumes independent units. H0: unit root (non-stationary)."),
            table("Residual-based tests of no cointegration", ["Test", "Statistic", "p-value"],
                  [["Kao ADF (common slopes)", kao.get("stat"), kao.get("p")], ["Group-mean ADF on unit residuals", f(g_stat, 3), None]], "Country-by-country ADF t-statistics: " + ", ".join(f"{n}: {v:.1f}" for n, v in zip(names, g_i))),
            table(f"Simulated critical values and p-values ({R} replications)", ["Test", "Statistic", "5% cv", "10% cv", "p-value"], res,
                  "Null: independent random walks, no cointegration. p = (count + 1)/(R + 1). Reject for small (negative) statistics."),
            table("Cross-sectional dependence and robustness", ["Test", "Statistic", "5% cv", "p-value"], res2,
                  f"Pesaran CD test on differenced residuals: CD = {f(cd[0], 2)}, p = {f(cd[1], 4)}, mean absolute correlation {f(cd[2], 2)}." if cd[0] is not None else None),
            table("Long-run coefficients", ["Variable", "Fixed effects (levels)", "FE cluster s.e.", "DOLS mean group", "MG s.e.", "MG median"],
                  [[x, f(float(np.asarray(fe.params)[j]), 3), f(float(np.asarray(fe.bse)[j]), 3), f(mg[j], 3), f(mg_se[j], 3), f(np.median(B[:, j]), 3)] for j, x in enumerate(xs)],
                  "DOLS with one lead and one lag by unit, averaged over units.")]
    ch = [chart("Country ADF t-statistics on the cointegrating residuals", names, [("ADF t", g_i)], "", "t-statistic", kind="bar")]
    summary = [stat("Units", N), stat("Periods", T), stat("Kao p", kao.get("p")), stat("Group ADF", g_stat, 3), stat("Westerlund Gt p", float(res[1][-1]), 3)]
    return result("pcoint", f"Panel cointegration: {y}", len(d), summary, tabs, ch,
                  ["Westerlund Gt is a simplified error-correction version (its own statistic, simulated critical values), not the packaged Westerlund (2007) bootstrap; treat p-values as approximate."])


# ---------------------------------------------------------------------------------------------------------------
# Panel quantile regression: pooled, Canay two-step and MM-QR
# ---------------------------------------------------------------------------------------------------------------
def _pq_estimators(Y, X, taus):
    from statsmodels.regression.quantile_regression import QuantReg
    n_, T_, k_ = X.shape
    Xc = np.column_stack([np.ones(n_ * T_), X.reshape(-1, k_)])
    out = {"pooled": [], "canay": [], "mmqr": []}
    Xm = X - X.mean(1, keepdims=True); ym = Y - Y.mean(1, keepdims=True)
    b = np.linalg.lstsq(Xm.reshape(-1, k_), ym.ravel(), rcond=None)[0]
    alpha = Y.mean(1) - X.mean(1) @ b; ystar = (Y - alpha[:, None]).ravel(); res = (ym - Xm @ b).ravel()
    a_abs = np.abs(ym - Xm @ b); am = a_abs - a_abs.mean(1, keepdims=True)
    gam = np.linalg.lstsq(Xm.reshape(-1, k_), am.ravel(), rcond=None)[0]; delta = a_abs.mean(1) - X.mean(1) @ gam
    scale = (delta[:, None] + X @ gam).ravel(); pos = scale > 1e-3
    for tau in taus:
        out["pooled"].append(QuantReg(Y.ravel(), Xc).fit(q=tau, max_iter=3000).params[1:])
        out["canay"].append(QuantReg(ystar, Xc).fit(q=tau, max_iter=3000).params[1:])
        qt = QuantReg(res[pos], scale[pos]).fit(q=tau, max_iter=3000).params[0]
        out["mmqr"].append(b + gam * qt)
    return {m: np.array(v) for m, v in out.items()}, int((~pos).sum()), b


def pquant(df, spec):
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs)
    _balanced(d, N, T, "Panel quantile regression")
    d = d.sort_values(["_u", "_ti"]); k = len(xs)
    taus = [float(v) for v in str(spec.get("taus") or "0.1,0.25,0.5,0.75,0.9").replace(" ", "").split(",") if v]
    if not taus or any(not 0 < t < 1 for t in taus) or len(taus) < 3:
        raise OESError("Give at least three quantiles strictly between 0 and 1, separated by commas.")
    taus = sorted(taus)
    twoway = str(spec.get("time_effects", "yes")) == "yes"
    Xw = d[xs].copy()
    if twoway and T > 2:
        yd = pd.get_dummies(d["_ti"], drop_first=True).astype(float); yd.columns = [f"_t{c}" for c in yd.columns]; Xw = pd.concat([Xw, yd], axis=1)
    Yarr = d[y].to_numpy(float).reshape(N, T); Xarr = Xw.to_numpy(float).reshape(N, T, -1)
    est, nbad, b_fe = _pq_estimators(Yarr, Xarr, taus)
    R = int_option(spec, "boot", 100, 20, 500, "Bootstrap replications"); rng = np.random.default_rng(int(spec.get("seed") or 0)); boot = {m: [] for m in est}
    for _ in range(R):
        idx = rng.integers(0, N, N)
        try:
            e_, _, _ = _pq_estimators(Yarr[idx], Xarr[idx], taus)
        except Exception:
            continue
        for m in boot: boot[m].append(e_[m])
    boot = {m: np.array(v) for m, v in boot.items()}
    if len(boot["pooled"]) < 10:
        raise OESError("The bootstrap failed too often; check for constant regressors within the resampled units.")
    lab = {"pooled": "Pooled quantile regression", "canay": "Canay two-step fixed-effects QR", "mmqr": "MM-QR (location-scale)"}
    tabs = []
    first = xs[0]
    rows = []
    for m in est:
        for j, tau in enumerate(taus):
            lo, hi = np.percentile(boot[m][:, j, 0], [5, 95]); rows.append([lab[m], tau, f(est[m][j, 0], 4), f(boot[m][:, j, 0].std(ddof=1), 4), f(lo, 4), f(hi, 4)])
    tabs.append(table(f"Coefficient on {first} by quantile", ["Method", "Quantile", "Coef.", "Bootstrap s.e.", "90% low", "90% high"], rows,
                      f"Bootstrap over units ({len(boot['pooled'])} replications). Worker fixed-effects (mean) coefficient: {b_fe[0]:.4f}."))
    full = []
    for m in est:
        for jx, x in enumerate(xs):
            full.append([lab[m], x] + [f(est[m][j, jx], 3) for j in range(len(taus))])
    tabs.append(table("All coefficients by quantile", ["Method", "Variable"] + [f"q{t:g}" for t in taus], full))
    drows = []; i0, i1 = 0, len(taus) - 1
    for m in est:
        dd = boot[m][:, i1, 0] - boot[m][:, i0, 0]; e_ = est[m][i1, 0] - est[m][i0, 0]; lo, hi = np.percentile(dd, [2.5, 97.5]); p = (np.abs(dd - dd.mean()) >= abs(e_)).mean()
        drows.append([lab[m], f(e_, 4), f(lo, 4), f(hi, 4), f(p, 4)])
    tabs.append(table(f"Difference in the {first} coefficient: q{taus[i1]:g} minus q{taus[i0]:g}", ["Method", "Difference", "95% low", "95% high", "p-value (bootstrap)"], drows))
    sd_alpha = float(np.std(Yarr.mean(1) - Xarr.mean(1) @ b_fe)); sd_w = float(np.std((Yarr - Yarr.mean(1, keepdims=True)) - (Xarr - Xarr.mean(1, keepdims=True)) @ b_fe))
    tabs.append(table("Unit effects and scale", ["Item", "Value"], [["SD of the estimated unit effects", f(sd_alpha, 3)], ["SD of the within residuals", f(sd_w, 3)],
                                                                  ["Observations with non-positive fitted scale (dropped in MM-QR)", nbad]]))
    ch = [chart(f"Coefficient on {first} by quantile", taus, [(lab[m], est[m][:, 0]) for m in est] + [("Mean (fixed effects)", [b_fe[0]] * len(taus))], "Quantile", "Coefficient")]
    summary = [stat("Observations", len(d)), stat("Units", N), stat("Periods", T), stat(f"{first}: Canay q{taus[0]:g} to q{taus[-1]:g}", f"{est['canay'][0, 0]:.3f} to {est['canay'][-1, 0]:.3f}")]
    notes = ["Canay (2011): estimate unit effects from a within regression, subtract them, then run pooled quantile regression. MM-QR (Machado and Santos Silva 2019): location-scale model with unit effects in both location and scale. Pooled quantile regression ignores unit effects."]
    return result("pquant", f"Panel quantile regression: {y}", len(d), summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Spatial panel: SAR, SEM, SDM with a weights matrix from a neighbour list or coordinates
# ---------------------------------------------------------------------------------------------------------------
def _weights(d, spec, names):
    unit = spec["unit"]; N = len(names); idx = {str(n): i for i, n in enumerate(names)}
    A = np.zeros((N, N))
    ncol = spec.get("neighbours")
    if ncol:
        if ncol not in d.columns:
            raise OESError(f"Column {ncol} not found.")
        first = d.drop_duplicates("_u").sort_values("_u")
        for i, v in zip(first["_u"], first[ncol]):
            if pd.isna(v): continue
            for tok in str(v).replace(",", ";").replace(" ", ";").split(";"):
                tok = tok.strip()
                if not tok: continue
                if tok not in idx:
                    raise OESError(f"Neighbour '{tok}' of unit {names[i]} is not a unit in the data.")
                A[i, idx[tok]] = 1
        A = np.maximum(A, A.T)
    else:
        la, lo = spec.get("lat"), spec.get("lon")
        if not (la and lo):
            raise OESError("Choose either a neighbour-list column or both latitude and longitude columns to build the weights matrix.")
        first = d.drop_duplicates("_u").sort_values("_u")
        P = first[[la, lo]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if np.isnan(P).any():
            raise OESError("Latitude and longitude must be numeric for every unit.")
        k = int_option(spec, "knn", 5, 1, max(N - 1, 1), "Nearest neighbours")
        la_r, lo_r = np.radians(P[:, 0]), np.radians(P[:, 1])
        dl = np.sin((la_r[:, None] - la_r[None]) / 2) ** 2 + np.cos(la_r)[:, None] * np.cos(la_r)[None] * np.sin((lo_r[:, None] - lo_r[None]) / 2) ** 2
        D = 2 * 6371 * np.arcsin(np.sqrt(np.clip(dl, 0, 1))); np.fill_diagonal(D, np.inf)
        for i in range(N):
            A[i, np.argsort(D[i])[:k]] = 1
        A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    if (A.sum(1) == 0).any():
        raise OESError("Some units have no neighbours: " + ", ".join(str(names[i]) for i in np.where(A.sum(1) == 0)[0][:5]))
    return A


def spatial(df, spec):
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, unit, time, N, T = _panel(df, spec, [y] + xs, extra=[c for c in (spec.get("lat"), spec.get("lon")) if c] if not spec.get("neighbours") else [])
    if spec.get("neighbours"):
        d = d.merge(df[[unit, spec["neighbours"]]].drop_duplicates(unit), on=unit, how="left") if spec["neighbours"] not in d.columns else d
    _balanced(d, N, T, "Spatial panel models")
    if N < 10:
        raise OESError("Spatial models need at least 10 units.")
    d = d.sort_values(["_ti", "_u"]).reset_index(drop=True)
    names = d.drop_duplicates("_u").sort_values("_u")[unit].astype(str).tolist()
    A = _weights(d, dict(spec, unit=unit), names); W = A / A.sum(1, keepdims=True)
    om = np.linalg.eigvals(W); k = len(xs)
    lo_b = (1 / om.real.min() if om.real.min() < 0 else -.99) + 1e-3; hi_b = 1 - 1e-3
    Y = d[y].to_numpy(float).reshape(T, N); X = np.stack([d[v].to_numpy(float).reshape(T, N) for v in xs], -1)
    twoway = lambda B: B - B.mean(1, keepdims=True) - B.mean(0, keepdims=True) + B.mean((0, 1))
    Ys = twoway(Y); Xs = np.stack([twoway(X[:, :, j]) for j in range(k)], -1)
    WY = Ys @ W.T; WX = np.stack([Xs[:, :, j] @ W.T for j in range(k)], -1)
    Z = Xs.reshape(-1, k); yv = Ys.ravel(); n_obs = N * T
    b_ols = np.linalg.lstsq(Z, yv, rcond=None)[0]; e = yv - Z @ b_ols; s2 = e @ e / n_obs; E = e.reshape(T, N)
    I_t = np.array([E[t] @ W @ E[t] / (E[t] @ E[t]) for t in range(T)]); I_pool = float((E * (E @ W.T)).sum() / (E ** 2).sum()); EI = -1 / (N - 1)
    tr = float(np.trace(W.T @ W + W @ W)); We = (E @ W.T).ravel(); Wy = WY.ravel(); WXb = WX.reshape(-1, k) @ b_ols
    M = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    D = float((WXb @ M(WXb)) / s2 + T * tr); g_err, g_lag = float(e @ We / s2), float(e @ Wy / s2)
    lm_err = g_err ** 2 / (T * tr); lm_lag = g_lag ** 2 / D
    rlm_err = (g_err - T * tr / D * g_lag) ** 2 / (T * tr - (T * tr) ** 2 / D); rlm_lag = (g_lag - g_err) ** 2 / (D - T * tr)
    ld = lambda r: float(np.sum(np.log(np.abs(1 - r * om))))
    const = -(n_obs / 2) * (np.log(2 * np.pi) + 1)
    def sar_fit(Zm):
        b0 = np.linalg.lstsq(Zm, yv, rcond=None)[0]; b1 = np.linalg.lstsq(Zm, Wy, rcond=None)[0]; e0, e1 = yv - Zm @ b0, Wy - Zm @ b1
        nll = lambda r: (n_obs / 2) * np.log(((e0 - r * e1) @ (e0 - r * e1)) / n_obs) - T * ld(r)
        r = optimize.minimize_scalar(nll, bounds=(lo_b, hi_b), method="bounded").x
        return b0 - r * b1, float(r), float(-nll(r) + const)
    WZs = WX.reshape(-1, k)
    def sem_fit():
        def parts(l):
            yt, Zt = yv - l * Wy, Z - l * WZs; b = np.linalg.lstsq(Zt, yt, rcond=None)[0]; return b, yt - Zt @ b
        nll = lambda l: (n_obs / 2) * np.log((lambda ee: ee @ ee)(parts(l)[1]) / n_obs) - T * ld(l)
        l = optimize.minimize_scalar(nll, bounds=(lo_b, hi_b), method="bounded").x
        return parts(l)[0], float(l), float(-nll(l) + const)
    def hess(fn, p0):
        h = 1e-4 * np.maximum(np.abs(p0), 1e-2); kk = len(p0); H = np.zeros((kk, kk))
        for i in range(kk):
            for j in range(i, kk):
                a = p0.copy(); a[i] += h[i]; a[j] += h[j]; b_ = p0.copy(); b_[i] += h[i]; b_[j] -= h[j]; c = p0.copy(); c[i] -= h[i]; c[j] += h[j]; d_ = p0.copy(); d_[i] -= h[i]; d_[j] -= h[j]
                H[i, j] = H[j, i] = (fn(a) - fn(b_) - fn(c) + fn(d_)) / (4 * h[i] * h[j])
        return np.linalg.inv(-H)
    def sar_ll(p, Zm):
        kk = Zm.shape[1]; b, r, ls2 = p[:kk], p[kk], p[kk + 1]; ee = yv - r * Wy - Zm @ b
        return -(n_obs / 2) * (np.log(2 * np.pi) + ls2) + T * ld(r) - ee @ ee / (2 * np.exp(ls2))
    def sem_ll(p):
        b, l, ls2 = p[:k], p[k], p[k + 1]; ee = (yv - l * Wy) - (Z - l * WZs) @ b
        return -(n_obs / 2) * (np.log(2 * np.pi) + ls2) + T * ld(l) - ee @ ee / (2 * np.exp(ls2))
    Zd = np.concatenate([Xs, WX], -1).reshape(-1, 2 * k)
    b_sar, r_sar, ll_sar = sar_fit(Z); b_sdm, r_sdm, ll_sdm = sar_fit(Zd); b_sem, l_sem, ll_sem = sem_fit()
    ll_ols = -(n_obs / 2) * (np.log(2 * np.pi) + 1 + np.log(s2))
    rs = yv - r_sar * Wy - Z @ b_sar; rd = yv - r_sdm * Wy - Zd @ b_sdm; ytm = yv - l_sem * Wy; ee = ytm - (Z - l_sem * WZs) @ b_sem
    V_sar = hess(lambda p: sar_ll(p, Z), np.r_[b_sar, r_sar, np.log(rs @ rs / n_obs)])
    V_sdm = hess(lambda p: sar_ll(p, Zd), np.r_[b_sdm, r_sdm, np.log(rd @ rd / n_obs)])
    V_sem = hess(sem_ll, np.r_[b_sem, l_sem, np.log(ee @ ee / n_obs)])
    se_ = lambda V, kk: np.sqrt(np.clip(np.diag(V), 0, None))[:kk + 1]
    lab = xs + ["W x " + v for v in xs]
    def cell(b, V, i, kk): return f"{b[i]:.3f} ({se_(V, kk)[i]:.3f})"
    trow = []
    for i, nm in enumerate(lab):
        trow.append([nm, f"{b_ols[i]:.3f}" if i < k else "", cell(b_sar, V_sar, i, k) if i < k else "", cell(b_sem, V_sem, i, k) if i < k else "", cell(b_sdm, V_sdm, i, 2 * k)])
    trow.append(["Spatial coefficient (rho or lambda)", "", f"{r_sar:.3f} ({se_(V_sar, k)[k]:.3f})", f"{l_sem:.3f} ({se_(V_sem, k)[k]:.3f})", f"{r_sdm:.3f} ({se_(V_sdm, 2 * k)[2 * k]:.3f})"])
    trow.append(["Log-likelihood", f"{ll_ols:.1f}", f"{ll_sar:.1f}", f"{ll_sem:.1f}", f"{ll_sdm:.1f}"])
    trow.append(["AIC", f"{2 * (k + 1) - 2 * ll_ols:.1f}", f"{2 * (k + 2) - 2 * ll_sar:.1f}", f"{2 * (k + 2) - 2 * ll_sem:.1f}", f"{2 * (2 * k + 2) - 2 * ll_sdm:.1f}"])
    lr = lambda a, b, df_: (2 * (a - b), stats.chi2.sf(2 * (a - b), df_))
    lrt = []
    for nm, (s_, p_) in [("SAR vs non-spatial (rho = 0)", lr(ll_sar, ll_ols, 1)), ("SEM vs non-spatial (lambda = 0)", lr(ll_sem, ll_ols, 1)), ("SDM vs SAR (theta = 0)", lr(ll_sdm, ll_sar, k)), ("SDM vs SEM (common-factor restriction)", lr(ll_sdm, ll_sem, k))]:
        lrt.append(test_row(nm, s_, p_, "likelihood ratio"))
    diag = [test_row(f"Moran's I of residuals (pooled {I_pool:.3f}; expected {EI:.3f})", I_pool, float("nan"), f"positive in {int((I_t > 0).sum())} of {T} periods"),
            test_row("LM error", lm_err, stats.chi2.sf(lm_err, 1)), test_row("LM lag", lm_lag, stats.chi2.sf(lm_lag, 1)),
            test_row("Robust LM error", rlm_err, stats.chi2.sf(rlm_err, 1)), test_row("Robust LM lag", rlm_lag, stats.chi2.sf(rlm_lag, 1))]
    # direct, indirect and total effects with simulated standard errors
    rng = np.random.default_rng(int(spec.get("seed") or 0)); I_N = np.eye(N); Rs = int_option(spec, "sims", 500, 100, 3000, "Simulation draws")
    def effects(beta, theta, rho):
        S = np.linalg.inv(I_N - rho * W); out = []
        for j in range(k):
            Sk = S @ (beta[j] * I_N + theta[j] * W); dr = np.trace(Sk) / N; tot = Sk.sum() / N; out.append((dr, tot - dr, tot))
        return np.array(out)
    erows = []
    for nm, bb, th, rho, V, kk in [("SAR", b_sar, np.zeros(k), r_sar, V_sar, k), ("SDM", b_sdm[:k], b_sdm[k:], r_sdm, V_sdm, 2 * k)]:
        est_e = effects(bb, th, rho)
        mean = np.r_[b_sar if kk == k else b_sdm, rho]; cov = V[:kk + 1, :kk + 1]; draws = rng.multivariate_normal(mean, cov, Rs); sims = []
        for dr in draws:
            if lo_b < dr[-1] < hi_b:
                sims.append(effects(dr[:k], dr[k:2 * k] if kk == 2 * k else np.zeros(k), dr[-1]))
        sd = np.array(sims).std(0)
        for j, v in enumerate(xs):
            erows.append([nm, v] + [f"{est_e[j, m]:.3f} (t = {est_e[j, m] / sd[j, m]:.1f})" for m in range(3)])
    tabs = [table("Spatial weights", ["Item", "Value"], [["Units", N], ["Neighbour pairs", int(A.sum() / 2)], ["Neighbours per unit: mean, min, max", f"{A.sum(1).mean():.1f}, {int(A.sum(1).min())}, {int(A.sum(1).max())}"],
                                                         ["Bounds for the spatial coefficient", f"{lo_b:.2f} to 1"]], "Row-standardised weights from the neighbour list (symmetrised) or from the nearest neighbours by great-circle distance."),
            table("Spatial dependence in the non-spatial fixed-effects residuals", ["Test", "Statistic", "p-value", "Detail"], [[r[0], r[1], r[2], r[3]] for r in diag], "Anselin et al. (1996) Lagrange multiplier tests, panel version."),
            table("Coefficients (standard errors) after removing unit and time effects", ["Variable", "Non-spatial", "SAR", "SEM", "SDM"], trow),
            tests_table(lrt, "Likelihood-ratio tests"),
            table("Direct, indirect and total effects (simulated t)", ["Model", "Variable", "Direct", "Indirect (spillover)", "Total"], erows,
                  f"In the SAR and SDM the coefficient is not the marginal effect: LeSage and Pace effects, {Rs} draws from the estimated distribution.")]
    ch = [chart("Log-likelihood by spatial coefficient (SAR)", np.linspace(lo_b, hi_b, 40), [("Log-likelihood", [ (lambda r: -((n_obs / 2) * np.log(((lambda a, b_: (a - r * b_) @ (a - r * b_))(yv - Z @ np.linalg.lstsq(Z, yv, rcond=None)[0], Wy - Z @ np.linalg.lstsq(Z, Wy, rcond=None)[0])) / n_obs) - T * ld(r)) + const)(r) for r in np.linspace(lo_b, hi_b, 40)])], "rho", "Log-likelihood")]
    summary = [stat("Observations", n_obs), stat("Units", N), stat("Periods", T), stat("SAR rho", r_sar, 3), stat("SEM lambda", l_sem, 3), stat("Moran's I (residuals)", I_pool, 3)]
    notes = ["Fixed effects are removed by two-way demeaning and the concentrated likelihood is maximised over the spatial coefficient (Lee and Yu 2010 style, without the bias correction). Standard errors come from the numerical Hessian."]
    return result("spatial", f"Spatial panel models: {y}", n_obs, summary, tabs, ch, notes)
