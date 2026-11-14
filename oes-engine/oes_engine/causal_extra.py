"""Event study with Sun-Abraham and sensitivity analysis, synthetic difference-in-differences, matching / IPW / doubly robust estimation and
shift-share (Bartik) instrumental variables. Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog, minimize

from .core import (OESError, as_list, chart, check_columns, distinct, f, float_option, int_option, need, numeric, result, stat, table, time_order)


def _panel(df, spec, y, extra=()):
    unit, tcol = need(spec, "unit", "unit identifier"), need(spec, "time", "time identifier")
    check_columns(df, [unit, tcol])
    d = numeric(df, [y] + list(extra), keep=[unit, tcol]); d = d.rename(columns={unit: "_u", tcol: "_t"})
    d["_u"] = d["_u"].astype(str); d["_t"] = pd.to_numeric(d["_t"], errors="coerce")
    if d["_t"].isna().any():
        raise OESError("The time identifier must be numeric (a year or a period number).")
    if d.duplicated(["_u", "_t"]).any():
        raise OESError("Each unit and period may appear only once.")
    return d.sort_values(["_u", "_t"]).reset_index(drop=True), unit, tcol


# ------------------------------------------------------------------------------------------------------------------
def _fe_dummies(dd):
    return pd.concat([pd.get_dummies(dd._u).astype(float), pd.get_dummies(dd._t, drop_first=True).astype(float)], axis=1)


def _twfe_es(dd, ycol, lo, hi):
    lc = dd.l.clip(lo, hi); ev = [l for l in range(lo, hi + 1) if l != -1]
    X = pd.concat([_fe_dummies(dd)] + [pd.Series(((dd.g > 0) & (lc == l)).astype(float), name=l) for l in ev], axis=1)
    b = np.linalg.lstsq(X.values, dd[ycol].values, rcond=None)[0]
    return pd.Series(b[-len(ev):], index=ev)


def _sa_es(dd, ycol, cohorts, window):
    names, cols = [], [_fe_dummies(dd)]
    for g in cohorts:
        for l in range(int(dd._t.min() - g), int(dd._t.max() - g) + 1):
            if l == -1:
                continue
            cols.append(pd.Series(((dd.g == g) & (dd.l == l)).astype(float), name=(g, l))); names.append((g, l))
    X = pd.concat(cols, axis=1)
    b = pd.Series(np.linalg.lstsq(X.values, dd[ycol].values, rcond=None)[0][-len(names):], index=pd.MultiIndex.from_tuples(names))
    n = dd[dd.g > 0].groupby("g")._u.nunique(); out = {}
    for l in window:
        if l == -1:
            continue
        gs = [g for g in cohorts if (g, l) in b.index and n.get(g, 0) > 0]
        if not gs:
            continue
        wt = np.array([n[g] for g in gs], float); wt /= wt.sum(); out[l] = sum(w_ * b[(g, l)] for w_, g in zip(wt, gs))
    return pd.Series(out)


def eventstudy(df, spec):
    y = need(spec, "y", "outcome"); treat = need(spec, "treat", "treatment indicator (0/1)"); distinct([y], [treat])
    pre = int_option(spec, "pre", 5, 2, 12, "Pre-treatment periods"); post = int_option(spec, "post", 3, 1, 10, "Post-treatment periods")
    R = int_option(spec, "boot", 150, 50, 1000, "Bootstrap replications")
    d, unit, tcol = _panel(df, spec, y, [treat])
    first = d[d[treat] == 1].groupby("_u")._t.min()
    d["g"] = d._u.map(first).fillna(0).astype(int); d["l"] = np.where(d.g > 0, d._t - d.g, np.nan)
    cohorts = sorted(d.g[d.g > 0].unique())
    if not cohorts:
        raise OESError("No unit is ever treated.")
    if (d.g == 0).sum() == 0:
        raise OESError("A never-treated group is needed for the interaction-weighted estimator.")
    window = range(-pre, post + 1); units = sorted(d._u.unique()); size = d[d.g > 0].groupby("g")._u.nunique()
    b_tw = _twfe_es(d, y, -pre, post); b_sa = _sa_es(d, y, cohorts, window)
    rng = np.random.default_rng(int(spec.get("seed") or 0)); draws = []
    for _ in range(R):
        pick = rng.choice(np.array(units), len(units)); parts = []
        for k, s_ in enumerate(pick):
            x = d[d._u == s_].copy(); x["_u"] = f"b{k}"; parts.append(x)
        draws.append(_sa_es(pd.concat(parts, ignore_index=True), y, cohorts, window))
    draws = pd.DataFrame(draws).reindex(columns=b_sa.index); se = draws.std()
    rng_l = {int(g): (int(d._t.min() - g), int(d._t.max() - g)) for g in cohorts}
    ncoh = {l: int(sum(rng_l[int(g)][0] <= l <= rng_l[int(g)][1] for g in cohorts)) for l in b_sa.index}
    rows = [[int(l), f(b_tw.get(l)), f(b_sa[l]), f(se[l]), f(b_sa[l] / se[l], 3), f(2 * stats.norm.sf(abs(b_sa[l] / se[l])), 4), ncoh[l]] for l in b_sa.index]
    tabs = [table("Event-study coefficients (reference period: the year before adoption)", ["Event time", "Conventional TWFE", "Sun-Abraham", "Bootstrap s.e.", "t", "p-value", "Cohorts observed"], rows,
                  f"Interaction-weighted estimator with never-treated controls; bootstrap over units ({R} draws). End points of the conventional regression are binned.")]
    prel = [l for l in b_sa.index if l < -1]; postl = [l for l in b_sa.index if l >= 0]
    bp = b_sa[prel].values; Sp = np.atleast_2d(np.cov(draws[prel].values.T))
    wald = float(bp @ np.linalg.solve(Sp, bp)); pw = stats.chi2.sf(wald, len(prel))
    tau = np.array([l + 1 for l in prel], float); ncp = float(tau @ np.linalg.solve(Sp, tau)); crit = stats.chi2.ppf(.95, len(prel))
    slopes = np.linspace(0, 2 * max(np.abs(bp).max(), 1e-3), 600); power = 1 - stats.ncx2.cdf(crit, len(prel), slopes ** 2 * ncp)
    s80 = float(slopes[np.argmax(power >= .8)]) if (power >= .8).any() else float("nan"); slope_est = float((tau @ bp) / (tau @ tau))
    tabs.append(table("Pre-trend test and its power", ["Item", "Value"], [["Wald test that the pre-treatment coefficients are zero", f(wald, 3)], ["p-value", f(pw, 4)],
                                                                          ["Slope of a linear pre-trend detected with 80% power (per period)", f(s80, 4)], ["Least-squares slope of the estimated pre-coefficients", f(slope_est, 4)]]))
    # smoothness-bounded sensitivity (Rambachan and Roth 2023): second differences of the bias are at most M
    Ls = list(range(-pre, post + 1)); idx = {l: i for i, l in enumerate(Ls)}
    b_pre_full = np.array([b_sa[l] if l != -1 else 0.0 for l in Ls[:pre + 1]]); M_hat = float(np.abs(np.diff(b_pre_full, 2)).max()) if pre >= 2 else 0.0

    def bounds(k, M):
        n = len(Ls); c = np.zeros(n); c[idx[k]] = 1; box = [(None, None)] * n
        for l in Ls[:pre + 1]:
            box[idx[l]] = (0.0, 0.0) if l == -1 else (b_sa[l] - 1.96 * se[l], b_sa[l] + 1.96 * se[l])
        A, bb = [], []
        for l in Ls[1:-1]:
            row = np.zeros(n); row[idx[l + 1]] += 1; row[idx[l]] -= 2; row[idx[l - 1]] += 1; A += [row, -row]; bb += [M, M]
        r1 = linprog(c, A_ub=A, b_ub=bb, bounds=box); r2 = linprog(-c, A_ub=A, b_ub=bb, bounds=box)
        if r1.status != 0 or r2.status != 0:
            return np.nan, np.nan
        return r1.fun, -r2.fun
    srows = []
    for mult in (0, .5, 1, 2):
        for k in [l for l in postl if l <= 2]:
            blo, bhi = bounds(k, mult * M_hat); lo_set, hi_set = b_sa[k] - bhi, b_sa[k] - blo
            lo_ci, hi_ci = lo_set - 1.96 * se[k], hi_set + 1.96 * se[k]
            srows.append([mult, int(k), f(b_sa[k]), f(lo_set), f(hi_set), f(lo_ci), f(hi_ci), "yes" if (lo_ci > 0 or hi_ci < 0) else "no"])
    tabs.append(table("Sensitivity to violations of parallel trends (smoothness bound M = multiple of the largest pre-trend curvature)", ["M / M_hat", "Event time", "Estimate", "Set low", "Set high", "CI low", "CI high", "Excludes 0"], srows,
                      f"M_hat = {M_hat:.4f}. M = 0 allows only a linear post-treatment trend continuing the pre-trend; larger M allows more curvature."))
    ls = list(b_sa.index)
    ch = [chart("Event-study coefficients", ls, [("Sun-Abraham", b_sa.values), ("Conventional TWFE", b_tw.reindex(ls).values)], "Periods since adoption", "Effect", kind="line",
                band=(b_sa.values - 1.96 * se.values, b_sa.values + 1.96 * se.values))]
    summary = [stat("Observations", len(d)), stat("Units", len(units)), stat("Cohorts", len(cohorts)), stat("Pre-trend Wald p-value", pw, 4),
               stat("Average post effect", float(b_sa[postl].mean()), 4)]
    notes = ["Event study with the Sun and Abraham (2021) interaction-weighted estimator, cohort-by-event-time effects averaged with cohort-size weights, never-treated units as controls.",
             "Pre-trend power and smoothness-bounded sensitivity follow Roth (2022) and Rambachan and Roth (2023)."]
    return result("eventstudy", f"Event study: {y}", len(d), summary, tabs, ch, notes)


# ------------------------------------------------------------------------------------------------------------------
def _simplex(A, b, ridge, intercept):
    n = A.shape[1]

    def fun(x):
        r = A @ x[:n] + (x[n] if intercept else 0) - b
        return r @ r + ridge * (x[:n] @ x[:n])
    x0 = np.r_[np.ones(n) / n, 0.0] if intercept else np.ones(n) / n
    r = minimize(fun, x0, method="SLSQP", bounds=[(0, 1)] * n + ([(None, None)] if intercept else []),
                 constraints=[{"type": "eq", "fun": lambda x: x[:n].sum() - 1}], options={"maxiter": 1000, "ftol": 1e-12})
    return r.x[:n]


def _sdid(Yc, yt, T0, scale, zeta_mult=1.0, ntr=1):
    N0, T = Yc.shape; T1 = T - T0
    sig = np.std(np.diff(Yc[:, :T0], axis=1)); sig = sig if sig > 0 else 1.0
    zeta = zeta_mult * (ntr * T1) ** .25 * sig
    s_ = sig                                   # the data are divided by the noise level before the weights are fitted
    om = _simplex(Yc[:, :T0].T / s_, yt[:T0] / s_, (zeta / s_) ** 2 * T0, True)
    lam = _simplex(Yc[:, :T0] / s_, Yc[:, T0:].mean(1) / s_, (1e-6) ** 2 * N0, True)
    t_post, t_pre = yt[T0:].mean(), lam @ yt[:T0]; c_post, c_pre = om @ Yc[:, T0:].mean(1), om @ (Yc[:, :T0] @ lam)
    tau = (t_post - t_pre) - (c_post - c_pre); gap = (yt - t_pre) - (om @ Yc - c_pre)
    return tau, om, lam, gap


def _synth(Yc, yt, T0, scale):
    w = _simplex(Yc[:, :T0].T / scale, yt[:T0] / scale, 0.0, False); return (yt[T0:] - w @ Yc[:, T0:]).mean(), w


def sdid(df, spec):
    y = need(spec, "y", "outcome"); treat = need(spec, "treat", "treatment indicator (0/1)"); distinct([y], [treat])
    d, unit, tcol = _panel(df, spec, y, [treat])
    Y = d.pivot(index="_u", columns="_t", values=y); D = d.pivot(index="_u", columns="_t", values=treat)
    if Y.isna().any().any():
        raise OESError("Synthetic DiD needs a balanced panel: every unit observed in every period.")
    years = list(Y.columns); treated = [u for u in Y.index if D.loc[u].max() == 1]
    if not treated or len(treated) >= len(Y) - 2:
        raise OESError("Need at least one treated unit and at least three control units.")
    t_first = min(D.loc[treated[0]][D.loc[treated[0]] == 1].index); T0 = sum(t < t_first for t in years)
    if any(D.loc[u][D.loc[u] == 1].index.min() != t_first for u in treated):
        raise OESError("Synthetic DiD needs a common adoption period for all treated units (use the staggered DiD model otherwise).")
    if T0 < 3 or len(years) - T0 < 1:
        raise OESError("Need at least three pre-treatment periods and one post-treatment period.")
    ctrl = [u for u in Y.index if u not in treated]; Yc = Y.loc[ctrl].values.astype(float); yt = Y.loc[treated].values.astype(float).mean(0); N0 = len(ctrl); ntr = len(treated)
    scale = float(np.std(Y.values)) or 1.0
    did = (yt[T0:].mean() - yt[:T0].mean()) - (Yc[:, T0:].mean() - Yc[:, :T0].mean())
    sc_est, w_sc = _synth(Yc, yt, T0, scale); tau, om, lam, gap = _sdid(Yc, yt, T0, scale, 1.0, ntr)
    nrep = int_option(spec, "placebos", 100, 20, 500, "Placebo draws"); rng = np.random.default_rng(int(spec.get("seed") or 0))
    plac = []
    sets = [[k] for k in range(N0)] if ntr == 1 else [list(rng.choice(N0, ntr, replace=False)) for _ in range(nrep)]
    for S in sets[:max(nrep, N0 if ntr == 1 else nrep)]:
        keep = [i for i in range(N0) if i not in S]
        if len(keep) < 3:
            continue
        ytp = Yc[S].mean(0); Yp = Yc[keep]
        d_ = (ytp[T0:].mean() - ytp[:T0].mean()) - (Yp[:, T0:].mean() - Yp[:, :T0].mean())
        plac.append((d_, _synth(Yp, ytp, T0, scale)[0], _sdid(Yp, ytp, T0, scale, 1.0, ntr)[0]))
    plac = np.array(plac); est = np.array([did, sc_est, tau]); pse = plac.std(0, ddof=1)
    rows = [[n_, f(e), f(s_), f(e / s_, 3), f(float((np.abs(plac[:, j]) >= abs(e)).mean()), 3)] for j, (n_, e, s_) in enumerate(zip(["Difference-in-differences", "Synthetic control", "Synthetic DiD"], est, pse))]
    tabs = [table("Estimates and placebo inference", ["Estimator", "Effect", "Placebo s.e.", "t", "Share of placebos at least as large"], rows,
                  f"{len(plac)} placebo assignments of the treatment to control units; the placebo standard error is approximate and probably understated when the treated unit is large."),
            table("Largest unit weights", ["Unit", "SDID weight", "Synthetic-control weight"], [[ctrl[i], f(om[i], 4), f(w_sc[i], 4)] for i in np.argsort(-om)[:8]],
                  f"Effective number of units: {1 / np.sum(om ** 2):.1f} (SDID), {1 / np.sum(w_sc ** 2):.1f} (synthetic control)."),
            table("Time weights (pre-treatment periods)", ["Period", "Weight"], [[str(int(t)), f(w, 4)] for t, w in zip(years[:T0], lam)]),
            table("Gap by period (treated minus its synthetic path)", ["Period", "Gap"], [[str(int(t)), f(g_)] for t, g_ in zip(years, gap)], "Pre-treatment gaps are measured relative to the time-weighted pre-treatment level.")]
    rob = []
    for mult in (.25, 1, 4, 16):
        t_, o_, l_, _ = _sdid(Yc, yt, T0, scale, mult, ntr); rob.append([f"regularisation x {mult:g}", f(t_), f(1 / np.sum(o_ ** 2), 2), f(l_.max(), 3)])
    tabs.append(table("Sensitivity to the regularisation", ["Specification", "SDID estimate", "Effective units", "Largest time weight"], rob))
    cf = om @ Yc + (yt[:T0] @ lam - om @ (Yc[:, :T0] @ lam))
    ch = [chart("Treated against the synthetic DiD counterfactual", [str(int(t)) for t in years], [("Treated", yt), ("Counterfactual", cf)], "", y),
          chart("Gap by period", [str(int(t)) for t in years], [("Gap", gap)], "", "Effect", kind="bar")]
    summary = [stat("Units", len(Y)), stat("Treated units", ntr), stat("Pre-treatment periods", T0), stat("Synthetic DiD effect", tau, 4), stat("Placebo s.e.", pse[2], 4)]
    notes = ["Synthetic difference-in-differences (Arkhangelsky et al. 2021): unit weights and time weights on the simplex with an intercept, ridge penalty on the unit weights, weighted DiD.",
             "Time weights often concentrate on the last pre-treatment period; check the sensitivity and the pre-treatment gaps."]
    return result("sdid", f"Synthetic difference-in-differences: {y}", len(d), summary, tabs, ch, notes)


# ------------------------------------------------------------------------------------------------------------------
def _pscore(Xm, D):
    import statsmodels.api as sm
    m = sm.Logit(D, sm.add_constant(Xm)).fit(disp=0, maxiter=200); return m.predict(sm.add_constant(Xm)), m


def _match_estimators(y, D, Xm, trim=None):
    import statsmodels.api as sm
    e, _ = _pscore(Xm, D)
    keep = np.ones(len(y), bool) if trim is None else (e > trim) & (e < 1 - trim)
    y, D, Xm, e = y[keep], D[keep], Xm[keep], e[keep]; out = {}
    t = np.where(D == 1)[0]; c = np.where(D == 0)[0]
    idx = np.abs(e[t][:, None] - e[c][None, :]).argmin(1); out["Matching (1 nearest neighbour), ATT"] = (y[t] - y[c][idx]).mean()
    w1 = D / e; w0 = (1 - D) / (1 - e); out["IPW, ATE"] = (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()
    wa = np.where(D == 1, 1.0, e / (1 - e)); out["IPW, ATT"] = y[D == 1].mean() - np.average(y[D == 0], weights=wa[D == 0])
    Z = sm.add_constant(Xm); m1 = sm.OLS(y[D == 1], Z[D == 1]).fit(); m0 = sm.OLS(y[D == 0], Z[D == 0]).fit(); mu1, mu0 = m1.predict(Z), m0.predict(Z)
    out["Regression adjustment, ATE"] = (mu1 - mu0).mean(); out["Regression adjustment, ATT"] = (y[D == 1] - mu0[D == 1]).mean()
    psi = mu1 - mu0 + D * (y - mu1) / e - (1 - D) * (y - mu0) / (1 - e); out["AIPW (doubly robust), ATE"] = psi.mean()
    out["AIPW (doubly robust), ATT"] = (D * (y - mu0) - (1 - D) * e / (1 - e) * (y - mu0)).sum() / D.sum()
    return out


def _smd(x, D, w=None):
    w = np.ones(len(x)) if w is None else w
    m1 = np.average(x[D == 1], weights=w[D == 1]); m0 = np.average(x[D == 0], weights=w[D == 0])
    return (m1 - m0) / np.sqrt((x[D == 1].var(ddof=1) + x[D == 0].var(ddof=1)) / 2)


def matching(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "outcome"); treat = need(spec, "treat", "treatment indicator (0/1)"); x = as_list(spec.get("x")); distinct([y], [treat], x)
    if not x:
        raise OESError("Choose at least one covariate.")
    d, labels = numeric(df, [y, treat] + x), None
    if not set(d[treat].unique()) <= {0, 1}:
        raise OESError("The treatment must be a 0/1 variable.")
    yv = d[y].to_numpy(float); D = d[treat].to_numpy(int); Xm = d[x].to_numpy(float)
    if D.sum() < 20 or (1 - D).sum() < 20:
        raise OESError("Need at least 20 treated and 20 untreated observations.")
    B = int_option(spec, "boot", 200, 50, 1000, "Bootstrap replications"); e, lg = _pscore(Xm, D)
    est = _match_estimators(yv, D, Xm); rng = np.random.default_rng(int(spec.get("seed") or 0)); boot = []
    for _ in range(B):
        ix = rng.integers(0, len(yv), len(yv))
        try:
            boot.append(_match_estimators(yv[ix], D[ix], Xm[ix]))
        except Exception:
            continue
    boot = pd.DataFrame(boot); se = boot.std(ddof=1)
    rows = [[k, f(v), f(se[k]), f(v - 1.96 * se[k]), f(v + 1.96 * se[k])] for k, v in est.items()]
    tabs = [table("Treatment-effect estimates", ["Estimator", "Estimate", "Bootstrap s.e.", "95% CI low", "95% CI high"], rows,
                  f"Raw difference in means: {yv[D == 1].mean() - yv[D == 0].mean():.4f}. The propensity score is re-estimated in every bootstrap draw ({len(boot)} draws)."),
            table("Propensity score", ["Statistic", "Treated", "Untreated"], [["Mean", f(e[D == 1].mean(), 4), f(e[D == 0].mean(), 4)], ["Minimum", f(e[D == 1].min(), 4), f(e[D == 0].min(), 4)], ["Maximum", f(e[D == 1].max(), 4), f(e[D == 0].max(), 4)]])]
    w_att = np.where(D == 1, 1.0, e / (1 - e))
    tabs.append(table("Covariate balance (standardised mean differences)", ["Covariate", "Raw", "After ATT weighting"], [[c, f(_smd(d[c].to_numpy(float), D), 3), f(_smd(d[c].to_numpy(float), D, w_att), 3)] for c in x],
                      f"|value| below 0.1 is the usual threshold. Effective number of untreated units after weighting: {w_att[D == 0].sum() ** 2 / (w_att[D == 0] ** 2).sum():.0f} of {(D == 0).sum()}; largest weight {w_att[D == 0].max():.2f}."))
    trows = []
    for tr in (None, .02, .05, .10):
        try:
            o = _match_estimators(yv, D, Xm, tr)
            trows.append(["no trimming" if tr is None else f"scores outside [{tr}, {1 - tr}] dropped", f(o["IPW, ATE"]), f(o["AIPW (doubly robust), ATE"]), f(o["Matching (1 nearest neighbour), ATT"])])
        except Exception:
            continue
    tabs.append(table("Sensitivity to trimming", ["Sample", "IPW ATE", "AIPW ATE", "Matching ATT"], trows))
    ols = sm.OLS(yv, sm.add_constant(np.column_stack([D, Xm]))).fit(); tstat = ols.tvalues[1]; fq = abs(tstat) / np.sqrt(ols.df_resid); rv = .5 * (np.sqrt(fq ** 4 + 4 * fq ** 2) - fq ** 2)
    tabs.append(table("Sensitivity to an omitted confounder (Cinelli and Hazlett 2020)", ["Item", "Value"], [["Regression estimate", f(ols.params[1])], ["t statistic", f(tstat, 2)], ["Robustness value", f(rv, 4)]],
                      "An omitted confounder would have to explain at least this share of the residual variance of both treatment and outcome to bring the estimate to zero."))
    ch = [chart("Propensity score by treatment status", np.round(np.linspace(0, 1, 21), 2), [("Untreated", np.histogram(e[D == 0], bins=np.linspace(0, 1, 22), density=True)[0]), ("Treated", np.histogram(e[D == 1], bins=np.linspace(0, 1, 22), density=True)[0])], "Propensity score", "Density", kind="bar")]
    summary = [stat("Observations", len(d)), stat("Treated", int(D.sum())), stat("AIPW ATE", est["AIPW (doubly robust), ATE"], 4), stat("AIPW ATT", est["AIPW (doubly robust), ATT"], 4), stat("Robustness value", rv, 4)]
    notes = ["Selection on observables: the estimates are valid if treatment is as good as random given the covariates (unconfoundedness) and treated and untreated units overlap.", "Balance is necessary, not sufficient."]
    return result("matching", f"Matching, IPW and doubly robust estimation: {y}", len(d), summary, tabs, ch, notes)


# ------------------------------------------------------------------------------------------------------------------
def _tsls(y, x, z, W=None):
    n = len(y); W = np.ones((n, 1)) if W is None else np.column_stack([np.ones(n), W])
    Z = np.column_stack([W, z]); X = np.column_stack([W, x]); xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    b = np.linalg.lstsq(xh, y, rcond=None)[0]; u = y - X @ b; A = np.linalg.inv(xh.T @ xh); V = A @ (xh.T * u ** 2) @ xh @ A
    return b[-1], np.sqrt(V[-1, -1])


def bartik(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "outcome"); x = need(spec, "x", "endogenous variable"); shares = as_list(spec.get("shares")); controls = as_list(spec.get("controls")); distinct([y], [x], shares, controls)
    if len(shares) < 5:
        raise OESError("Choose at least five share columns (one per industry).")
    raw = spec.get("shocks"); 
    try:
        shocks = np.array([float(v) for v in (raw if isinstance(raw, list) else str(raw).replace(";", ",").split(","))])
    except Exception:
        raise OESError("Enter the industry shocks as a comma-separated list of numbers.")
    if len(shocks) != len(shares):
        raise OESError(f"{len(shares)} share columns need {len(shares)} shocks (found {len(shocks)}).")
    d = numeric(df, [y, x] + shares + controls); S = d[shares].to_numpy(float); yv = d[y].to_numpy(float); xv = d[x].to_numpy(float); W = d[controls].to_numpy(float) if controls else None
    B = S @ shocks; L, N = S.shape; sn = S.sum(0) / S.sum()
    Zc = np.column_stack([np.ones(L), B] + ([W] if W is not None else [])); fs = sm.OLS(xv, Zc).fit(cov_type="HC1")
    ols = sm.OLS(yv, np.column_stack([np.ones(L), xv] + ([W] if W is not None else []))).fit(cov_type="HC1")
    b_iv, se_iv = _tsls(yv, xv, B, W); rf = sm.OLS(yv, Zc).fit(cov_type="HC1")
    # Rotemberg weights (without controls partialled out when none are given)
    Zs = S - S.mean(0); xt = xv - xv.mean(); yt = yv - yv.mean()
    if W is not None:
        M = np.eye(L) - np.column_stack([np.ones(L), W]) @ np.linalg.pinv(np.column_stack([np.ones(L), W])); Zs = M @ S; xt = M @ xv; yt = M @ yv
    b_n = (Zs.T @ yt) / (Zs.T @ xt); alpha = shocks * (Zs.T @ xt); alpha = alpha / alpha.sum()
    order = np.argsort(-np.abs(alpha)); top = order[:10]
    # industry-level (Borusyak, Hull and Jaravel) standard error
    s_n = S.sum(0); y_n = (S * yv[:, None]).sum(0) / s_n; x_n = (S * xv[:, None]).sum(0) / s_n; Wn = np.sqrt(s_n)
    Zm = np.column_stack([np.ones(N), shocks]) * Wn[:, None]; Xm_ = np.column_stack([np.ones(N), x_n]) * Wn[:, None]; ym = y_n * Wn
    xh = Zm @ np.linalg.lstsq(Zm, Xm_, rcond=None)[0]; bb = np.linalg.lstsq(xh, ym, rcond=None)[0]; u = ym - Xm_ @ bb
    A = np.linalg.inv(xh.T @ xh); V = A @ (xh.T * u ** 2) @ xh @ A; se_bhj = float(np.sqrt(V[1, 1]))
    tabs = [table("Estimates", ["Estimator", "Coefficient", "Std. err."], [["OLS", f(ols.params[1]), f(ols.bse[1])], ["2SLS with the shift-share instrument (region-robust s.e.)", f(b_iv), f(se_iv)],
                                                                          ["Industry-level IV (shock-level robust s.e.)", f(bb[1]), f(se_bhj)], ["Reduced form", f(rf.params[1]), f(rf.bse[1])]],
                  "With controls, the industry-level coefficient is not identical to the region-level 2SLS."),
            table("First stage", ["Statistic", "Value"], [["Coefficient on the instrument", f(fs.params[1])], ["t statistic", f(fs.tvalues[1], 2)], ["F statistic", f(fs.tvalues[1] ** 2, 2)], ["R-squared", f(fs.rsquared, 3)]]),
            table("Rotemberg weights: industries with the largest absolute weights", ["Industry", "Weight", "Shock", "Industry-specific IV estimate"], [[shares[i], f(alpha[i], 4), f(shocks[i], 4), f(b_n[i], 4)] for i in top],
                  f"Share of absolute weight on the top 5: {np.abs(alpha[order[:5]]).sum() / np.abs(alpha).sum():.0%}; negative weights: {int((alpha < 0).sum())} industries ({alpha[alpha < 0].sum():.3f} of total). Effective number of industries: {1 / np.sum(sn ** 2):.1f}.")]
    rows = []
    for k in (0, 1, 3, 5):
        keep = [i for i in range(N) if i not in list(order[:k])]; Bk = S[:, keep] @ shocks[keep]; bk, sk = _tsls(yv, xv, Bk, W); rows.append([f"drop top {k} industries" if k else "all industries", f(bk), f(sk)])
    tabs.append(table("Leave-out sensitivity (drop the highest-weight industries from the instrument)", ["Instrument", "2SLS", "Std. err."], rows))
    ch = [chart("Rotemberg weights (top 10)", [shares[i] for i in top], [("Weight", alpha[top])], "", "Weight", kind="bar")]
    summary = [stat("Regions", L), stat("Industries", N), stat("First-stage F", fs.tvalues[1] ** 2, 1), stat("2SLS", b_iv, 4), stat("OLS", ols.params[1], 4), stat("Shock-level s.e.", se_bhj, 4)]
    notes = ["Shift-share instrument: initial industry shares times national industry shocks. Identification rests on share exogeneity (Goldsmith-Pinkham, Sorkin and Swift 2020) or shock exogeneity (Borusyak, Hull and Jaravel 2022); the diagnostics differ.",
             "Report which assumption you rely on and use the shock-level standard error."]
    return result("bartik", f"Shift-share IV: {y} on {x}", L, summary, tabs, ch, notes)
