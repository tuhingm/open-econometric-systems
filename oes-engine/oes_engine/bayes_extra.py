"""Bayesian local projections, sign-restricted Bayesian VAR, Bayesian hierarchical panel regression and Markov switching with
time-varying transition probabilities. Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import stats

from .core import (OESError, as_list, chart, check_columns, distinct, f, int_option, float_option, need, result, stat, table,
                   numeric, time_order)
from .timeseries import _series_frame


# ---------------------------------------------------------------------------------------------------------------
# Bayesian local projections with a smoothness (random-walk across horizons) prior, tightness chosen by marginal likelihood
# ---------------------------------------------------------------------------------------------------------------
def _lp_estimates(Z, ir, isk, ctrl_idx, contemp_idx, P, H):
    T = Z.shape[0]; th, se = [], []
    for h in range(1, H + 1):
        rows = np.arange(P, T - h)
        if len(rows) < 12 + Z.shape[1] * P:
            raise OESError("Too few observations for this horizon and number of lags.")
        y = Z[rows + h, ir]; x = Z[rows, isk]
        parts = ([Z[rows][:, contemp_idx]] if contemp_idx else []) + [Z[rows - l] for l in range(1, P + 1)] + [np.ones(len(rows))]
        Xc = np.column_stack([p if p.ndim > 1 else p[:, None] for p in parts])
        M = np.eye(len(rows)) - Xc @ np.linalg.pinv(Xc)
        xr, yr = M @ x, M @ y; b = xr @ yr / (xr @ xr); u = yr - b * xr
        g = xr * u; S = g @ g
        for l in range(1, h + 2):
            S += 2 * (1 - l / (h + 2)) * (g[l:] @ g[:-l])
        th.append(b); se.append(np.sqrt(S) / (xr @ xr))
    return np.array(th), np.array(se)


def _prior_cov(tau, s0, H):
    i = np.arange(H); return s0 ** 2 + tau ** 2 * np.minimum.outer(i, i)


def _marg_ll(th, se, tau, s0):
    C = _prior_cov(tau, s0, len(th)) + np.diag(se ** 2); s, ld = np.linalg.slogdet(C)
    return -.5 * (ld + th @ np.linalg.solve(C, th))


def _post(th, se, tau, s0):
    A = _prior_cov(tau, s0, len(th)); K = A @ np.linalg.inv(A + np.diag(se ** 2)); return K @ th, (np.eye(len(th)) - K) @ A


def blp(df, spec):
    y = need(spec, "y", "response variable"); shock = need(spec, "shock", "shock variable")
    contemp = as_list(spec.get("contemp")); controls = as_list(spec.get("controls"))
    distinct([y], [shock], contemp, controls)
    P = int_option(spec, "lags", 4, 1, 12, "Lags"); H = int_option(spec, "horizon", 16, 4, 40, "Horizon")
    cols_ = [y, shock] + contemp + controls
    d, labels = _series_frame(df, spec, cols_)
    if len(d) < 40:
        raise OESError("Local projections need at least 40 observations.")
    Z = d[cols_].to_numpy(float); ir, isk = 0, 1; cidx = [0] + list(range(2, 2 + len(contemp)))   # the response variable's own current value is ordered before the shock
    th, se = _lp_estimates(Z, ir, isk, None, cidx, P, H)
    s0 = spec.get("s0"); s0 = float(s0) if s0 not in (None, "") else 3 * max(np.abs(th).max(), 1e-6)
    tau_in = spec.get("tau")
    grid = np.logspace(-3, 1, 80) * s0 / 10
    ll = np.array([_marg_ll(th, se, t, s0) for t in grid])
    tau = float(tau_in) if tau_in not in (None, "") else float(grid[ll.argmax()])
    m, C = _post(th, se, tau, s0); sd = np.sqrt(np.diag(C))
    rng = np.random.default_rng(int(spec.get("seed") or 0)); dr = rng.multivariate_normal(m, C + 1e-12 * np.eye(H), 10000)
    hs = np.arange(1, H + 1); lo9, hi9 = np.percentile(dr, [5, 95], 0); lo, hi = np.percentile(dr, [16, 84], 0)
    rows = [[int(h), f(a), f(b), f(c), f(l9), f(h9), f(p, 3)] for h, a, b, c, l9, h9, p in
            zip(hs, th, se, m, lo9, hi9, (dr > 0).mean(0))]
    tabs = [table("Impulse response: local projection and Bayesian posterior",
                  ["Horizon", "LP estimate", "LP s.e. (Newey-West)", "Posterior mean", "90% set low", "90% set high", "P(response > 0)"], rows,
                  "The posterior combines each horizon's local projection with a random-walk prior on the response across horizons; the horizon estimates are treated as independent."),
            table("Prior", ["Item", "Value"], [["Prior sd of the horizon-1 response (s0)", f(s0, 4)], ["Smoothness tau" + (" (given)" if tau_in not in (None, "") else " (marginal likelihood)"), f(tau, 5)],
                                              ["Log marginal likelihood at tau", f(ll.max() if tau_in in (None, "") else _marg_ll(th, se, tau, s0), 3)],
                                              ["Log marginal likelihood, almost no smoothing (tau large)", f(ll[-1], 3)], ["Log marginal likelihood, constant response (tau small)", f(ll[0], 3)]])]
    ch = [chart(f"Response of {y} to {shock}", hs, [("Bayesian LP (posterior mean)", m), ("Local projection", th)], "Horizon", "Response", band=(lo9, hi9))]
    pk = int(np.argmax(np.abs(m)))
    summary = [stat("Observations", len(d)), stat("Horizons", H), stat("Smoothness tau", tau, 4), stat("Peak posterior response", m[pk], 4), stat("Peak at horizon", int(hs[pk]))]
    notes = ["Bayesian local projections (Miranda-Agrippino and Ricco 2021; Barnichon and Brownlees 2019): one regression per horizon with the current value of the response variable and the contemporaneous controls (ordered before the shock) and lags, "
             "then a random-walk prior across horizons. The tightness is chosen by maximising the marginal likelihood unless given."]
    return result("blp", f"Bayesian local projections: {y} on {shock}", len(d), summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Sign-restricted Bayesian VAR (natural-conjugate Minnesota-type prior, rotations drawn uniformly)
# ---------------------------------------------------------------------------------------------------------------
def _design(Z, P):
    return Z[P:], np.column_stack([Z[P - l:len(Z) - l] for l in range(1, P + 1)] + [np.ones(len(Z) - P)])


def _niw(Z, P, lam):
    n = Z.shape[1]; Yd, Xd = _design(Z, P); T, k = Xd.shape
    s2 = np.array([np.var(np.diff(Z[:, i])) for i in range(n)])
    B0 = np.zeros((k, n)); B0[:n, :] = np.diag([.9 if np.var(np.diff(Z[:, i])) < .6 * np.var(Z[:, i]) else 0.0 for i in range(n)])
    om = np.r_[[lam ** 2 / (l ** 2 * s2[j]) for l in range(1, P + 1) for j in range(n)], 1e4]
    Oi = np.diag(1 / om); nu0 = n + 2; S0 = np.diag(s2) * (nu0 - n - 1)
    Om1 = np.linalg.inv(Oi + Xd.T @ Xd); B1 = Om1 @ (Oi @ B0 + Xd.T @ Yd)
    S1 = S0 + Yd.T @ Yd + B0.T @ Oi @ B0 - B1.T @ np.linalg.inv(Om1) @ B1
    return B1, Om1, (S1 + S1.T) / 2, nu0 + T


def _phis(B, P, n, H):
    A = [B[l * n:(l + 1) * n, :].T for l in range(P)]; Phi = [np.eye(n)]
    for h in range(1, H + 1):
        Phi.append(sum(A[l] @ Phi[h - 1 - l] for l in range(min(h, P))))
    return np.array(Phi)


def _haar(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n))); return q * np.sign(np.diag(r))


def sign_bvar(df, spec):
    vars_ = as_list(spec.get("vars")); shock_v = need(spec, "shock_var", "variable the shock moves on impact")
    neg = as_list(spec.get("neg")); pos = as_list(spec.get("pos"))
    if len(vars_) < 3:
        raise OESError("Choose at least three variables.")
    if shock_v not in vars_ or any(c not in vars_ for c in neg + pos):
        raise OESError("The shock variable and the restricted variables must be among the VAR variables.")
    if shock_v in neg or shock_v in pos:
        raise OESError("List the shock variable only in its own field: it is restricted to rise on impact.")
    if not (neg or pos):
        raise OESError("Choose at least one variable with a restricted sign.")
    P = int_option(spec, "lags", 4, 1, 8, "Lags"); K = int_option(spec, "restrict_horizon", 4, 1, 12, "Restriction horizon"); H = int_option(spec, "horizon", 20, 4, 40, "Horizon")
    nd = int_option(spec, "draws", 1000, 100, 5000, "Posterior draws"); lam = float_option(spec, "lambda1", 0.2, "Overall tightness"); tries = 100
    d, labels = _series_frame(df, spec, vars_); Z = d[vars_].to_numpy(float); n = len(vars_)
    if len(d) < 4 * n * P:
        raise OESError("Too few observations for this many variables and lags.")
    signs = {vars_.index(shock_v): +1}; signs.update({vars_.index(c): -1 for c in neg}); signs.update({vars_.index(c): +1 for c in pos})
    B1, Om1, S1, nu1 = _niw(Z, P, lam); L = np.linalg.cholesky(Om1)
    from scipy.stats import invwishart
    rng = np.random.default_rng(int(spec.get("seed") or 0)); keep, tried, parok = [], 0, 0; sj = vars_.index(shock_v)
    for i in range(nd):
        Sg = invwishart.rvs(df=nu1, scale=S1, random_state=rng); Pm = np.linalg.cholesky(Sg)
        Bd = B1 + L @ rng.standard_normal(B1.shape) @ Pm.T; Phi = _phis(Bd, P, n, H)
        for _ in range(tries):
            tried += 1; q = _haar(n, rng)[:, 0]; irf = np.array([Phi[h] @ Pm @ q for h in range(H + 1)])
            if irf[0, sj] < 0:
                irf = -irf
            if all((irf[:K, j] * s > 0).all() for j, s in signs.items()):
                keep.append(irf); parok += 1; break
    if len(keep) < 20:
        raise OESError("Fewer than 20 responses satisfy the restrictions. Relax the restrictions, shorten the restriction horizon or increase the draws.")
    ir = np.array(keep); med = np.median(ir, 0); lo, hi = np.percentile(ir, [16, 84], 0); lo9, hi9 = np.percentile(ir, [5, 95], 0)
    hsel = [h for h in [0, 1, 2, 4, 8, 12, 16, 20] if h <= H]
    rows = []
    for j, v in enumerate(vars_):
        for h in hsel:
            rows.append([f"{v}, horizon {h}", f(med[h, j]), f(lo[h, j]), f(hi[h, j]), f(lo9[h, j]), f(hi9[h, j]), f((ir[:, h, j] > 0).mean(), 3)])
    tabs = [table("Impulse responses to the identified shock (one standard deviation, rate normalised to rise on impact)",
                  ["Variable and horizon", "Median", "68% low", "68% high", "90% low", "90% high", "P(> 0)"], rows,
                  "Bands pool uncertainty about the reduced-form parameters and about the rotation; the pointwise median is not the response of a single model (Fry and Pagan 2011)."),
            table("Identification", ["Item", "Value"], [["Restrictions (signs for the first " + str(K) + " horizons)", ", ".join(f"{vars_[j]} {'+' if s > 0 else '-'}" for j, s in signs.items())],
                                                       ["Retained responses", len(keep)], ["Posterior draws with an admissible rotation", f(parok / nd, 3)], ["Share of rotations accepted", f(len(keep) / tried, 4)]])]
    ch = [chart(f"Response of {v}", range(H + 1), [("Median", med[:, j])], "Horizon", "", band=(lo9[:, j], hi9[:, j])) for j, v in enumerate(vars_)]
    summary = [stat("Observations", len(d)), stat("Retained responses", len(keep)), stat("Acceptance rate", len(keep) / tried, 4), stat("Lags", P)]
    notes = ["Sign-restricted Bayesian VAR (Uhlig 2005; Rubio-Ramirez, Waggoner and Zha 2010): natural-conjugate Minnesota-type prior, Haar-distributed rotations, restrictions imposed on impact and the following quarters."]
    return result("sign_bvar", f"Sign-restricted Bayesian VAR: shock on {shock_v}", len(d), summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Bayesian hierarchical panel regression (random intercepts and one random slope), Gibbs sampler with half-Cauchy spreads
# ---------------------------------------------------------------------------------------------------------------
def _gibbs_h(y, x, z, g, N, iters, seed, A_b, A_a, init):
    rng = np.random.default_rng(seed); n = len(y); th = np.zeros((N, 2)); mu = np.zeros(2); s2 = np.ones(2); a = np.ones(2); sig2 = 1.0
    k = z.shape[1]; gam = np.zeros(k); th += init * rng.standard_normal((N, 2)); mu = init * rng.standard_normal(2)
    members = [np.where(g == i)[0] for i in range(N)]; Xi = [np.column_stack([np.ones(len(ix)), x[ix]]) for ix in members]
    out = {"mu": [], "s": [], "gam": [], "sig": [], "th": []}
    for it in range(iters):
        u = y - z @ gam
        for i, ix in enumerate(members):
            V = np.linalg.inv(Xi[i].T @ Xi[i] / sig2 + np.diag(1 / s2)); m = V @ (Xi[i].T @ u[ix] / sig2 + mu / s2); th[i] = rng.multivariate_normal(m, V)
        for j in range(2):
            v = 1 / (N / s2[j] + 1 / 100); mu[j] = rng.normal(v * th[:, j].sum() / s2[j], np.sqrt(v))
        for j, A in enumerate([A_a, A_b]):
            ss = ((th[:, j] - mu[j]) ** 2).sum()
            s2[j] = 1 / rng.gamma((N + 1) / 2, 1 / (ss / 2 + 1 / a[j])); a[j] = 1 / rng.gamma(1, 1 / (1 / A ** 2 + 1 / s2[j]))
        if k:
            r = y - th[g, 0] - th[g, 1] * x; V = np.linalg.inv(z.T @ z / sig2 + np.eye(k) / 100); gam = rng.multivariate_normal(V @ (z.T @ r) / sig2, V)
        e = y - th[g, 0] - th[g, 1] * x - (z @ gam if k else 0)
        sig2 = 1 / rng.gamma(.01 + n / 2, 1 / (.01 + e @ e / 2))
        out["mu"].append(mu.copy()); out["s"].append(np.sqrt(s2)); out["gam"].append(gam.copy()); out["sig"].append(np.sqrt(sig2)); out["th"].append(th.copy())
    return {k_: np.array(v) for k_, v in out.items()}


def _rhat_ess(chains):
    c = np.vstack([ch[:len(ch) // 2] for ch in chains] + [ch[len(ch) // 2:2 * (len(ch) // 2)] for ch in chains]); m, n = c.shape
    B = n * c.mean(1).var(ddof=1); W = c.var(1, ddof=1).mean(); rh = np.sqrt(((n - 1) / n * W + B / n) / W) if W > 0 else np.nan
    x_ = c - c.mean(1, keepdims=True); rs = 0.0
    for l in range(1, min(n // 2, 200)):
        rho = np.mean([np.corrcoef(r[:-l], r[l:])[0, 1] for r in x_])
        if not np.isfinite(rho) or rho < 0:
            break
        rs += rho
    return rh, m * n / (1 + 2 * rs)


def hier_panel(df, spec):
    y = need(spec, "y", "dependent variable"); x = need(spec, "x", "variable with a country-specific slope"); unit = need(spec, "unit", "unit identifier")
    controls = as_list(spec.get("controls")); distinct([y], [x], controls, [unit])
    check_columns(df, [unit] + ([spec["time"]] if spec.get("time") else []))
    d = numeric(df, [y, x] + controls, keep=[unit]); d[unit] = d[unit].astype(str)
    cs = list(dict.fromkeys(d[unit])); N = len(cs)
    if N < 4:
        raise OESError("A hierarchical model needs at least four units.")
    g = d[unit].map({c: i for i, c in enumerate(cs)}).to_numpy(); yv = d[y].to_numpy(float); xv = d[x].to_numpy(float); zv = d[controls].to_numpy(float) if controls else np.zeros((len(d), 0))
    iters = int_option(spec, "iters", 3000, 600, 20000, "Iterations per chain"); burn = int_option(spec, "burn", 1000, 100, 10000, "Burn-in"); nch = int_option(spec, "chains", 2, 2, 6, "Chains")
    if burn >= iters:
        raise OESError("The burn-in must be shorter than the number of iterations.")
    A_b = float_option(spec, "scale_slope", 1.0, "Prior scale of the slope spread"); A_a = float_option(spec, "scale_intercept", 5.0, "Prior scale of the intercept spread")
    ch_ = [_gibbs_h(yv, xv, zv, g, N, iters, s, A_b, A_a, 1.5) for s in range(nch)]
    post = {k: np.concatenate([c[k][burn:] for c in ch_]) for k in ch_[0]}
    pr = lambda v: np.percentile(v, [5, 95])
    rows, names = [], []
    items = [("Average slope on " + x, lambda c: c["mu"][burn:, 1], post["mu"][:, 1]), ("Spread of slopes (sd)", lambda c: c["s"][burn:, 1], post["s"][:, 1]),
             ("Average intercept", lambda c: c["mu"][burn:, 0], post["mu"][:, 0]), ("Spread of intercepts (sd)", lambda c: c["s"][burn:, 0], post["s"][:, 0]),
             ("Error sd", lambda c: c["sig"][burn:], post["sig"])]
    for j, cname in enumerate(controls):
        items.append((f"Common slope on {cname}", (lambda jj: lambda c: c["gam"][burn:, jj])(j), post["gam"][:, j]))
    for lab, fn, v in items:
        rh, ess = _rhat_ess([fn(c) for c in ch_]); lo, hi = pr(v)
        rows.append([lab, f(v.mean()), f(np.median(v)), f(v.std()), f(lo), f(hi), f(rh, 3), f(ess, 0)])
    sep = []
    for i in range(N):
        m_ = g == i
        if m_.sum() > 3 + zv.shape[1]:
            r = np.linalg.lstsq(np.column_stack([np.ones(m_.sum()), xv[m_], zv[m_]]), yv[m_], rcond=None)[0]; sep.append(r[1])
        else:
            sep.append(np.nan)
    thm = post["th"][:, :, 1]; rows2 = [[cs[i], f(thm[:, i].mean(), 4), f(thm[:, i].std(), 4), f(np.percentile(thm[:, i], 5), 4), f(np.percentile(thm[:, i], 95), 4), f(sep[i], 4)] for i in range(N)]
    tabs = [table("Hyper-parameters and common coefficients", ["Parameter", "Mean", "Median", "Sd", "5%", "95%", "Split R-hat", "Effective sample size"], rows,
                  f"{nch} chains of {iters} iterations, the first {burn} discarded. Weak normal priors (sd 10); half-Cauchy priors for the spreads."),
            table(f"Unit-specific slopes on {x}", ["Unit", "Posterior mean", "Sd", "5%", "95%", "Separate OLS slope"], rows2,
                  "Posterior means are shrunk towards the average slope by an amount the data choose; compare with the unit-by-unit OLS slopes.")]
    o = np.argsort(thm.mean(0)); ch = [chart(f"Unit slopes on {x}: posterior means", [cs[i] for i in o], [("Hierarchical (posterior mean)", thm.mean(0)[o]), ("Separate OLS", np.array(sep)[o])], kind="bar")]
    summary = [stat("Observations", len(d)), stat("Units", N), stat("Average slope", post["mu"][:, 1].mean(), 4), stat("Spread of slopes (median)", np.median(post["s"][:, 1]), 4),
               stat("Sd of separate OLS slopes", float(np.nanstd(sep, ddof=1)), 4)]
    notes = ["Hierarchical panel regression: unit intercepts and slopes drawn from normal distributions with unknown means and spreads (half-Cauchy priors, Gelman 2006), common coefficients on the controls; Gibbs sampler.",
             "Check that R-hat is below 1.05 and the effective sample sizes are adequate; increase the iterations otherwise."]
    return result("hier_panel", f"Hierarchical panel regression: {y} on {x}", len(d), summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Markov switching with time-varying transition probabilities
# ---------------------------------------------------------------------------------------------------------------
def ms_tvtp(df, spec):
    import statsmodels.api as sm
    from .markov import _fit
    y = need(spec, "y", "dependent variable"); drv = as_list(spec.get("driver"))
    if not drv:
        raise OESError("Choose at least one variable that drives the transition probabilities.")
    distinct([y], drv)
    ref = spec.get("reference") if spec.get("reference") not in (None, "") else None
    d, labels = _series_frame(df, spec, [y] + drv + ([ref] if ref else []))
    lag = str(spec.get("lag_driver", "yes")).lower() in ("yes", "true", "1"); sv = str(spec.get("switching_variance", "yes")).lower() in ("yes", "true", "1")
    Z = d[drv].copy()
    if lag:
        Z = Z.shift(1)
    keep = Z.notna().all(axis=1).to_numpy(); yy = d[y][keep].astype(float).reset_index(drop=True); Zt = sm.add_constant(Z[keep].astype(float).reset_index(drop=True), has_constant="add")
    lab = [l for l, k in zip(labels, keep) if k]
    if len(yy) < 50:
        raise OESError("Markov switching with time-varying probabilities needs at least 50 observations.")
    r0 = _fit(sm.tsa.MarkovRegression(yy, k_regimes=2, trend="c", switching_variance=sv))
    r1 = _fit(sm.tsa.MarkovRegression(yy, k_regimes=2, trend="c", switching_variance=sv, exog_tvtp=Zt))
    if r0 is None or r1 is None:
        raise OESError("The model could not be estimated. Try a different driver or switch off the regime-specific variance.")
    def hi_of(r):
        return 0 if r.params["const[0]"] > r.params["const[1]"] else 1
    h0, h1 = hi_of(r0), hi_of(r1); l0, l1 = 1 - h0, 1 - h1
    LR = 2 * (r1.llf - r0.llf); df_ = 2 * len(drv); p_lr = stats.chi2.sf(LR, df_)
    tab_par = table("Regime parameters (time-varying probabilities)", ["Parameter", "Estimate", "Std. err.", "z", "p-value"],
                    [[nm, f(b), f(s), f(b / s, 3), f(2 * stats.norm.sf(abs(b / s)), 4)] for nm, b, s in zip(r1.params.index, r1.params.values, r1.bse.values)],
                    "Logit coefficients are for the probability of moving to regime 0 from regime 0 or 1 (statsmodels' convention); regime 1 is the high-growth state if its mean is higher.")
    tab_cmp = table("Comparison with constant transition probabilities", ["Model", "Log-likelihood", "AIC", "BIC"],
                    [["Constant probabilities", f(r0.llf, 3), f(r0.aic, 2), f(r0.bic, 2)], ["Time-varying probabilities", f(r1.llf, 3), f(r1.aic, 2), f(r1.bic, 2)]],
                    f"Likelihood-ratio test ({df_} restrictions): LR = {LR:.2f}, p = {p_lr:.4f}. The chi-square reference is only approximate in regime-switching models; bootstrap or permute the driver for a stricter test.")
    def stay(r, k, vec):
        lg = lambda v: 1 / (1 + np.exp(-v)); nm = [c for c in r.params.index if c.startswith("p[0->0].tvtp")]; nm1 = [c for c in r.params.index if c.startswith("p[1->0].tvtp")]
        a = np.array([r.params[c] for c in (nm if k == 0 else nm1)])
        return lg(vec @ a) if k == 0 else 1 - lg(vec @ a)
    rows = []
    for j, c in enumerate(drv):
        sd = float(Z[keep][c].std())
        for mult in (-1, 0, 1, 2):
            v = np.zeros(1 + len(drv)); v[0] = 1; v[1 + j] = mult * sd
            rows.append([c, f"{mult:+d} sd ({mult * sd:+.3g})", f(stay(r1, h1, v), 3), f(stay(r1, l1, v), 3)])
    tab_eff = table("Probability of staying in each regime at stated values of the driver (others at zero)", ["Driver", "Value", "P(stay in high-growth)", "P(stay in low-growth)"], rows)
    S0 = np.asarray(r0.smoothed_marginal_probabilities[l0]); S1 = np.asarray(r1.smoothed_marginal_probabilities[l1])
    SP = stay(r1, h1, np.asarray(Zt)); SL = stay(r1, l1, np.asarray(Zt))
    tabs = [tab_par, tab_cmp, tab_eff]
    if ref and ref in d.columns:
        rr = pd.to_numeric(d[ref][keep], errors="coerce").reset_index(drop=True).to_numpy()
        ok = np.isfinite(rr)
        def sc(S):
            c = (S > .5).astype(int); return [f((c[ok] == rr[ok]).mean(), 3), f(c[ok & (rr == 1)].mean(), 3), f(c[ok & (rr == 0)].mean(), 3), int(c.sum())]
        tabs.append(table(f"Regime dates against the reference series {ref}", ["Model", "Agreement", "Reference-1 quarters caught", "False alarms", "Low-growth periods"],
                          [["Constant probabilities"] + sc(S0), ["Time-varying probabilities"] + sc(S1)]))
    x = lab
    ch = [chart("Smoothed probability of the low-growth regime", x, [("Constant probabilities", S0), ("Time-varying probabilities", S1)], "", "Probability"),
          chart("Probability of staying in each regime", x, [("Stay in high-growth", SP), ("Stay in low-growth", SL)], "", "Probability")]
    summary = [stat("Observations", len(yy)), stat("Log-likelihood", r1.llf, 3), stat("LR vs constant", LR, 3), stat("LR p-value", p_lr, 4), stat("Mean, high-growth", float(r1.params[f"const[{h1}]"]), 3), stat("Mean, low-growth", float(r1.params[f"const[{l1}]"]), 3)]
    notes = ["Markov-switching regression with logit time-varying transition probabilities (Filardo 1994; Diebold, Lee and Weinbach 1994). The driver is lagged one period by default so that it is known at the start of the period.",
             "Maximum likelihood from several random starting values; check that the regimes are labelled consistently."]
    return result("ms_tvtp", f"Markov switching with time-varying probabilities: {y}", len(yy), summary, tabs, ch, notes)
