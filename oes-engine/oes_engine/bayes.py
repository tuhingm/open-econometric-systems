"""Bayesian models: linear regression, Bayesian VARs (Minnesota, Normal-Wishart, Independent Normal-Wishart,
Normal-diffuse; block exogeneity), FAVAR, mean-adjusted (steady-state) BVAR and threshold BVAR.

The priors and posteriors follow the ECB's BEAR toolbox technical guide (Dieppe, Legrand and van Roye 2016):
Minnesota with a fixed diagonal residual covariance (Litterman 1986); the natural-conjugate Normal-Wishart prior
(Kadiyala and Karlsson 1997) with analytic posterior and marginal likelihood; the Independent Normal-Wishart and
Normal-diffuse priors by Gibbs sampling; Villani's (2009) steady-state prior; and a two-regime threshold VAR whose
threshold and delay have an exact posterior on a grid (Normal-Wishart marginal likelihood of each split).

Coefficient matrices are k x n (k = 1 + n p): row 0 is the constant, then lag 1 of every variable, lag 2, ...
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import multigammaln

from .core import OESError, as_list, chart, distinct, f, float_option, int_option, need, result, stat, table
from .timeseries import _series_frame


# ================================================================================================ shared VAR tools
def lagmat(Y, p, const=True):
    T, n = Y.shape
    if T - p < 2:
        raise OESError("Too few observations for this lag order.")
    X = ([np.ones((T - p, 1))] if const else []) + [Y[p - l:T - l] for l in range(1, p + 1)]
    return Y[p:], np.hstack(X)


def reg_names(names, p, const=True):
    return (["const"] if const else []) + [f"L{l}.{v}" for l in range(1, p + 1) for v in names]


def ar_sigma(Y, p):
    """Residual standard deviation of an AR(p) with constant for each variable (scales the Minnesota prior)."""
    out = []
    for i in range(Y.shape[1]):
        y, X = lagmat(Y[:, [i]], p)
        e = y[:, 0] - X @ np.linalg.lstsq(X, y[:, 0], rcond=None)[0]
        out.append(np.sqrt(e @ e / max(len(e) - X.shape[1], 1)))
    return np.maximum(np.array(out), 1e-8)


def minnesota(n, p, sig, lam1=0.1, lam2=0.5, lam3=1.0, lam4=100.0, delta=1.0, block=None, lam5=0.001, const=True):
    """Prior means B0 (k x n) and variances V0 (k x n) of the Minnesota prior (BEAR guide, section 1.3)."""
    k = (1 if const else 0) + n * p
    delta = np.broadcast_to(np.asarray(delta, float), (n,))
    B0, V0 = np.zeros((k, n)), np.zeros((k, n))
    o = 1 if const else 0
    for i in range(n):
        if const:
            V0[0, i] = (sig[i] * lam4) ** 2
        B0[o + i, i] = delta[i]
        for l in range(1, p + 1):
            for j in range(n):
                r = o + (l - 1) * n + j
                V0[r, i] = (lam1 / l ** lam3) ** 2 if i == j else (sig[i] * lam1 * lam2 / (sig[j] * l ** lam3)) ** 2
                if block is not None and block[i] and not block[j]:
                    V0[r, i] *= lam5 ** 2          # block exogeneity: the block is not affected by the other variables
    return B0, V0


def nw_prior(n, p, sig, lam1=0.1, lam3=1.0, lam4=100.0, delta=1.0):
    """Natural-conjugate Normal-Wishart prior: B | S ~ MN(B0, S, Phi0), S ~ IW(S0, a0) (BEAR guide, 1.4)."""
    B0, _ = minnesota(n, p, sig, lam1, 1.0, lam3, lam4, delta)
    phi = [lam4 ** 2] + [(lam1 / (l ** lam3 * sig[j])) ** 2 for l in range(1, p + 1) for j in range(n)]
    a0 = n + 2
    return B0, np.diag(phi), (a0 - n - 1) * np.diag(sig ** 2), a0


def nw_posterior(Y, X, B0, Phi0, S0, a0):
    Pi = np.linalg.inv(Phi0)
    Phib = np.linalg.inv(Pi + X.T @ X)
    Bb = Phib @ (Pi @ B0 + X.T @ Y)
    Sb = Y.T @ Y + S0 + B0.T @ Pi @ B0 - Bb.T @ np.linalg.inv(Phib) @ Bb
    Sb = (Sb + Sb.T) / 2
    return Bb, Phib, Sb, a0 + len(Y)


def nw_logml(Y, X, B0, Phi0, S0, a0):
    """Log marginal likelihood of the Normal-Wishart model (Karlsson 2013)."""
    T, n = Y.shape
    Bb, Phib, Sb, ab = nw_posterior(Y, X, B0, Phi0, S0, a0)
    ld = lambda M: np.linalg.slogdet(M)[1]
    return (-n * T / 2 * np.log(np.pi) + multigammaln(ab / 2, n) - multigammaln(a0 / 2, n)
            + n / 2 * (ld(Phib) - ld(Phi0)) + a0 / 2 * ld(S0) - ab / 2 * ld(Sb))


def iw(S, df, rng):
    return stats.invwishart.rvs(df=df, scale=S, random_state=rng)


def nw_draws(Bb, Phib, Sb, ab, ndraw, rng):
    k, n = Bb.shape
    Lp = np.linalg.cholesky(Phib + 1e-12 * np.eye(k))
    Bs, Ss = np.empty((ndraw, k, n)), np.empty((ndraw, n, n))
    for d in range(ndraw):
        S = np.atleast_2d(iw(Sb, ab, rng))
        Bs[d] = Bb + Lp @ rng.standard_normal((k, n)) @ np.linalg.cholesky(S).T
        Ss[d] = S
    return Bs, Ss


def gibbs_inw(Y, X, B0, V0, S0, a0, ndraw, burn, rng, thin=1):
    """Independent Normal-Wishart (or Normal-diffuse with S0 = 0, a0 = 0) Gibbs sampler (BEAR guide, 1.5-1.6)."""
    T, n = Y.shape
    k = X.shape[1]
    b0, Oi = B0.flatten("F"), np.diag(1 / V0.flatten("F"))
    XX, XY = X.T @ X, X.T @ Y
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    Bs, Ss = np.empty((ndraw, k, n)), np.empty((ndraw, n, n))
    for it in range(burn + ndraw * thin):
        E = Y - X @ B
        S = np.atleast_2d(iw(S0 + E.T @ E, a0 + T, rng))
        Si = np.linalg.inv(S)
        P = Oi + np.kron(Si, XX)
        L = np.linalg.cholesky((P + P.T) / 2)
        m = np.linalg.solve(P, Oi @ b0 + (XY @ Si).flatten("F"))
        b = m + np.linalg.solve(L.T, rng.standard_normal(k * n))
        B = b.reshape((k, n), order="F")
        j = it - burn
        if j >= 0 and j % thin == 0:
            Bs[j // thin], Ss[j // thin] = B, S
    return Bs, Ss


def minnesota_posterior(Y, X, B0, V0, sig):
    """Litterman's Minnesota posterior with the residual covariance fixed at diag(sigma^2): equation by equation."""
    k, n = B0.shape
    M, Vs = np.empty((k, n)), []
    for i in range(n):
        Vi = np.linalg.inv(np.diag(1 / V0[:, i]) + X.T @ X / sig[i] ** 2)
        M[:, i] = Vi @ (B0[:, i] / V0[:, i] + X.T @ Y[:, i] / sig[i] ** 2)
        Vs.append(Vi)
    return M, Vs


def minnesota_draws(M, Vs, sig, ndraw, rng):
    k, n = M.shape
    Bs = np.empty((ndraw, k, n))
    for i in range(n):
        L = np.linalg.cholesky(Vs[i])
        Bs[:, :, i] = M[:, i] + rng.standard_normal((ndraw, k)) @ L.T
    return Bs, np.broadcast_to(np.diag(sig ** 2), (ndraw, n, n)).copy()


def companion(B, n, p, const=True):
    A = B[1:] if const else B
    F = np.zeros((n * p, n * p))
    F[:n] = A.T
    if p > 1:
        F[n:, :-n] = np.eye(n * (p - 1))
    return F


def irf(B, S, n, p, h, const=True):
    """Cholesky impulse responses: out[t, response, shock]."""
    F = companion(B, n, p, const)
    P = np.linalg.cholesky(S + 1e-12 * np.eye(n))
    out = np.empty((h + 1, n, n))
    Fk = np.eye(n * p)
    for t in range(h + 1):
        out[t] = Fk[:n, :n] @ P
        Fk = Fk @ F
    return out


def simulate_forecast(B, S, Yhist, p, h, rng, const=True):
    n = Yhist.shape[1]
    hist = list(Yhist[-p:])
    L = np.linalg.cholesky(S + 1e-12 * np.eye(n))
    out = np.empty((h, n))
    for t in range(h):
        x = ([1.0] if const else []) + [v for l in range(1, p + 1) for v in hist[-l]]
        y = np.asarray(x) @ B + L @ rng.standard_normal(n)
        hist.append(y)
        out[t] = y
    return out


def post_table(title, names, draws, note=None):
    """Posterior summary of draws (ndraw x m)."""
    D = np.asarray(draws, float)
    q = np.percentile(D, [2.5, 50, 97.5], axis=0)
    rows = [[n_, f(D[:, j].mean()), f(D[:, j].std(ddof=1)), f(q[0, j]), f(q[1, j]), f(q[2, j]), f((D[:, j] > 0).mean(), 3)]
            for j, n_ in enumerate(names)]
    return table(title, ["Parameter", "Mean", "Std. dev.", "2.5%", "Median", "97.5%", "P(> 0)"], rows, note)


def irf_charts(I, names, h, shocks=None, max_charts=9, title_suffix=""):
    """I: draws x (h+1) x n x n. Median with 16-84% band."""
    med, lo, hi = (np.percentile(I, q, axis=0) for q in (50, 16, 84))
    out = []
    shocks = range(len(names)) if shocks is None else shocks
    for j in shocks:
        for i in range(len(names)):
            if len(out) >= max_charts:
                return out
            out.append(chart(f"Response of {names[i]} to a shock in {names[j]}{title_suffix}", list(range(h + 1)),
                             [("median", med[:, i, j])], "horizon", names[i], band=(lo[:, i, j], hi[:, i, j])))
    return out


def forecast_charts(F, Yhist, names, labels, h, back=24):
    med, lo, hi = (np.percentile(F, q, axis=0) for q in (50, 16, 84))
    back = min(back, len(Yhist))
    x = list(labels[-back:]) + [f"h{t + 1}" for t in range(h)]
    out = []
    for i, v in enumerate(names):
        hist = list(Yhist[-back:, i]) + [None] * h
        fc = [None] * (back - 1) + [Yhist[-1, i]] + list(med[:, i])
        blo = [None] * (back - 1) + [Yhist[-1, i]] + list(lo[:, i])
        bhi = [None] * (back - 1) + [Yhist[-1, i]] + list(hi[:, i])
        out.append(chart(f"Forecast of {v} (median, 68% band)", x, [("data", hist), ("forecast", fc)], "", v, band=(blo, bhi)))
    return out


def _floats(spec, key, n, label):
    v = spec.get(key)
    if v in (None, ""):
        return None
    vals = [float(s) for s in str(v).replace(";", ",").split(",") if s.strip() != ""] if not isinstance(v, (list, tuple)) else [float(s) for s in v]
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        raise OESError(f"{label}: give one value, or one per variable ({n}).")
    return np.array(vals)


def _hyper(spec):
    return dict(lam1=float_option(spec, "lambda1", 0.1, "Overall tightness (lambda1)"),
                lam2=float_option(spec, "lambda2", 0.5, "Cross-variable tightness (lambda2)"),
                lam3=float_option(spec, "lambda3", 1.0, "Lag decay (lambda3)"),
                lam4=float_option(spec, "lambda4", 100.0, "Constant tightness (lambda4)"))


def _draw_settings(spec, draws=2000, burn=1000):
    return (int_option(spec, "draws", draws, 100, 20000, "Posterior draws"), int_option(spec, "burn", burn, 0, 20000, "Burn-in"),
            np.random.default_rng(int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")))


# ================================================================================================ Bayesian regression
def bayes_reg(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "explanatory variables"))
    distinct([y], x)
    d, labels = _series_frame(df, spec, [y] + x)
    Y = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d[x].to_numpy(float)])
    n, k = X.shape
    names = ["const"] + x
    prior = spec.get("prior", "gprior")
    method = spec.get("method", "conjugate")
    b0 = np.zeros(k)
    pm = _floats(spec, "prior_mean", k - 1, "Prior means of the slopes")
    if pm is not None:
        b0[1:] = pm
    ols = sm.OLS(Y, X).fit()
    a0, d0 = 0.01, 0.01
    if prior == "gprior":
        g = float_option(spec, "g", n, "g")
        Xc = X[:, 1:] - X[:, 1:].mean(0)
        V0i = np.zeros((k, k))
        V0i[0, 0] = 1e-10                                   # (almost) flat prior on the constant
        V0i[1:, 1:] = Xc.T @ Xc / g
        prior_txt = f"Zellner g-prior on the slopes, g = {g:g}, flat constant"
    else:
        sd = float_option(spec, "prior_sd", 10.0, "Prior standard deviation")
        V0i = np.diag([1e-10] + [1 / sd ** 2] * (k - 1))
        prior_txt = f"normal prior on the slopes, N(prior mean, {sd:g}^2), flat constant"
    V0 = np.linalg.inv(V0i)
    ndraw, burn, rng = _draw_settings(spec, 5000, 1000)
    tabs, charts, summary, notes = [], [], [], []
    if method == "conjugate":
        Vn = np.linalg.inv(V0i + X.T @ X)
        bn = Vn @ (V0i @ b0 + X.T @ Y)
        an = a0 + n / 2
        dn = d0 + 0.5 * (Y @ Y + b0 @ V0i @ b0 - bn @ np.linalg.inv(Vn) @ bn)
        scale = np.sqrt(dn / an * np.diag(Vn))
        nu = 2 * an
        sd = scale * np.sqrt(nu / (nu - 2))
        q = stats.t.ppf([0.025, 0.5, 0.975], nu)
        rows = [[v, f(bn[j]), f(sd[j]), f(bn[j] + q[0] * scale[j]), f(bn[j]), f(bn[j] + q[2] * scale[j]),
                 f(stats.t.sf(-bn[j] / scale[j], nu), 3)] for j, v in enumerate(names)]
        tabs.append(table(f"Posterior of the coefficients ({prior_txt})", ["Parameter", "Mean", "Std. dev.", "2.5%", "Median", "97.5%", "P(> 0)"], rows,
                          f"Exact Student-t marginal posteriors with {nu:g} degrees of freedom (Normal-inverse-gamma conjugate prior)."))
        logml = (-n / 2 * np.log(2 * np.pi) + 0.5 * (np.linalg.slogdet(Vn)[1] + np.linalg.slogdet(V0i)[1])
                 + a0 * np.log(d0) - an * np.log(dn) + _lgamma(an) - _lgamma(a0))
        summary += [stat("Posterior mean of sigma^2", dn / (an - 1)), stat("Log marginal likelihood", logml, 3)]
    else:
        B, s2 = ols.params.copy(), ols.mse_resid
        draws, s2d = np.empty((ndraw, k)), np.empty(ndraw)
        for it in range(burn + ndraw):
            P = V0i + X.T @ X / s2
            L = np.linalg.cholesky(P)
            mean = np.linalg.solve(P, V0i @ b0 + X.T @ Y / s2)
            B = mean + np.linalg.solve(L.T, rng.standard_normal(k))
            e = Y - X @ B
            s2 = 1 / rng.gamma(a0 + n / 2, 1 / (d0 + e @ e / 2))
            if it >= burn:
                draws[it - burn], s2d[it - burn] = B, s2
        tabs.append(post_table(f"Posterior of the coefficients ({prior_txt}, independent prior; Gibbs sampler)", names, draws,
                               f"{ndraw} draws after a burn-in of {burn}."))
        rows = []
        for j, v in enumerate(names):
            ess = _ess(draws[:, j])
            z = _geweke(draws[:, j])
            rows.append([v, f(ess, 0), f(z, 3), f(2 * stats.norm.sf(abs(z)), 4)])
        tabs.append(table("MCMC convergence diagnostics", ["Parameter", "Effective sample size", "Geweke z", "p-value"], rows,
                          "Geweke compares the first 10% and last 50% of the chain; a large |z| suggests the chain has not converged."))
        j = 1 if k > 1 else 0
        charts.append(chart(f"Trace of {names[j]}", list(range(1, ndraw + 1, max(1, ndraw // 500))), [(names[j], draws[::max(1, ndraw // 500), j])], "draw", names[j]))
        hist, edges = np.histogram(draws[:, j], bins=30)
        charts.append(chart(f"Posterior density of {names[j]}", [f((a + b) / 2, 4) for a, b in zip(edges[:-1], edges[1:])], [("draws", hist)], names[j], "count", kind="bar"))
        summary.append(stat("Posterior mean of sigma^2", float(s2d.mean())))
    tabs.append(table("OLS for comparison", ["Parameter", "Coef.", "Std. err."], [[v, f(b), f(s)] for v, b, s in zip(names, ols.params, ols.bse)]))
    summary = [stat("Observations", n), stat("Prior", "g-prior" if prior == "gprior" else "normal"),
               stat("Method", "exact (conjugate)" if method == "conjugate" else "Gibbs sampler")] + summary
    notes.append("Inverse-gamma(0.01, 0.01) prior on the error variance.")
    return result("bayes_reg", f"Bayesian linear regression: {y}", n, summary, tabs, charts, notes)


def _lgamma(x):
    from scipy.special import gammaln
    return gammaln(x)


def _ess(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = x @ x / n
    if v == 0:
        return float(n)
    s = 0.0
    for lag in range(1, min(n - 1, 1000)):
        r = (x[:-lag] @ x[lag:]) / n / v
        if r < 0.05:
            break
        s += r
    return n / (1 + 2 * s)


def _geweke(x):
    x = np.asarray(x, float)
    a, b = x[: len(x) // 10], x[len(x) // 2:]
    va = np.var(a, ddof=1) * len(a) / _ess(a) / len(a)
    vb = np.var(b, ddof=1) * len(b) / _ess(b) / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb) if va + vb > 0 else 0.0


# ================================================================================================ BVAR
PRIORS = {"minnesota": "Minnesota (Litterman)", "nw": "Normal-Wishart", "inw": "Independent Normal-Wishart", "nd": "Normal-diffuse"}


def estimate_bvar(Y, p, prior, hyper, delta, ndraw, burn, rng, block=None):
    """Returns (Bs, Ss, extra) posterior draws for a VAR(p) with constant."""
    n = Y.shape[1]
    sig = ar_sigma(Y, p)
    Yt, X = lagmat(Y, p)
    extra = {}
    if prior == "minnesota":
        B0, V0 = minnesota(n, p, sig, delta=delta, block=block, **hyper)
        M, Vs = minnesota_posterior(Yt, X, B0, V0, sig)
        Bs, Ss = minnesota_draws(M, Vs, sig, ndraw, rng)
    elif prior == "nw":
        if block is not None and any(block):
            raise OESError("Block exogeneity is not available with the Normal-Wishart prior (its Kronecker structure ties all equations); use Minnesota, Independent Normal-Wishart or Normal-diffuse.")
        B0, Phi0, S0, a0 = nw_prior(n, p, sig, hyper["lam1"], hyper["lam3"], hyper["lam4"], delta)
        post = nw_posterior(Yt, X, B0, Phi0, S0, a0)
        Bs, Ss = nw_draws(*post, ndraw, rng)
        extra["logml"] = nw_logml(Yt, X, B0, Phi0, S0, a0)
    elif prior in ("inw", "nd"):
        B0, V0 = minnesota(n, p, sig, delta=delta, block=block, **hyper)
        if prior == "inw":
            S0, a0 = (n + 1) * np.diag(sig ** 2), n + 2
        else:
            S0, a0 = np.zeros((n, n)), 0
        Bs, Ss = gibbs_inw(Yt, X, B0, V0, S0, a0, ndraw, burn, rng)
    else:
        raise OESError("Unknown prior.")
    return Bs, Ss, Yt, X, extra


def _block(spec, names):
    blk = as_list(spec.get("block"))
    if not blk:
        return None
    bad = [b for b in blk if b not in names]
    if bad:
        raise OESError("Block-exogenous variables must be among the VAR variables: " + ", ".join(bad))
    if len(blk) == len(names):
        raise OESError("Leave at least one variable outside the exogenous block.")
    return np.array([v in blk for v in names])


def _var_common_out(names, p, Bs, Ss, Yt, X, Y, labels, spec, h, title_extra=""):
    tabs = []
    rn = reg_names(names, p)
    for i, v in enumerate(names):
        tabs.append(post_table(f"Equation: {v}", rn, Bs[:, :, i]))
    I = np.array([irf(Bs[d], Ss[d], len(names), p, h) for d in range(len(Bs))])
    hf = int_option(spec, "fhorizon", 8, 1, 60, "Forecast horizon")
    rng = np.random.default_rng(1)
    step = max(1, len(Bs) // 500)
    F = np.array([simulate_forecast(Bs[d], Ss[d], Y, p, hf, rng) for d in range(0, len(Bs), step)])
    charts = irf_charts(I, names, h) + forecast_charts(F, Y, names, labels, hf)
    eig = [np.abs(np.linalg.eigvals(companion(Bs[d], len(names), p))).max() for d in range(0, len(Bs), step)]
    return tabs, charts, float(np.mean(np.array(eig) < 1))


def bvar(df, spec):
    names = as_list(need(spec, "vars", "variables (at least two)"))
    if len(names) < 2:
        raise OESError("A VAR needs at least two variables.")
    distinct(names)
    d, labels = _series_frame(df, spec, names)
    Y = d[names].to_numpy(float)
    p = int_option(spec, "lags", 4, 1, 12, "Lags")
    prior = spec.get("prior", "minnesota")
    hyper = _hyper(spec)
    delta = _floats(spec, "delta", len(names), "Prior mean of the own first lag")
    delta = 1.0 if delta is None else delta
    h = int_option(spec, "horizon", 20, 1, 60, "IRF horizon")
    ndraw, burn, rng = _draw_settings(spec, 1000, 500)
    block = _block(spec, names)
    Bs, Ss, Yt, X, extra = estimate_bvar(Y, p, prior, hyper, delta, ndraw, burn, rng, block)
    tabs, charts, stable = _var_common_out(names, p, Bs, Ss, Yt, X, Y, labels, spec, h)
    ols = np.linalg.lstsq(X, Yt, rcond=None)[0]
    shrink = float(np.linalg.norm(Bs.mean(0)[1:] - ols[1:]) / max(np.linalg.norm(ols[1:]), 1e-12))
    summary = [stat("Observations", len(Yt)), stat("Prior", PRIORS[prior]), stat("Lags", p), stat("lambda1", hyper["lam1"]),
               stat("Share of stable draws", stable, 3), stat("Distance from OLS (relative)", shrink, 3)]
    if "logml" in extra:
        summary.append(stat("Log marginal likelihood", extra["logml"], 3))
    notes = [f"Prior: {PRIORS[prior]}; own first lag prior mean {', '.join(f'{v:g}' for v in np.broadcast_to(delta, (len(names),)))}; "
             f"lambda1 = {hyper['lam1']:g}, lambda2 = {hyper['lam2']:g}, lambda3 = {hyper['lam3']:g}, lambda4 = {hyper['lam4']:g}.",
             "Impulse responses: Cholesky identification in the order of the variables; median and 68% credible band."]
    if block is not None:
        notes.append("Block exogeneity: " + ", ".join(v for v, b in zip(names, block) if b) + " do not respond to the lags of the other variables.")
    if prior in ("inw", "nd"):
        notes.append(f"Gibbs sampler: {ndraw} draws after {burn} burn-in.")
    return result("bvar", "Bayesian VAR: " + PRIORS[prior], len(Yt), summary, tabs, charts, notes)


# ================================================================================================ FAVAR
def favar(df, spec):
    yv = as_list(need(spec, "y", "observed variables (policy variable last)"))
    xv = as_list(need(spec, "x", "informational series"))
    distinct(yv, xv)
    K = int_option(spec, "factors", 3, 1, 10, "Number of factors")
    if len(xv) < K + 1:
        raise OESError("The informational panel needs more series than factors.")
    d, labels = _series_frame(df, spec, yv + xv)
    Xr = d[xv].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0, ddof=1)
    if (sd == 0).any():
        raise OESError("Some informational series are constant.")
    Xs = (Xr - mu) / sd
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    Fct = U[:, :K] * s[:K]
    share = s ** 2 / (s ** 2).sum()
    names = [f"F{j + 1}" for j in range(K)] + yv
    Y = np.column_stack([Fct, d[yv].to_numpy(float)])
    p = int_option(spec, "lags", 2, 1, 12, "Lags")
    prior = spec.get("prior", "nw")
    hyper = _hyper(spec)
    h = int_option(spec, "horizon", 20, 1, 60, "IRF horizon")
    ndraw, burn, rng = _draw_settings(spec, 1000, 500)
    Bs, Ss, Yt, X, extra = estimate_bvar(Y, p, prior, hyper, 0.0, ndraw, burn, rng)
    # loadings of every informational series on (factors, observed variables)
    Lam = np.linalg.lstsq(np.column_stack([np.ones(len(Y)), Y]), Xs, rcond=None)[0][1:]          # (K + m) x N
    I = np.array([irf(Bs[dd], Ss[dd], len(names), p, h) for dd in range(len(Bs))])          # draws x h x (K+m) x (K+m)
    pol = len(names) - 1
    resp_x = np.einsum("dhks,kn->dhns", I, Lam)[..., pol] * sd                                  # draws x h x N, original units
    med, lo, hi = (np.percentile(resp_x, q, axis=0) for q in (50, 16, 84))
    charts = []
    for i in range(len(yv)):
        Ii = I[:, :, K + i, pol]
        charts.append(chart(f"Response of {yv[i]} to a shock in {yv[-1]}", list(range(h + 1)), [("median", np.median(Ii, 0))], "horizon", yv[i],
                            band=(np.percentile(Ii, 16, 0), np.percentile(Ii, 84, 0))))
    for j in range(min(len(xv), 9 - len(charts))):
        charts.append(chart(f"Response of {xv[j]} to a shock in {yv[-1]}", list(range(h + 1)), [("median", med[:, j])], "horizon", xv[j], band=(lo[:, j], hi[:, j])))
    r2 = 1 - ((Xs - np.column_stack([np.ones(len(Y)), Y]) @ np.vstack([np.zeros((1, len(xv))), Lam])) ** 2).sum(0) / (Xs ** 2).sum(0)
    tabs = [table("Principal-component factors", ["Factor", "Share of variance", "Cumulative"],
                  [[f"F{j + 1}", f(share[j], 4), f(share[:j + 1].sum(), 4)] for j in range(K)]),
            table(f"Responses of the informational series to a shock in {yv[-1]} (median)", ["Series", "Impact", f"h = {min(4, h)}", f"h = {min(12, h)}", "R-squared of loadings"],
                  [[v, f(med[0, j]), f(med[min(4, h), j]), f(med[min(12, h), j]), f(r2[j], 3)] for j, v in enumerate(xv)])]
    for i, v in enumerate(yv):
        tabs.append(post_table(f"Equation: {v}", reg_names(names, p), Bs[:, :, K + i]))
    summary = [stat("Observations", len(Yt)), stat("Factors", K), stat("Informational series", len(xv)), stat("Variance explained", float(share[:K].sum()), 3),
               stat("Prior", PRIORS[prior])]
    notes = ["Two-step FAVAR (Bernanke, Boivin and Eliasz 2005): factors are principal components of the standardised informational "
             "series; a Bayesian VAR is estimated on the factors and the observed variables; responses of every informational series "
             "come from its loadings on the VAR variables.",
             f"The shock is the last observed variable ({yv[-1]}), identified recursively (ordered last)."]
    return result("favar", f"FAVAR: shock to {yv[-1]}", len(Yt), summary, tabs, charts, notes)


# ================================================================================================ mean-adjusted BVAR
def mabvar(df, spec):
    names = as_list(need(spec, "vars", "variables (at least two)"))
    if len(names) < 2:
        raise OESError("A VAR needs at least two variables.")
    distinct(names)
    d, labels = _series_frame(df, spec, names)
    Y = d[names].to_numpy(float)
    n = len(names)
    p = int_option(spec, "lags", 4, 1, 12, "Lags")
    hyper = _hyper(spec)
    m0 = _floats(spec, "ss_mean", n, "Prior means of the steady states")
    s0 = _floats(spec, "ss_sd", n, "Prior standard deviations of the steady states")
    if m0 is None:
        m0 = Y.mean(0)
    if s0 is None:
        s0 = Y.std(0, ddof=1)
    ndraw, burn, rng = _draw_settings(spec, 1000, 500)
    sig = ar_sigma(Y, p)
    B0, V0 = minnesota(n, p, sig, delta=0.0, const=False, **hyper)
    b0, Oi = B0.flatten("F"), np.diag(1 / V0.flatten("F"))
    k = n * p
    T = len(Y) - p
    psi = m0.copy()
    S = np.diag(sig ** 2)
    S0, a0 = (n + 1) * np.diag(sig ** 2), n + 2
    P0i = np.diag(1 / s0 ** 2)
    draws_psi, Bs, Ss = np.empty((ndraw, n)), np.empty((ndraw, k + 1, n)), np.empty((ndraw, n, n))
    for it in range(burn + ndraw):
        Z = Y - psi
        Zt, Xz = lagmat(Z, p, const=False)
        Si = np.linalg.inv(S)
        P = Oi + np.kron(Si, Xz.T @ Xz)
        L = np.linalg.cholesky((P + P.T) / 2)
        m = np.linalg.solve(P, Oi @ b0 + (Xz.T @ Zt @ Si).flatten("F"))
        A = (m + np.linalg.solve(L.T, rng.standard_normal(k * n))).reshape((k, n), order="F")
        E = Zt - Xz @ A
        S = np.atleast_2d(iw(S0 + E.T @ E, a0 + T, rng))
        Si = np.linalg.inv(S)
        # steady state given A and S: w_t = y_t - sum A_l' y_{t-l} = D psi + e_t, D = I - sum A_l'
        Yt, Xy = lagmat(Y, p, const=False)
        W = Yt - Xy @ A
        D = np.eye(n) - sum(A[(l - 1) * n:l * n].T for l in range(1, p + 1))
        Pp = P0i + T * D.T @ Si @ D
        mp = np.linalg.solve(Pp, P0i @ m0 + D.T @ Si @ W.sum(0))
        psi = mp + np.linalg.solve(np.linalg.cholesky(Pp).T, rng.standard_normal(n))
        if it >= burn:
            j = it - burn
            draws_psi[j] = psi
            Bs[j] = np.vstack([(D @ psi)[None, :], A])        # implied constant c = (I - sum A) psi
            Ss[j] = S
    h = int_option(spec, "horizon", 20, 1, 60, "IRF horizon")
    tabs = [post_table("Steady states (posterior)", names, draws_psi,
                       "Prior: " + "; ".join(f"{v} ~ N({a:.3g}, {b:.3g}^2)" for v, a, b in zip(names, m0, s0)))]
    hf = int_option(spec, "fhorizon", 20, 1, 80, "Forecast horizon")
    t2, charts, stable = _var_common_out(names, p, Bs, Ss, None, None, Y, labels, dict(spec, fhorizon=hf), h)
    tabs += t2
    summary = [stat("Observations", T), stat("Lags", p)] + [stat(f"Steady state of {v}", float(draws_psi[:, i].mean())) for i, v in enumerate(names)]
    notes = ["Mean-adjusted BVAR (Villani 2009): y_t - psi = sum A_l (y_{t-l} - psi) + e_t, with an informative normal prior on the "
             "steady state psi and an independent Minnesota prior on the lags (prior mean 0); Gibbs sampler with " f"{ndraw} draws after {burn}.",
             "Long-horizon forecasts converge to the steady state, so the prior on psi disciplines them."]
    return result("mabvar", "Mean-adjusted (steady-state) BVAR", T, summary, tabs, charts, notes)


# ================================================================================================ threshold BVAR
def tvar(df, spec):
    names = as_list(need(spec, "vars", "variables (at least two)"))
    if len(names) < 2:
        raise OESError("A VAR needs at least two variables.")
    zname = need(spec, "threshold_var", "threshold variable")
    distinct(names)
    cols = names + ([zname] if zname not in names else [])
    d, labels = _series_frame(df, spec, cols)
    Y = d[names].to_numpy(float)
    z = d[zname].to_numpy(float)
    n = len(names)
    p = int_option(spec, "lags", 2, 1, 8, "Lags")
    dmax = int_option(spec, "max_delay", 2, 1, 8, "Maximum delay")
    trim = float_option(spec, "trim", 0.15, "Minimum share of observations in each regime")
    lam1 = float_option(spec, "lambda1", 0.2, "Overall tightness (lambda1)")
    ngrid = int_option(spec, "grid", 40, 5, 200, "Threshold grid points")
    ndraw, _, rng = _draw_settings(spec, 1000, 0)
    sig = ar_sigma(Y, p)
    Yt, X = lagmat(Y, p)
    s = max(p, dmax)
    Yt, X = Yt[s - p:], X[s - p:]
    T = len(Yt)
    t_idx = np.arange(s, s + T)
    B0, Phi0, S0, a0 = nw_prior(n, p, sig, lam1, 1.0, 100.0, 0.0)
    grid = np.unique(np.quantile(z[s - 1:len(z) - 1], np.linspace(trim, 1 - trim, ngrid)))
    k = X.shape[1]
    post = []
    for dl in range(1, dmax + 1):
        zz = z[t_idx - dl]
        for r in grid:
            hi = zz > r
            if hi.sum() < max(trim * T, k + n + 2) or (~hi).sum() < max(trim * T, k + n + 2):
                continue
            ml = nw_logml(Yt[~hi], X[~hi], B0, Phi0, S0, a0) + nw_logml(Yt[hi], X[hi], B0, Phi0, S0, a0)
            post.append((dl, r, ml))
    if not post:
        raise OESError("No threshold leaves enough observations in both regimes; lower the trimming share or the lag order.")
    P = np.array([q[2] for q in post])
    w = np.exp(P - P.max())
    w /= w.sum()
    best = post[int(np.argmax(w))]
    dl, r = best[0], best[1]
    hi = z[t_idx - dl] > r
    lin = nw_logml(Yt, X, B0, Phi0, S0, a0)
    tabs = [table("Posterior of the delay", ["Delay d", "Posterior probability"],
                  [[dd, f(sum(wi for (q, wi) in zip(post, w) if q[0] == dd), 4)] for dd in range(1, dmax + 1)]),
            table("Threshold and regimes (at the posterior mode)", ["Item", "Value"],
                  [["Threshold r", f(r)], ["Posterior mean of r", f(sum(q[1] * wi for q, wi in zip(post, w)))],
                   ["Delay d", dl], ["Observations: low regime", int((~hi).sum())], ["Observations: high regime", int(hi.sum())],
                   ["Log Bayes factor, threshold vs linear VAR", f(float(np.log(np.exp(P - P.max()).mean()) + P.max() - lin), 3)]],
                  f"Regime is high when {zname}(t - d) > r. Bayes factor averages the marginal likelihood over the grid (uniform prior).")]
    I_reg = {}
    h = int_option(spec, "horizon", 20, 1, 60, "IRF horizon")
    for lab, mask in (("low", ~hi), ("high", hi)):
        pst = nw_posterior(Yt[mask], X[mask], B0, Phi0, S0, a0)
        Bs, Ss = nw_draws(*pst, ndraw, rng)
        for i, v in enumerate(names):
            tabs.append(post_table(f"{lab.capitalize()} regime, equation: {v}", reg_names(names, p), Bs[:, :, i]))
        I_reg[lab] = np.array([irf(Bs[q], Ss[q], n, p, h) for q in range(ndraw)])
    charts = []
    pr = {}
    for (q, wi) in zip(post, w):
        if q[0] == dl:
            pr[q[1]] = pr.get(q[1], 0) + wi
    xs = sorted(pr)
    charts.append(chart(f"Posterior of the threshold (delay {dl})", [f(v, 4) for v in xs], [("probability", [pr[v] for v in xs])], zname, "probability", kind="bar"))
    zz = z[t_idx - dl]
    charts.append(chart(f"{zname}(t-{dl}) and the threshold", labels[s:s + T], [(zname, zz), ("threshold", [r] * T)]))
    for j in range(n):
        for i in range(n):
            if len(charts) >= 11:
                break
            charts.append(chart(f"Response of {names[i]} to a shock in {names[j]}", list(range(h + 1)),
                                [("low regime", np.median(I_reg["low"][:, :, i, j], 0)), ("high regime", np.median(I_reg["high"][:, :, i, j], 0))], "horizon", names[i]))
    summary = [stat("Observations", T), stat("Threshold", float(r)), stat("Delay", dl), stat("High-regime share", float(hi.mean()), 3),
               stat("Lags", p)]
    notes = ["Bayesian threshold VAR: Normal-Wishart prior (Minnesota-type, lambda1 = %g) in each regime; the threshold and delay have "
             "an exact posterior over the grid because each regime's marginal likelihood is analytic. Regime IRFs are Cholesky "
             "responses within each regime (no switching over the horizon)." % lam1]
    return result("tvar", f"Threshold BVAR (threshold variable {zname})", T, summary, tabs, charts, notes)
