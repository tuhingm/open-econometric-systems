"""Panel VAR with the six estimators of the BEAR toolbox (Dieppe, Legrand and van Roye 2016, chapter 4).

  mg            OLS mean group: a VAR for each unit, coefficients averaged (Pesaran and Smith 1995)
  pooled        Bayesian pooled estimator: one homogeneous VAR, Normal-Wishart prior
  zh            Zellner and Hong (1989) random effects: unit coefficients shrunk towards the pooled estimate
  hierarchical  Jarocinski (2010) hierarchical prior: beta_i ~ N(b, lambda1 Omega_b), lambda1 estimated
  static        Canova and Ciccarelli (2009) factor structure with cross-unit interdependencies
  dynamic       the same factor structure with factors that follow random walks (time-varying)

For the factor estimators every equation of every unit contains the lags of all variables of all units; the
coefficients are driven by a few factors: a common one, one per unit (of the equation), one per variable (of the
equation), an own-lag factor, a same-unit (domestic) factor and one per lag. Series are demeaned by unit.
"""
import numpy as np
import pandas as pd

from .bayes import (_draw_settings, ar_sigma, irf, irf_charts, iw, lagmat, minnesota, nw_draws, nw_posterior, nw_prior,
                    post_table, reg_names)
from .core import OESError, as_list, chart, distinct, f, float_option, int_option, need, numeric, result, stat, table, time_order
from .statespace_var import _mvn

EST = {"mg": "OLS mean group", "pooled": "Bayesian pooled", "zh": "Zellner-Hong random effects", "hierarchical": "Hierarchical (Jarocinski)",
       "static": "Static factor (Canova-Ciccarelli)", "dynamic": "Dynamic factor (Canova-Ciccarelli)"}


def _panel_series(df, spec, names):
    unit, time = need(spec, "unit", "unit identifier"), need(spec, "time", "time identifier")
    distinct([unit, time], names)
    d = numeric(df, names, keep=[unit, time])
    d = time_order(d, time)
    times = list(dict.fromkeys(d[time].astype(str)))
    units, ids = [], []
    for u, g in d.groupby(unit, sort=False):
        g = g.drop_duplicates(time)
        g = g.set_index(g[time].astype(str))
        units.append(g.reindex(times)[names].to_numpy(float))
        ids.append(str(u))
    return units, ids, times


def _balanced(units, ids):
    keep = [(Y, i) for Y, i in zip(units, ids) if np.isfinite(Y).all()]
    if len(keep) < 2:
        raise OESError("This estimator needs a balanced panel with at least two complete units.")
    return [k[0] for k in keep], [k[1] for k in keep]


def pvar(df, spec):
    names = as_list(need(spec, "vars", "variables"))
    if len(names) < 2:
        raise OESError("A panel VAR needs at least two variables.")
    est = spec.get("estimator", "mg")
    if est not in EST:
        raise OESError("Unknown panel VAR estimator.")
    units, ids, times = _panel_series(df, spec, names)
    p = int_option(spec, "lags", 1, 1, 4, "Lags")
    h = int_option(spec, "horizon", 12, 1, 40, "IRF horizon")
    ndraw, burn, rng = _draw_settings(spec, 1000, 500)
    n = len(names)
    fn = {"mg": _mg, "pooled": _pooled, "zh": _zh, "hierarchical": _hier, "static": _factor, "dynamic": _factor}[est]
    out = fn(units, ids, times, names, p, h, ndraw, burn, rng, spec, dynamic=(est == "dynamic"))
    tabs, charts, summary, notes, nobs = out
    tabs[:0] = _panel_pretests(units, names, p)
    summary = [stat("Estimator", EST[est]), stat("Lags", p)] + summary
    return result("pvar", f"Panel VAR: {EST[est]}", nobs, summary, tabs, charts, notes)


def _panel_pretests(units, names, p):
    """Panel unit roots (Maddala-Wu Fisher ADF), cross-sectional dependence (Pesaran CD) and lag choice by pooled BIC."""
    from . import _panel_tests as PT
    from .core import test_row, tests_table
    rows = []
    for i, v in enumerate(names):
        W = pd.DataFrame({u: Y[:, i] for u, Y in enumerate(units)})
        fa = PT._fisher_adf(W, 4)
        if fa:
            rows.append(test_row(f"Fisher ADF (Maddala-Wu): {v}", fa["P"], fa["P_p"], f"{fa['n']} units; H0: all units have a unit root"))
        cd, pcd, rr = PT._cd(W.to_numpy(float))
        if cd is not None:
            rows.append(test_row(f"Pesaran CD: {v}", cd, pcd, f"mean |rho| = {f(rr, 3)}; H0: cross-sectional independence"))
    out = [tests_table(rows, "Panel unit-root and cross-sectional dependence tests",
                       "Stationary series are needed for a panel VAR in levels; strong cross-sectional dependence favours the factor estimators, which model spillovers between units.")]
    bic = []
    for q in range(1, 5):
        data = _unit_data(units, q)
        if not data:
            continue
        Yt = np.vstack([d[0][4 - q:] if len(d[0]) > 4 - q else d[0] for d in data]); X = np.vstack([d[1][4 - q:] if len(d[1]) > 4 - q else d[1] for d in data])
        B = np.linalg.lstsq(X, Yt, rcond=None)[0]
        E = Yt - X @ B
        S = E.T @ E / len(E)
        bic.append([q, f(np.linalg.slogdet(S)[1] + np.log(len(E)) * B.size / len(E), 4)])
    if bic:
        best = min(bic, key=lambda r: r[1])[0]
        out.append(table("Lag order: BIC of the pooled VAR (common sample)", ["Lags", "BIC (lower is better)"], bic, f"BIC prefers {best} lag(s); the model uses {p}."))
    return out


def _unit_data(units, p):
    out = []
    for Y in units:
        Y = Y[np.isfinite(Y).all(1)]
        if len(Y) > p + 2:
            out.append(lagmat(Y, p))
    return out


def _mg(units, ids, times, names, p, h, ndraw, burn, rng, spec, dynamic=False):
    n = len(names)
    data = _unit_data(units, p)
    Bs, Ss = [], []
    for Yt, X in data:
        if len(Yt) <= X.shape[1] + 1:
            continue
        B = np.linalg.lstsq(X, Yt, rcond=None)[0]
        E = Yt - X @ B
        Bs.append(B); Ss.append(E.T @ E / (len(E) - X.shape[1]))
    if len(Bs) < 2:
        raise OESError("Too few units have enough periods for unit-by-unit VARs.")
    Bs, Ss = np.array(Bs), np.array(Ss)
    N = len(Bs)
    rn = reg_names(names, p)
    tabs = []
    for i, v in enumerate(names):
        m, s = Bs[:, :, i].mean(0), Bs[:, :, i].std(0, ddof=1) / np.sqrt(N)
        tabs.append(table(f"Mean-group equation: {v}", ["Regressor", "Mean", "Std. err.", "z", "Min over units", "Max over units"],
                          [[r, f(m[j]), f(s[j]), f(m[j] / s[j], 4), f(Bs[:, j, i].min()), f(Bs[:, j, i].max())] for j, r in enumerate(rn)]))
    from .diagnostics import resid_tests, fevd_table
    E = np.vstack([Yt - X @ B for (Yt, X), B in zip(data, Bs)])
    tabs.append(resid_tests(E, names, title="Residual diagnostics (unit VARs, pooled residuals)"))
    I = np.array([irf(Bs[u], Ss[u], n, p, h) for u in range(N)])
    tabs.append(fevd_table(I, names))
    charts = irf_charts(I, names, h)
    for c in charts:
        c["title"] += " (median and 16-84% range across units)"
    notes = ["OLS VAR for each unit; coefficients averaged with Pesaran-Smith standard errors (s.d. across units / sqrt(N)). "
             "Impulse responses are computed unit by unit; the band shows their dispersion across units, not estimation uncertainty."]
    return tabs, charts, [stat("Units", N), stat("Periods", len(times))], notes, sum(len(d[0]) for d in data)


def _pooled(units, ids, times, names, p, h, ndraw, burn, rng, spec, dynamic=False):
    n = len(names)
    data = _unit_data(units, p)
    Yt = np.vstack([d[0] for d in data]); X = np.vstack([d[1] for d in data])
    sig = ar_sigma(np.vstack([u[np.isfinite(u).all(1)] for u in units]), p)
    lam1 = float_option(spec, "lambda1", 0.1, "Overall tightness (lambda1)")
    B0, Phi0, S0, a0 = nw_prior(n, p, sig, lam1, 1.0, 100.0, 0.0)
    Bs, Ss = nw_draws(*nw_posterior(Yt, X, B0, Phi0, S0, a0), ndraw, rng)
    tabs = [post_table(f"Pooled equation: {v}", reg_names(names, p), Bs[:, :, i]) for i, v in enumerate(names)]
    from .diagnostics import resid_tests, fevd_table
    tabs.append(resid_tests(Yt - X @ Bs.mean(0), names, title="Residual diagnostics (pooled VAR, posterior mean)"))
    I = np.array([irf(Bs[q], Ss[q], n, p, h) for q in range(ndraw)])
    tabs.append(fevd_table(I, names))
    notes = [f"One VAR for all units (homogeneous coefficients and constants), Normal-Wishart prior with Minnesota scale (lambda1 = {lam1:g}, own-lag prior mean 0)."]
    return tabs, irf_charts(I, names, h), [stat("Units", len(data)), stat("Pooled observations", len(Yt))], notes, len(Yt)


def _zh(units, ids, times, names, p, h, ndraw, burn, rng, spec, dynamic=False):
    n = len(names)
    data = _unit_data(units, p)
    Yt = np.vstack([d[0] for d in data]); X = np.vstack([d[1] for d in data])
    k = X.shape[1]
    Bp = np.linalg.lstsq(X, Yt, rcond=None)[0]
    E = Yt - X @ Bp
    s2 = float((E ** 2).sum() / (E.size - Bp.size))
    lam1 = float_option(spec, "lambda1", 0.1, "Overall tightness (lambda1)")
    bbar = Bp.flatten("F")
    Oi = np.eye(k * n) / lam1
    draws_mean = np.zeros((ndraw, k * n))
    unit_means, I_all = [], []
    for Yi, Xi in data:
        P = Oi + np.kron(np.eye(n), Xi.T @ Xi) / s2
        m = np.linalg.solve(P, Oi @ bbar + (Xi.T @ Yi).flatten("F") / s2)
        L = np.linalg.cholesky((P + P.T) / 2)
        D = m + np.linalg.solve(L.T, rng.standard_normal((k * n, ndraw))).T
        draws_mean += D / len(data)
        unit_means.append(m)
    Bm = draws_mean.reshape((ndraw, k, n), order="F")
    tabs = [post_table(f"Average unit equation: {v}", reg_names(names, p), Bm[:, :, i]) for i, v in enumerate(names)]
    U = np.array(unit_means)
    own = [U[:, i * k + 1 + i] for i in range(n)]
    tabs.append(table("Own first-lag coefficient by unit (posterior mean)", ["Unit"] + names,
                      [[ids[u]] + [f(own[i][u]) for i in range(n)] for u in range(len(data))]))
    Sig = s2 * np.eye(n)
    from .diagnostics import resid_tests
    tabs.append(resid_tests(np.vstack([Yi - Xi @ m.reshape((k, n), order="F") for (Yi, Xi), m in zip(data, unit_means)]), names,
                            title="Residual diagnostics (unit posterior means, pooled residuals)"))
    I = np.array([irf(Bm[q], Sig, n, p, h) for q in range(ndraw)])
    notes = [f"Zellner-Hong (1989): beta_i ~ N(b, lambda1 I) with b the pooled OLS estimate and a common residual variance sigma^2 I "
             f"(sigma^2 = {s2:.4g}); lambda1 = {lam1:g} sets how strongly units are shrunk towards the pool. Responses use sigma^2 I (one-s.d. shocks)."]
    return tabs, irf_charts(I, names, h), [stat("Units", len(data)), stat("Common sigma^2", s2)], notes, len(Yt)


def _hier(units, ids, times, names, p, h, ndraw, burn, rng, spec, dynamic=False):
    n = len(names)
    data = _unit_data(units, p)
    N = len(data)
    k = data[0][1].shape[1]
    q = k * n
    sig = ar_sigma(np.vstack([u[np.isfinite(u).all(1)] for u in units]), p)
    _, Vb = minnesota(n, p, sig, lam1=1.0, lam2=0.5, lam3=1.0, lam4=100.0, delta=0.0)
    hmask = np.ones((k, n)); hmask[0] = 0                     # constants are unit-specific with a flat prior
    hmask = hmask.flatten("F")
    Ob_i = np.diag(hmask / Vb.flatten("F"))
    qh = int(hmask.sum())
    s0 = v0 = 0.001
    betas = [np.linalg.lstsq(X, Y, rcond=None)[0].flatten("F") for Y, X in data]
    Sigs = [np.cov((Y - X @ b.reshape((k, n), order="F")).T).reshape(n, n) + 1e-6 * np.eye(n) for (Y, X), b in zip(data, betas)]
    dev = np.array(betas) - np.mean(betas, axis=0)
    lam = max(float(np.einsum("uq,q,uq->", dev, np.diag(Ob_i), dev)) / (N * qh), 1e-4)
    keep_b, keep_lam, keep_unit = np.empty((ndraw, q)), np.empty(ndraw), np.zeros((N, q))
    for it in range(burn + ndraw):
        bm = np.mean(betas, axis=0)
        b = bm.copy()
        b[hmask == 1] = _mvn(bm[hmask == 1], lam * Vb.flatten("F")[hmask == 1] * np.eye(qh) / N, rng)
        ssq = sum((bi - b) @ Ob_i @ (bi - b) for bi in betas)
        lam = 1 / rng.gamma((s0 + N * qh) / 2, 2 / (v0 + ssq))
        Pi = Ob_i / lam
        for u, (Y, X) in enumerate(data):
            Si = np.linalg.inv(Sigs[u])
            P = Pi + np.kron(Si, X.T @ X)
            m = np.linalg.solve(P, Pi @ b + (X.T @ Y @ Si).flatten("F"))
            betas[u] = m + np.linalg.solve(np.linalg.cholesky((P + P.T) / 2).T, rng.standard_normal(q))
            E = Y - X @ betas[u].reshape((k, n), order="F")
            Sigs[u] = np.atleast_2d(iw(E.T @ E + 1e-6 * np.eye(n), len(Y), rng))
        if it >= burn:
            keep_b[it - burn], keep_lam[it - burn] = b, lam
            keep_unit += np.array(betas) / ndraw
    Bm = keep_b.reshape((ndraw, k, n), order="F")
    tabs = [post_table(f"Hierarchical mean equation (b): {v}", reg_names(names, p), Bm[:, :, i]) for i, v in enumerate(names)]
    tabs.append(post_table("Overall shrinkage", ["lambda1"], keep_lam[:, None], "Small lambda1: units are similar and are pooled; large lambda1: units differ."))
    from .diagnostics import mcmc_table
    rn_ = reg_names(names, p)
    tabs.append(mcmc_table(np.column_stack([keep_b, keep_lam]), [f"b {names[i]}: {r}" for i in range(n) for r in rn_] + ["lambda1"]))
    tabs.append(table("Own first-lag coefficient by unit (posterior mean)", ["Unit"] + names,
                      [[ids[u]] + [f(keep_unit[u, i * k + 1 + i]) for i in range(n)] for u in range(N)]))
    Sig = np.mean(Sigs, axis=0)
    I = np.array([irf(Bm[q_], Sig, n, p, h) for q_ in range(ndraw)])
    notes = ["Jarocinski (2010): the lag coefficients beta_i ~ N(b, lambda1 Omega_b) with Omega_b Minnesota-shaped (constants unit-specific, flat prior; the constant row reports their average), a flat prior on b, lambda1 ~ IG(0.001/2, 0.001/2) "
             f"and diffuse unit covariances; Gibbs sampler with {ndraw} draws after {burn}. Responses of the hierarchical mean b use the average unit covariance."]
    return tabs, irf_charts(I, names, h), [stat("Units", N), stat("Posterior mean of lambda1", float(keep_lam.mean()))], notes, sum(len(d[0]) for d in data)


def _factor(units, ids, times, names, p, h, ndraw, burn, rng, spec, dynamic=False):
    units, ids = _balanced(units, ids)
    N, n = len(units), len(names)
    if N * n > 60:
        raise OESError(f"The factor estimators model all {N} x {n} = {N * n} series jointly; use at most 60 (fewer units or variables).")
    Y = np.hstack([u - u.mean(0) for u in units])                  # T x (N n), unit-major
    m = N * n
    Yt, X = lagmat(Y, p, const=False)                               # X: (T-p) x (m p)
    T = len(Yt)
    unit_of = np.repeat(np.arange(N), n)
    var_of = np.tile(np.arange(n), N)
    # factor regressors W_t (m x f): coefficient on regressor c in equation e =
    #   theta_common + theta_unit[unit(e)] + theta_var[var(e)] + theta_own 1(c is e's own lag) + theta_dom 1(unit(c) = unit(e)) + theta_lag[lag(c)]
    fnames = ["common"] + [f"unit:{u}" for u in ids] + [f"variable:{v}" for v in names] + ["own lag", "same unit (domestic)"] + [f"lag {l}" for l in range(1, p + 1)]
    fdim = len(fnames)
    W = np.zeros((T, m, fdim))
    for l in range(p):
        Xl = X[:, l * m:(l + 1) * m]                                # lag l+1 of all series
        tot = Xl.sum(1)
        for e in range(m):
            dom = Xl[:, unit_of == unit_of[e]].sum(1)
            W[:, e, 0] += tot
            W[:, e, 1 + unit_of[e]] += tot
            W[:, e, 1 + N + var_of[e]] += tot
            W[:, e, 1 + N + n] += Xl[:, e]
            W[:, e, 2 + N + n] += dom
            W[:, e, 3 + N + n + l] += tot
    s2 = Yt.var(0) + 1e-8
    prior_v = float_option(spec, "prior_var", 1.0, "Prior variance of the factors")
    keep_th = np.empty((ndraw, T, fdim)) if dynamic else np.empty((ndraw, fdim))
    keep_s2 = np.empty((ndraw, m))
    theta = np.zeros(fdim)
    TH = np.zeros((T, fdim))
    qv = 0.01
    for it in range(burn + ndraw):
        if not dynamic:
            P = np.eye(fdim) / prior_v + np.einsum("tef,e,teg->fg", W, 1 / s2, W)
            rhs = np.einsum("tef,e,te->f", W, 1 / s2, Yt)
            theta = _mvn(np.linalg.solve(P, rhs), np.linalg.inv(P), rng)
            fit = np.einsum("tef,f->te", W, theta)
        else:
            tf, Pf = np.empty((T, fdim)), np.empty((T, fdim, fdim))
            th, P = np.zeros(fdim), prior_v * np.eye(fdim)
            R = np.diag(s2)
            for t in range(T):
                P = P + qv * np.eye(fdim)
                Z = W[t]
                F = Z @ P @ Z.T + R
                K = np.linalg.solve(F, Z @ P).T
                th = th + K @ (Yt[t] - Z @ th)
                P = P - K @ Z @ P
                tf[t], Pf[t] = th, P
            TH[-1] = _mvn(tf[-1], Pf[-1], rng)
            for t in range(T - 2, -1, -1):
                G = np.linalg.solve(Pf[t] + qv * np.eye(fdim), Pf[t]).T
                TH[t] = _mvn(tf[t] + G @ (TH[t + 1] - tf[t]), Pf[t] - G @ Pf[t], rng)
            dth = np.diff(np.vstack([np.zeros(fdim), TH]), axis=0)
            qv = 1 / rng.gamma(1 + T * fdim / 2, 1 / (0.001 + (dth ** 2).sum() / 2))
            fit = np.einsum("tef,tf->te", W, TH)
        E = Yt - fit
        s2 = 1 / rng.gamma(1 + T / 2, 1 / (0.01 + (E ** 2).sum(0) / 2))
        if it >= burn:
            keep_th[it - burn] = TH if dynamic else theta
            keep_s2[it - burn] = s2
    tabs, charts = [], []
    thmean = keep_th.mean(1) if dynamic else keep_th
    tabs.append(post_table("Factors" + (" (averaged over time)" if dynamic else ""), fnames, thmean,
                           "Each coefficient of the panel VAR is the sum of the factors that apply to it (see the notes)."))
    from .diagnostics import mcmc_table
    tabs.append(mcmc_table(np.column_stack([thmean, keep_s2]), fnames + [f"sigma2 {ids[e // n]}:{names[e % n]}" for e in range(m)]))

    def coef_matrix(th):
        """Implied (m p) x m coefficient matrix from factor values th."""
        B = np.zeros((m * p, m))
        for l in range(p):
            for e in range(m):
                for c in range(m):
                    B[l * m + c, e] = (th[0] + th[1 + unit_of[e]] + th[1 + N + var_of[e]] + th[1 + N + n] * (c == e)
                                       + th[2 + N + n] * (unit_of[c] == unit_of[e]) + th[3 + N + n + l])
        return B

    labs = times[p:]
    if dynamic:
        for j in [0, 1 + N + n, 2 + N + n]:
            D = keep_th[:, :, j]
            charts.append(chart(f"Time-varying factor: {fnames[j]}", labs, [("median", np.median(D, 0))], "", "factor",
                                band=(np.percentile(D, 16, 0), np.percentile(D, 84, 0))))
    step = max(1, ndraw // 200)
    resp = []
    for q_ in range(0, ndraw, step):
        th = keep_th[q_, -1] if dynamic else keep_th[q_]
        B = np.vstack([np.zeros((1, m)), coef_matrix(th)])
        S = np.diag(keep_s2[q_])
        I = irf(B, S, m, p, h)                                       # (h+1) x m x m
        # average over units: response of variable i in the shocked unit (own) and in the other units (spillover)
        own = np.zeros((h + 1, n, n)); spill = np.zeros((h + 1, n, n))
        for u in range(N):
            for j in range(n):
                s_idx = u * n + j
                for i in range(n):
                    own[:, i, j] += I[:, u * n + i, s_idx] / N
                    others = [w * n + i for w in range(N) if w != u]
                    spill[:, i, j] += I[:, others, s_idx].mean(1) / N
        resp.append((own, spill))
    own = np.array([r[0] for r in resp]); spill = np.array([r[1] for r in resp])
    for j in range(n):
        for i in range(n):
            if len(charts) >= 12:
                break
            charts.append(chart(f"Response of {names[i]} to a {names[j]} shock: same unit vs other units", list(range(h + 1)),
                                [("same unit", np.median(own[:, :, i, j], 0)), ("other units (spillover)", np.median(spill[:, :, i, j], 0))], "horizon", names[i]))
    notes = ["Canova and Ciccarelli (2009) factor structure: every equation contains the lags of all variables of all units; coefficient = common "
             "factor + factor of the equation's unit + factor of the equation's variable + own-lag factor (if the regressor is the equation's own lag) "
             "+ domestic factor (if the regressor is from the same unit) + lag factor. Series are demeaned by unit; diagonal residual variances; "
             f"factors ~ N(0, {prior_v:g}). Gibbs sampler, {ndraw} draws after {burn}.",
             "Responses are to one-standard-deviation shocks (diagonal covariance), averaged over the shocked units."]
    if dynamic:
        notes.append("Dynamic version: the factors follow random walks (Carter-Kohn); responses use the factors of the last period.")
    return tabs, charts, [stat("Units", N), stat("Series modelled jointly", m), stat("Factors", fdim)], notes, T * N
