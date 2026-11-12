"""Markov-switching models.

ms_reg and ms_ar use statsmodels' MarkovRegression and MarkovAutoregression (Hamilton 1989; Kim and Nelson 1999).
msvar and mspvar are estimated here by the EM algorithm (Hamilton 1990; Krolzig 1997): Hamilton filter, Kim
smoother and weighted least squares. The panel version shares one regime path across all units (synchronised
regimes) with pooled regime-dependent coefficients; the likelihood of a period is the product over units.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .bayes import irf, lagmat, reg_names
from .core import OESError, as_list, chart, distinct, f, int_option, need, numeric, result, stat, table, time_order
from .timeseries import _series_frame


# ================================================================================================ statsmodels models
def _ms_tables(res, k, labels, y, names_y):
    pn = list(res.model.param_names)
    bse = np.asarray(res.bse, float) if res.bse is not None else np.full(len(pn), np.nan)
    params = np.asarray(res.params, float)
    z = params / bse
    rows = [[nm, f(b), f(s), f(zz, 4), f(2 * stats.norm.sf(abs(zz)), 4) if np.isfinite(zz) else None] for nm, b, s, zz in zip(pn, params, bse, z)]
    tabs = [table("Parameter estimates", ["Parameter", "Estimate", "Std. err.", "z", "p-value"], rows,
                  "p[i->j] is the probability of moving from regime i to regime j; [r] marks a regime-specific parameter.")]
    P = np.asarray(res.regime_transition)[..., 0]               # P[i, j] = Pr(s_t = i | s_{t-1} = j)
    tabs.append(table("Transition probabilities (row: from, column: to)", ["From"] + [f"Regime {j}" for j in range(k)],
                      [[f"Regime {i}"] + [f(P[j, i], 4) for j in range(k)] for i in range(k)]))
    sm_ = np.asarray(res.smoothed_marginal_probabilities)
    if sm_.ndim == 1:
        sm_ = np.column_stack([sm_, 1 - sm_])
    dur = np.asarray(res.expected_durations, float).ravel()
    yy = np.asarray(y)[-len(sm_):]
    reg = sm_.argmax(1)
    tabs.append(table("Regimes", ["Regime", "Expected duration (periods)", "Share of periods (smoothed prob. > 0.5)", f"Mean of {names_y} in the regime"],
                      [[f"Regime {j}", f(dur[j], 3), f((sm_[:, j] > 0.5).mean(), 3), f(yy[reg == j].mean() if (reg == j).any() else None)] for j in range(k)]))
    lab = labels[-len(sm_):]
    charts = [chart("Smoothed regime probabilities", lab, [(f"Regime {j}", sm_[:, j]) for j in range(k)], "", "probability"),
              chart(f"{names_y} and its regime-dependent fitted mean", lab, [(names_y, yy), ("fitted", np.asarray(res.fittedvalues)[-len(sm_):])])]
    return tabs, charts, sm_


def _fit(mod, rng_seed=0):
    np.random.seed(rng_seed)
    best = None
    for kw in ({}, {"search_reps": 20}):
        try:
            r = mod.fit(disp=False, **kw)
            if best is None or (np.isfinite(r.llf) and r.llf > best.llf + 1e-6):
                best = r
        except Exception:
            continue
    if best is None:
        raise OESError("The Markov-switching model could not be estimated; try fewer regimes or fewer switching parameters.")
    return best


def ms_reg(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "dependent variable")
    x = as_list(spec.get("x"))
    distinct([y], x)
    d, labels = _series_frame(df, spec, [y] + x)
    k = int_option(spec, "regimes", 2, 2, 4, "Number of regimes")
    sv = str(spec.get("switching_variance", "yes")).lower() in ("yes", "true", "1")
    sx = str(spec.get("switching_exog", "yes")).lower() in ("yes", "true", "1")
    mod = sm.tsa.MarkovRegression(d[y].astype(float).reset_index(drop=True), k_regimes=k, trend="c",
                                  exog=d[x].astype(float).reset_index(drop=True) if x else None,
                                  switching_exog=sx, switching_variance=sv)
    res = _fit(mod)
    tabs, charts, _ = _ms_tables(res, k, labels, d[y].to_numpy(float), y)
    summary = [stat("Observations", int(res.nobs)), stat("Regimes", k), stat("Log-likelihood", res.llf, 3), stat("AIC", res.aic, 3), stat("BIC", res.bic, 3)]
    notes = [f"Markov-switching regression (Hamilton 1989) with a switching constant" + (", switching slopes" if x and sx else "")
             + (" and switching variance" if sv else "") + "; maximum likelihood (EM, then quasi-Newton), standard errors from the Hessian."]
    return result("ms_reg", f"Markov-switching regression: {y}", res.nobs, summary, tabs, charts, notes)


def ms_ar(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "dependent variable")
    d, labels = _series_frame(df, spec, [y])
    k = int_option(spec, "regimes", 2, 2, 4, "Number of regimes")
    order = int_option(spec, "order", 4, 1, 8, "AR order")
    sa = str(spec.get("switching_ar", "no")).lower() in ("yes", "true", "1")
    sv = str(spec.get("switching_variance", "no")).lower() in ("yes", "true", "1")
    mod = sm.tsa.MarkovAutoregression(d[y].astype(float).reset_index(drop=True), k_regimes=k, order=order, switching_ar=sa, switching_variance=sv)
    res = _fit(mod)
    tabs, charts, sm_ = _ms_tables(res, k, labels, d[y].to_numpy(float), y)
    summary = [stat("Observations", int(res.nobs)), stat("Regimes", k), stat("AR order", order), stat("Log-likelihood", res.llf, 3),
               stat("AIC", res.aic, 3), stat("BIC", res.bic, 3)]
    notes = ["Hamilton's (1989) Markov-switching autoregression: the regime shifts the mean of the series (the AR part applies to "
             "deviations from the regime mean)" + ("; AR coefficients switch" if sa else "") + ("; variance switches" if sv else "") + "."]
    return result("ms_ar", f"Markov-switching AR({order}): {y}", res.nobs, summary, tabs, charts, notes)


# ================================================================================================ EM for MS-VAR / MS panel VAR
def _mvn_logpdf(E, S):
    n = S.shape[0]
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, E.T)
    return -0.5 * (n * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + (z ** 2).sum(0))


def _ms_em(units, K, kind, maxit=5000, tol=1e-9, starts=6, seed=0):
    """units: list of (Y, X) with the same T; X includes no constant (intercepts switch). Returns the best fit."""
    T = units[0][0].shape[0]
    n = units[0][0].shape[1]
    q = units[0][1].shape[1]
    rng = np.random.default_rng(seed)
    best = None
    ybar = np.mean([u[0][:, 0] for u in units], axis=0)
    for s_ in range(starts):
        if s_ == 0:                                  # start from a split of the first variable at its quantiles
            cut = np.quantile(ybar, np.linspace(0, 1, K + 1)[1:-1])
            lab = np.searchsorted(cut, ybar)
            xi = np.eye(K)[lab] * 0.8 + 0.2 / K
        else:
            xi = rng.dirichlet(np.ones(K), size=T)
        P = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(P, 0.9)                      # P[i, j] = Pr(s_t = j | s_{t-1} = i)
        llf_old = -np.inf
        for it in range(maxit):
            nu, A, S = _ms_mstep(units, xi, K, kind, n, q)
            # E-step: Hamilton filter
            logf = np.zeros((T, K))
            for (Y, X) in units:
                for j in range(K):
                    E = Y - nu[j] - (X @ A[j] if q else 0)
                    logf[:, j] += _mvn_logpdf(E, S[j])
            mx = logf.max(1, keepdims=True)
            fdens = np.exp(logf - mx)
            w, vec = np.linalg.eig(P.T)
            erg = np.real(vec[:, np.argmin(np.abs(w - 1))]); erg = np.abs(erg) / np.abs(erg).sum()
            filt, pred = np.empty((T, K)), np.empty((T, K))
            llf = 0.0
            prev = erg
            for t in range(T):
                pr = prev @ P
                pred[t] = pr
                num = pr * fdens[t]
                c = num.sum()
                llf += np.log(c) + mx[t, 0]
                filt[t] = num / c
                prev = filt[t]
            sm_ = np.empty((T, K))
            sm_[-1] = filt[-1]
            joint = np.zeros((K, K))
            for t in range(T - 2, -1, -1):
                ratio = sm_[t + 1] / np.maximum(pred[t + 1], 1e-300)
                sm_[t] = filt[t] * (P @ ratio)
                joint += filt[t][:, None] * P * ratio[None, :]
            P = joint / joint.sum(1, keepdims=True)
            P = np.clip(P, 1e-10, 1); P /= P.sum(1, keepdims=True)
            xi = sm_
            if abs(llf - llf_old) < tol:
                break
            llf_old = llf
        nu, A, S = _ms_mstep(units, xi, K, kind, n, q)
        if best is None or llf > best["llf"]:
            best = dict(llf=llf, nu=nu, A=A, S=S, P=P, smoothed=xi, filtered=filt, iterations=it + 1)
    # order regimes by the intercept of the first variable (regime 0 = lowest)
    o = np.argsort([best["nu"][j][0] for j in range(K)])
    best.update(nu=[best["nu"][j] for j in o], A=[best["A"][j] for j in o], S=[best["S"][j] for j in o],
                P=best["P"][np.ix_(o, o)], smoothed=best["smoothed"][:, o], filtered=best["filtered"][:, o])
    return best


def _ms_mstep(units, xi, K, kind, n, q):
    switch_A = kind == "MSIAH"
    switch_S = kind in ("MSIH", "MSIAH")
    W = xi.sum(0) * len(units)
    if switch_A:
        nu, A, S = [], [], []
        for j in range(K):
            ZZ = np.zeros((1 + q, 1 + q)); ZY = np.zeros((1 + q, n))
            for (Y, X) in units:
                Z = np.column_stack([np.ones(len(Y)), X]) if q else np.ones((len(Y), 1))
                Zw = Z * xi[:, [j]]
                ZZ += Zw.T @ Z; ZY += Zw.T @ Y
            Bj = np.linalg.solve(ZZ + 1e-10 * np.eye(1 + q), ZY)
            nu.append(Bj[0]); A.append(Bj[1:] if q else np.zeros((0, n)))
        for j in range(K):
            Sj = np.zeros((n, n))
            for (Y, X) in units:
                E = Y - nu[j] - (X @ A[j] if q else 0)
                Sj += (E * xi[:, [j]]).T @ E
            S.append(Sj / max(W[j], 1e-10) + 1e-8 * np.eye(n))
        if not switch_S:
            Sc = sum(S[j] * W[j] for j in range(K)) / W.sum()
            S = [Sc] * K
        return nu, A, S
    # common slopes: GLS with regime-specific covariances, iterated twice (ECM)
    S = [np.eye(n)] * K
    nu = [np.zeros(n)] * K
    A = np.zeros((q, n))
    for _ in range(2 if switch_S else 1):
        Si = [np.linalg.inv(s) for s in S]
        dim = K * n + q * n
        M = np.zeros((dim, dim)); v = np.zeros(dim)
        for (Y, X) in units:
            for j in range(K):
                w = xi[:, j]
                Wj = Si[j]
                # blocks: intercept of regime j (n), slopes vec(A) (q n, column-major: equation by equation)
                sw = w.sum()
                ij = slice(j * n, (j + 1) * n)
                M[ij, ij] += sw * Wj
                v[ij] += Wj @ (w @ Y)
                if q:
                    XwX = (X * w[:, None]).T @ X
                    Xw1 = X.T @ w
                    sl = slice(K * n, dim)
                    M[sl, sl] += np.kron(Wj, XwX)
                    cross = np.kron(Wj, Xw1[:, None])            # (q n) x n
                    M[sl, ij] += cross
                    M[ij, sl] += cross.T
                    v[sl] += ((X * w[:, None]).T @ Y @ Wj).flatten("F")
        th = np.linalg.solve(M + 1e-10 * np.eye(dim), v)
        nu = [th[j * n:(j + 1) * n] for j in range(K)]
        A = th[K * n:].reshape((q, n), order="F") if q else np.zeros((0, n))
        S = []
        for j in range(K):
            Sj = np.zeros((n, n))
            for (Y, X) in units:
                E = Y - nu[j] - (X @ A if q else 0)
                Sj += (E * xi[:, [j]]).T @ E
            S.append(Sj / max(W[j], 1e-10) + 1e-8 * np.eye(n))
        if not switch_S:
            Sc = sum(S[j] * W[j] for j in range(K)) / W.sum()
            S = [Sc] * K
    return nu, [A] * K, S


def _linear_llf(units):
    ll = 0.0
    n = units[0][0].shape[1]
    Ys = np.vstack([u[0] for u in units])
    Zs = np.vstack([np.column_stack([np.ones(len(u[0])), u[1]]) for u in units])
    B = np.linalg.lstsq(Zs, Ys, rcond=None)[0]
    E = Ys - Zs @ B
    S = E.T @ E / len(E)
    return float(_mvn_logpdf(E, S).sum()), B.size + n * (n + 1) // 2


def _ms_output(model_id, title, fit, names, p, K, kind, labels, units, extra_notes, h=20):
    n = len(names)
    q = n * p
    nu, A, S, P, sm_ = fit["nu"], fit["A"], fit["S"], fit["P"], fit["smoothed"]
    T = sm_.shape[0]
    tabs = [table("Regime-dependent intercepts", ["Variable"] + [f"Regime {j}" for j in range(K)], [[v] + [f(nu[j][i]) for j in range(K)] for i, v in enumerate(names)])]
    means = []
    for j in range(K):
        Asum = sum(A[j][(l - 1) * n:l * n].T for l in range(1, p + 1)) if q else np.zeros((n, n))
        try:
            means.append(np.linalg.solve(np.eye(n) - Asum, nu[j]))
        except np.linalg.LinAlgError:
            means.append(np.full(n, np.nan))
    tabs.append(table("Implied regime means (steady state within each regime)", ["Variable"] + [f"Regime {j}" for j in range(K)],
                      [[v] + [f(means[j][i]) for j in range(K)] for i, v in enumerate(names)]))
    tabs.append(table("Transition probabilities (row: from, column: to)", ["From"] + [f"Regime {j}" for j in range(K)],
                      [[f"Regime {i}"] + [f(P[i, j], 4) for j in range(K)] for i in range(K)]))
    dur = 1 / (1 - np.diag(P))
    w, vec = np.linalg.eig(P.T)
    erg = np.real(vec[:, np.argmin(np.abs(w - 1))]); erg = np.abs(erg) / np.abs(erg).sum()
    tabs.append(table("Regimes", ["Regime", "Expected duration", "Ergodic probability", "Share of periods (smoothed prob. > 0.5)"],
                      [[f"Regime {j}", f(dur[j], 3), f(erg[j], 4), f((sm_[:, j] > 0.5).mean(), 3)] for j in range(K)]))
    if q:
        rn = reg_names(names, p, const=False)
        for j in range(K if kind == "MSIAH" else 1):
            tabs.append(table(f"Autoregressive coefficients" + (f", regime {j}" if kind == "MSIAH" else " (common to all regimes)"),
                              ["Regressor"] + names, [[r] + [f(A[j][ri, i]) for i in range(n)] for ri, r in enumerate(rn)]))
    for j in range(K if kind != "MSI" else 1):
        tabs.append(table("Residual covariance" + (f", regime {j}" if kind != "MSI" else " (common)"), [""] + names,
                          [[a] + [f(S[j][ia, ib]) for ib in range(n)] for ia, a in enumerate(names)]))
    llf = fit["llf"]
    npar = K * n + (K if kind == "MSIAH" else 1) * q * n + (K if kind != "MSI" else 1) * n * (n + 1) // 2 + K * (K - 1)
    nobs = T * len(units)
    lin, kl = _linear_llf(units)
    LR = 2 * (llf - lin)
    tabs.append(table("Model comparison", ["Statistic", "Value"],
                      [["Log-likelihood (Markov switching)", f(llf, 3)], ["Log-likelihood (linear VAR)", f(lin, 3)],
                       ["LR statistic", f(LR, 3)], ["Davies upper-bound p-value (approx.)", f(stats.chi2.sf(LR, max(npar - kl, 1)), 4)],
                       ["AIC (MS)", f(-2 * llf + 2 * npar, 3)], ["AIC (linear)", f(-2 * lin + 2 * kl, 3)],
                       ["BIC (MS)", f(-2 * llf + np.log(nobs) * npar, 3)], ["BIC (linear)", f(-2 * lin + np.log(nobs) * kl, 3)]],
                      "Under the null of no switching the transition probabilities are unidentified, so the chi-square p-value is only a rough guide; prefer the information criteria."))
    lab = labels[-T:]
    charts = [chart("Smoothed regime probabilities", lab, [(f"Regime {j}", sm_[:, j]) for j in range(K)], "", "probability")]
    for i, v in enumerate(names[:3]):
        series = np.mean([u[0][:, i] for u in units], axis=0)
        charts.append(chart(f"{v}{' (average over units)' if len(units) > 1 else ''} and the regime-weighted mean", lab,
                            [(v, series), ("regime mean", sm_ @ np.array([m[i] for m in means]))]))
    if q:
        for j in range(n):
            for i in range(n):
                if len(charts) >= 4 + 9:
                    break
                ser = []
                for r in range(K if kind != "MSI" else 1):
                    B = np.vstack([nu[r][None, :], A[r]])
                    ser.append((f"Regime {r}" if kind != "MSI" else "all regimes", irf(B, S[r], n, p, h)[:, i, j]))
                charts.append(chart(f"Response of {names[i]} to a shock in {names[j]}", list(range(h + 1)), ser, "horizon", names[i]))
    summary = [stat("Periods", T), stat("Units", len(units)), stat("Regimes", K), stat("Specification", kind), stat("Lags", p),
               stat("Log-likelihood", llf, 3), stat("EM iterations", fit["iterations"])]
    notes = [{"MSI": "MSI-VAR: only the intercepts switch.", "MSIH": "MSIH-VAR: intercepts and covariance matrices switch; autoregressive coefficients are common.",
              "MSIAH": "MSIAH-VAR: intercepts, autoregressive coefficients and covariance matrices all switch."}[kind],
             "Maximum likelihood by the EM algorithm (Hamilton filter, Kim smoother) from several starting values; regimes are ordered by the "
             "intercept of the first variable. EM gives point estimates without standard errors. Impulse responses are regime-specific "
             "Cholesky responses (no switching over the horizon)."] + extra_notes
    return result(model_id, title, nobs, summary, tabs, charts, notes)


def _kind(spec):
    kind = str(spec.get("switching", "MSIH")).upper()
    if kind not in ("MSI", "MSIH", "MSIAH"):
        raise OESError("Switching must be MSI, MSIH or MSIAH.")
    return kind


def msvar(df, spec):
    names = as_list(need(spec, "vars", "variables"))
    distinct(names)
    d, labels = _series_frame(df, spec, names)
    Y = d[names].to_numpy(float)
    K = int_option(spec, "regimes", 2, 2, 4, "Number of regimes")
    p = int_option(spec, "lags", 1, 0, 8, "Lags")
    kind = _kind(spec)
    if p:
        Yt, X = lagmat(Y, p, const=False)
    else:
        Yt, X = Y, np.zeros((len(Y), 0))
    fit = _ms_em([(Yt, X)], K, kind, starts=int_option(spec, "starts", 6, 1, 50, "Starting values"))
    return _ms_output("msvar", f"Markov-switching VAR ({kind}, {K} regimes)", fit, names, p, K, kind, labels, [(Yt, X)], [])


def mspvar(df, spec):
    names = as_list(need(spec, "vars", "variables"))
    unit, time = need(spec, "unit", "unit identifier"), need(spec, "time", "time identifier")
    distinct([unit, time], names)
    d = numeric(df, names, keep=[unit, time])
    d = time_order(d, time)
    K = int_option(spec, "regimes", 2, 2, 4, "Number of regimes")
    p = int_option(spec, "lags", 1, 0, 4, "Lags")
    kind = _kind(spec)
    demean = str(spec.get("demean", "yes")).lower() in ("yes", "true", "1")
    times = sorted(d[time].astype(str).unique(), key=lambda s: (pd.to_numeric(s, errors="coerce"), s))
    units = []
    for u, g in d.groupby(unit, sort=False):
        g = g.drop_duplicates(time).set_index(g[time].astype(str))
        if len(g) != len(times):
            continue
        Y = g.loc[times, names].to_numpy(float)
        if demean:
            Y = Y - Y.mean(0)
        if p:
            Yt, X = lagmat(Y, p, const=False)
        else:
            Yt, X = Y, np.zeros((len(Y), 0))
        units.append((Yt, X))
    dropped = d[unit].nunique() - len(units)
    if len(units) < 2:
        raise OESError("The Markov-switching panel VAR needs a balanced panel with at least two complete units.")
    fit = _ms_em(units, K, kind, starts=int_option(spec, "starts", 6, 1, 50, "Starting values"))
    notes = [f"Panel version: one regime path shared by all {len(units)} units (synchronised regimes), pooled regime-dependent coefficients; "
             + ("each unit's series are demeaned first, so the regime intercepts are deviations from unit means." if demean else "no unit fixed effects.")]
    if dropped:
        notes.append(f"{dropped} unit(s) with missing periods were left out (a balanced panel is required).")
    return _ms_output("mspvar", f"Markov-switching panel VAR ({kind}, {K} regimes)", fit, names, p, K, kind, times, units, notes)
