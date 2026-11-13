"""Markov-switching models.

ms_reg and ms_ar use statsmodels' MarkovRegression and MarkovAutoregression (Hamilton 1989; Kim and Nelson 1999).
msvar and mspvar are estimated here by the EM algorithm (Hamilton 1990; Krolzig 1997): Hamilton filter, Kim
smoother and weighted least squares. The panel version shares one regime path across all units (synchronised
regimes) with pooled regime-dependent coefficients; the likelihood of a period is the product over units.
"""
import math

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
              chart(f"{names_y} and its regime-dependent fitted mean", lab, [(names_y, yy), ("fitted", _align(res.fittedvalues, len(sm_)))])]
    return tabs, charts, sm_


def _fit(mod, rng_seed=0, quick=False, warm=None):
    """Maximum likelihood from the default start, random starts and (if given) the solution of a restricted model (the likelihood has local maxima); keeps the highest."""
    best = None
    import warnings
    runs = [({}, rng_seed)] if quick else [({}, rng_seed)] + [({"search_reps": 20}, rng_seed + j) for j in range(4)]
    for kw, sd in runs:
        np.random.seed(sd)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = mod.fit(disp=False, **kw)
            if best is None or (np.isfinite(r.llf) and r.llf > best.llf + 1e-6):
                best = r
        except Exception:
            continue
    if best is not None and warm is not None:
        try:
            names = list(mod.param_names)
            sp = np.array(mod.start_params, float)
            rn, rp = list(warm.model.param_names), np.asarray(warm.params, float)
            for i, nm in enumerate(names):
                base = nm.split("[")[0]
                m_ = [j for j, q in enumerate(rn) if q == nm] or [j for j, q in enumerate(rn) if q.split("[")[0] == base]
                if m_:
                    sp[i] = rp[m_[0]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = mod.fit(start_params=sp, disp=False)
            if np.isfinite(r.llf) and r.llf > best.llf + 1e-6:
                best = r
        except Exception:
            pass
    if best is None:
        raise OESError("The Markov-switching model could not be estimated; try fewer regimes or fewer switching parameters.")
    return best


def _align(v, n):
    """Last n values of v, padded with NaN at the start if v is shorter (lagged models drop observations in the fitted series)."""
    v = np.asarray(v, float).ravel()
    return v[-n:] if len(v) >= n else np.concatenate([np.full(n - len(v), np.nan), v])


def _collapse_notes(res):
    """Warn when a regime is tiny or its variance is ~0: the likelihood then peaks on one or two outliers (a degenerate solution)."""
    try:
        sm_ = np.asarray(res.smoothed_marginal_probabilities)
        if sm_.ndim == 1:
            sm_ = np.column_stack([sm_, 1 - sm_])
        out, n = [], sm_.shape[0]
        for j in range(sm_.shape[1]):
            cnt = int((sm_.argmax(1) == j).sum())
            if cnt < max(5, 0.05 * n):
                out.append(f"Regime {j} holds only {cnt} of {n} periods: the solution has probably collapsed onto a few outliers. "
                           "Treat the regimes as unreliable: use fewer regimes or fewer switching parameters, or check for outlying observations.")
        vp = [q for q, nm in zip(np.asarray(res.params, float), res.model.param_names) if nm.startswith("sigma2")]
        if vp and min(vp) < 1e-4 * max(max(vp), 1e-12):
            out.append("A regime has a variance of almost zero, a sign of a degenerate solution (standard errors are then meaningless).")
        return out
    except Exception:
        return []


def _ref_spec(spec, df):
    ref = spec.get("reference")
    if ref in (None, ""):
        return None
    if ref not in df.columns:
        raise OESError(f"Reference column {ref} not found.")
    return ref


def _sm_paper_tables(res, mod_factory, y, labels, kmax, ref=None, refvals=None):
    """Unit roots, number of regimes (k = 1..kmax), regime classification, dating, residual and reference checks."""
    from . import diagnostics as D
    pre, post = [D.unit_roots(np.asarray(y, float)[:, None], ["y"])], []
    pre[0]["rows"][0][0] = "dependent variable"
    rows = []
    for k_ in range(1, kmax + 1):
        try:
            if k_ == 1:
                r_ = mod_factory(1)
                if hasattr(r_, "df_model") and not hasattr(r_, "hqic"):    # OLS: count the variance as a parameter, as the switching models do
                    n_, q_ = float(r_.nobs), len(r_.params) + 1
                    rows.append([k_, f(r_.llf, 3), f(2 * q_ - 2 * r_.llf, 3), f(q_ * np.log(n_) - 2 * r_.llf, 3),
                                 f(2 * q_ * np.log(np.log(n_)) - 2 * r_.llf, 3)])
                    continue
            else:
                r_ = res if k_ == res.model.k_regimes else _fit(mod_factory(k_), quick=True)
            rows.append([k_, f(r_.llf, 3), f(r_.aic, 3), f(r_.bic, 3), f(r_.hqic, 3)])
        except Exception:
            rows.append([k_, None, None, None, None])
    ok = [r_ for r_ in rows if r_[2] is not None]
    lin = ok[0] if ok and ok[0][0] == 1 else None
    pre.append(table("Number of regimes: information criteria", ["Regimes", "Log-likelihood", "AIC", "BIC", "HQIC"], rows,
                     "Regimes = 1 is the linear model. Under the null of no switching some parameters are unidentified, so a standard LR test "
                     "is not valid (Hansen 1992; Garcia 1998): compare models with the criteria and with the economic meaning of the regimes."))
    if lin and len(ok) > 1:
        two = [r_ for r_ in ok if r_[0] == res.model.k_regimes][0]
        pre.append(table("Linearity: likelihood-ratio statistic", ["Statistic", "Value"],
                         [["LR = 2 (logL switching - logL linear)", f(2 * (two[1] - lin[1]), 3)],
                          ["Garcia (1998) 5% critical value, two-regime models (approximate)", "about 14"],
                          ["Davies upper-bound p-value", f(_davies(2 * (two[1] - lin[1]), max(int(res.model.k_params) - 2, 1)), 4)]],
                         "A large LR (well above the Garcia critical value) rejects the linear model."))
    sm_ = np.asarray(res.smoothed_marginal_probabilities)
    if sm_.ndim == 1:
        sm_ = np.column_stack([sm_, 1 - sm_])
    lab = labels[-len(sm_):]
    post.append(table("Regime classification", ["Measure", "Value"], [["Regime classification measure, RCM (Ang and Bekaert 2002)", f(D.rcm(sm_), 2)]],
                      "RCM = 0: every period is assigned to a regime with certainty; RCM = 100: no information. Below about 30 is a sharp classification."))
    post.append(D.regime_dates(sm_, lab))
    fv = np.asarray(res.fittedvalues, float)
    m_ = min(len(sm_), len(fv))
    resid = np.asarray(y, float)[-m_:] - fv[-m_:]
    post.append(D.resid_tests(resid[:, None], ["residual"], title="Residual diagnostics (observed minus regime-weighted fit)"))
    if ref is not None:
        r_ = np.asarray(refvals, float)[-len(sm_):]
        best = max(range(sm_.shape[1]), key=lambda j: (((sm_[:, j] > .5) == (r_ > .5)).mean()))
        post.append(table(f"Comparison with {ref} (regime {best})", ["Measure", "Value"], D.concordance(sm_, r_, best)))
    return pre, post


def _refcol(df, d, ref):
    if ref is None:
        return None
    return pd.to_numeric(df.loc[d.index, ref], errors="coerce").fillna(0).to_numpy()


def _davies(lr, q):
    """Davies (1987) upper bound for a test with q nuisance parameters unidentified under the null (one-dimensional)."""
    lr = max(lr, 1e-12)
    return float(min(1.0, stats.chi2.sf(lr, q) + lr ** ((q - 1) / 2) * np.exp(-lr / 2) * 2 ** (-q / 2) / math.gamma(q / 2)))


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
    warm = None
    if x and sx:
        try:
            warm = _fit(sm.tsa.MarkovRegression(d[y].astype(float).reset_index(drop=True), k_regimes=k, trend="c",
                                                exog=d[x].astype(float).reset_index(drop=True), switching_exog=False, switching_variance=sv), quick=True)
        except Exception:
            warm = None
    res = _fit(mod, warm=warm)
    tabs, charts, _ = _ms_tables(res, k, labels, d[y].to_numpy(float), y)
    ref = _ref_spec(spec, df)
    yy = d[y].astype(float).reset_index(drop=True)
    ex = d[x].astype(float).reset_index(drop=True) if x else None

    def factory(kk):
        if kk == 1:
            return sm.OLS(yy, sm.add_constant(ex) if x else np.ones(len(yy))).fit()
        return sm.tsa.MarkovRegression(yy, k_regimes=kk, trend="c", exog=ex, switching_exog=sx, switching_variance=sv)
    pre, post = _sm_paper_tables(res, factory, yy.to_numpy(), labels, min(3, k + 1), ref, _refcol(df, d, ref))
    tabs = pre + tabs + post
    summary = [stat("Observations", int(res.nobs)), stat("Regimes", k), stat("Log-likelihood", res.llf, 3), stat("AIC", res.aic, 3), stat("BIC", res.bic, 3)]
    notes = [f"Markov-switching regression (Hamilton 1989) with a switching constant" + (", switching slopes" if x and sx else "")
             + (" and switching variance" if sv else "") + "; maximum likelihood (EM, then quasi-Newton), standard errors from the Hessian."]
    return result("ms_reg", f"Markov-switching regression: {y}", res.nobs, summary, tabs, charts, notes + _collapse_notes(res))


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
    ref = _ref_spec(spec, df)
    yy = d[y].astype(float).reset_index(drop=True)

    def factory(kk):
        if kk == 1:
            from statsmodels.tsa.ar_model import AutoReg
            return AutoReg(yy, lags=order, trend="c").fit()
        return sm.tsa.MarkovAutoregression(yy, k_regimes=kk, order=order, switching_ar=sa, switching_variance=sv)
    pre, post = _sm_paper_tables(res, factory, yy.to_numpy(), labels, min(3, k + 1), ref, _refcol(df, d, ref))
    tabs = pre + tabs + post
    summary = [stat("Observations", int(res.nobs)), stat("Regimes", k), stat("AR order", order), stat("Log-likelihood", res.llf, 3),
               stat("AIC", res.aic, 3), stat("BIC", res.bic, 3)]
    notes = ["Hamilton's (1989) Markov-switching autoregression: the regime shifts the mean of the series (the AR part applies to "
             "deviations from the regime mean)" + ("; AR coefficients switch" if sa else "") + ("; variance switches" if sv else "") + "."]
    return result("ms_ar", f"Markov-switching AR({order}): {y}", res.nobs, summary, tabs, charts, notes + _collapse_notes(res))


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


def _pack(fit, K, kind, n, q):
    th = [np.concatenate(fit["nu"])]
    th += [fit["A"][j].flatten("F") for j in range(K)] if kind == "MSIAH" else [fit["A"][0].flatten("F")]
    for j in range(K if kind != "MSI" else 1):
        L = np.linalg.cholesky(fit["S"][j])
        th.append(np.r_[np.log(np.diag(L)), L[np.tril_indices(n, -1)]])
    P = np.clip(fit["P"], 1e-8, 1)
    th.append(np.log(P[:, :-1] / P[:, [-1]]).ravel())
    return np.concatenate(th)


def _unpack(th, K, kind, n, q):
    i = 0
    nu = [th[i + j * n:i + (j + 1) * n] for j in range(K)]; i += K * n
    nA = K if kind == "MSIAH" else 1
    A = [th[i + j * q * n:i + (j + 1) * q * n].reshape((q, n), order="F") for j in range(nA)]; i += nA * q * n
    if nA == 1:
        A = A * K
    S = []
    m = n + n * (n - 1) // 2
    for j in range(K if kind != "MSI" else 1):
        v = th[i:i + m]; i += m
        L = np.zeros((n, n)); L[np.diag_indices(n)] = np.exp(v[:n]); L[np.tril_indices(n, -1)] = v[n:]
        S.append(L @ L.T)
    if len(S) == 1:
        S = S * K
    lg = th[i:i + K * (K - 1)].reshape(K, K - 1)
    E_ = np.exp(np.column_stack([lg, np.zeros(K)]))
    return nu, A, S, E_ / E_.sum(1, keepdims=True)


def _ms_llf(th, units, K, kind, n, q):
    nu, A, S, P = _unpack(th, K, kind, n, q)
    T = units[0][0].shape[0]
    logf = np.zeros((T, K))
    for (Y, X) in units:
        for j in range(K):
            logf[:, j] += _mvn_logpdf(Y - nu[j] - (X @ A[j] if q else 0), S[j])
    mx = logf.max(1, keepdims=True)
    fd = np.exp(logf - mx)
    w, vec = np.linalg.eig(P.T)
    prev = np.abs(np.real(vec[:, np.argmin(np.abs(w - 1))])); prev /= prev.sum()
    ll = 0.0
    for t in range(T):
        num = (prev @ P) * fd[t]
        c = num.sum()
        ll += np.log(c) + mx[t, 0]
        prev = num / c
    return ll


def _ms_se(fit, units, K, kind, n, q):
    """Standard errors from the numerical Hessian of the log-likelihood at the EM estimates."""
    from .diagnostics import numerical_hessian
    th = _pack(fit, K, kind, n, q)
    if len(th) > 80:
        return None
    try:
        H = numerical_hessian(lambda t: -_ms_llf(t, units, K, kind, n, q), th)
        V = np.linalg.pinv((H + H.T) / 2)
        se = np.sqrt(np.clip(np.diag(V), 0, None))
    except Exception:
        return None
    # delta method for the transition probabilities
    def probs(t):
        return _unpack(t, K, kind, n, q)[3].ravel()
    eps = 1e-5
    J = np.array([(probs(th + eps * np.eye(len(th))[i]) - probs(th - eps * np.eye(len(th))[i])) / (2 * eps) for i in range(len(th))]).T
    se_P = np.sqrt(np.clip(np.diag(J @ V @ J.T), 0, None)).reshape(K, K)
    return dict(se=se, se_P=se_P)


def _ms_output(model_id, title, fit, names, p, K, kind, labels, units, extra_notes, h=20, kmax_fits=None, ref=None):
    n = len(names)
    q = n * p
    nu, A, S, P, sm_ = fit["nu"], fit["A"], fit["S"], fit["P"], fit["smoothed"]
    T = sm_.shape[0]
    from . import diagnostics as D
    se = _ms_se(fit, units, K, kind, n, q)
    if se:
        sev = se["se"]
        rows = []
        for i, v in enumerate(names):
            for j in range(K):
                b, s_ = nu[j][i], sev[j * n + i]
                rows.append([f"{v}, regime {j}", f(b), f(s_), f(b / s_ if s_ > 0 else None, 4), f(2 * stats.norm.sf(abs(b / s_)) if s_ > 0 else None, 4)])
        tabs = [table("Regime-dependent intercepts", ["Parameter", "Estimate", "Std. err.", "z", "p-value"], rows,
                      "Standard errors from the numerical Hessian of the log-likelihood at the EM estimates.")]
    else:
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
                      [[f"Regime {i}"] + [f"{P[i, j]:.4f}" + (f" ({se['se_P'][i, j]:.4f})" if se else "") for j in range(K)] for i in range(K)],
                      "Standard errors in parentheses (delta method)." if se else None))
    dur = 1 / (1 - np.diag(P))
    w, vec = np.linalg.eig(P.T)
    erg = np.real(vec[:, np.argmin(np.abs(w - 1))]); erg = np.abs(erg) / np.abs(erg).sum()
    tabs.append(table("Regimes", ["Regime", "Expected duration", "Ergodic probability", "Share of periods (smoothed prob. > 0.5)"],
                      [[f"Regime {j}", f(dur[j], 3), f(erg[j], 4), f((sm_[:, j] > 0.5).mean(), 3)] for j in range(K)]))
    if q:
        rn = reg_names(names, p, const=False)
        off = K * n
        for j in range(K if kind == "MSIAH" else 1):
            def cell(ri, i):
                b = A[j][ri, i]
                if not se:
                    return f(b)
                s_ = se["se"][off + j * q * n + i * q + ri]
                return f"{b:.4f} ({s_:.4f})"
            tabs.append(table(f"Autoregressive coefficients" + (f", regime {j}" if kind == "MSIAH" else " (common to all regimes)"),
                              ["Regressor"] + [f"eq. {v}" for v in names], [[r] + [cell(ri, i) for i in range(n)] for ri, r in enumerate(rn)],
                              "Standard errors in parentheses." if se else None))
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
    if kmax_fits:
        rows = [[kk, f(ll_, 3), f(-2 * ll_ + 2 * np_, 3), f(-2 * ll_ + np.log(nobs) * np_, 3), f(-2 * ll_ + 2 * np.log(np.log(nobs)) * np_, 3)] for kk, ll_, np_ in kmax_fits]
        tabs.insert(0, table("Number of regimes: information criteria", ["Regimes", "Log-likelihood", "AIC", "BIC", "HQIC"], rows,
                             "Regimes = 1 is the linear VAR. A standard LR test is not valid under the null of no switching; compare the criteria."))
    tabs.insert(0, D.unit_roots(np.vstack([u[0] for u in units]) if len(units) == 1 else np.mean([u[0] for u in units], axis=0), names))
    if len(units) > 1:
        tabs[0]["title"] = "Unit-root tests (cross-unit average of the series)"
    tabs.append(table("Regime classification", ["Measure", "Value"], [["Regime classification measure, RCM (Ang and Bekaert 2002)", f(D.rcm(sm_), 2)]],
                      "RCM = 0: sharp classification; 100: none. Below about 30 is sharp."))
    tabs.append(D.regime_dates(sm_, lab))
    resid = np.vstack([u[0] - sum(sm_[:, [j]] * (nu[j] + (u[1] @ A[j] if q else 0)) for j in range(K)) for u in units])
    tabs.append(D.resid_tests(resid, names, title="Residual diagnostics (observed minus regime-weighted fit)"))
    if ref is not None:
        best = max(range(K), key=lambda j: (((sm_[:, j] > .5) == (ref[-T:] > .5)).mean()))
        tabs.append(table(f"Comparison with the reference series (regime {best})", ["Measure", "Value"], D.concordance(sm_, ref[-T:], best)))
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
             "intercept of the first variable. Standard errors come from the numerical Hessian of the log-likelihood at the EM estimates. Impulse responses are regime-specific "
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
    starts = int_option(spec, "starts", 6, 1, 50, "Starting values")
    fit = _ms_em([(Yt, X)], K, kind, starts=starts)
    kf = _regime_numbers([(Yt, X)], K, kind, fit, starts)
    ref = _ref_spec(spec, df)
    refv = pd.to_numeric(df.loc[d.index, ref], errors="coerce").fillna(0).to_numpy()[p:] if ref else None
    return _ms_output("msvar", f"Markov-switching VAR ({kind}, {K} regimes)", fit, names, p, K, kind, labels, [(Yt, X)], [], kmax_fits=kf, ref=refv)


def _regime_numbers(units, K, kind, fit, starts):
    n = units[0][0].shape[1]
    q = units[0][1].shape[1]
    out = []
    lin, kl = _linear_llf(units)
    out.append((1, lin, kl))
    for kk in sorted({2, 3, K}):
        if kk > 3:
            continue
        f_ = fit if kk == K else _ms_em(units, kk, kind, starts=max(2, starts // 2))
        npar = kk * n + (kk if kind == "MSIAH" else 1) * q * n + (kk if kind != "MSI" else 1) * n * (n + 1) // 2 + kk * (kk - 1)
        out.append((kk, f_["llf"], npar))
    return out


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
    starts = int_option(spec, "starts", 6, 1, 50, "Starting values")
    fit = _ms_em(units, K, kind, starts=starts)
    kf = _regime_numbers(units, K, kind, fit, starts)
    notes = [f"Panel version: one regime path shared by all {len(units)} units (synchronised regimes), pooled regime-dependent coefficients; "
             + ("each unit's series are demeaned first, so the regime intercepts are deviations from unit means." if demean else "no unit fixed effects.")]
    if dropped:
        notes.append(f"{dropped} unit(s) with missing periods were left out (a balanced panel is required).")
    return _ms_output("mspvar", f"Markov-switching panel VAR ({kind}, {K} regimes)", fit, names, p, K, kind, times, units, notes, kmax_fits=kf)
