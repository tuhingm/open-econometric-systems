"""Time-series models of the Time-Series Research Guide: OLS, ARDL, NARDL, VAR, VECM and 2SLS / GMM."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .core import (OESError, as_list, chart, check_columns, coef_table, distinct, f, float_option, int_option, need,
                   numeric, result, stat, table, test_row, tests_table, time_order)


def _series_frame(df, spec, cols):
    tcol = spec.get("time")
    if tcol:
        check_columns(df, [tcol])
        df = time_order(df, tcol)
    d = numeric(df, cols, keep=[tcol] if tcol else [])
    labels = d[tcol].astype(str).tolist() if tcol else [str(i + 1) for i in range(len(d))]
    return d.reset_index(drop=True), labels


def _diagnostics(res, lags=4):
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, het_arch, linear_reset
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    rows = []
    e = np.asarray(res.resid)
    try:
        bg = acorr_breusch_godfrey(res, nlags=lags)
        rows.append(test_row(f"Breusch-Godfrey serial correlation ({lags} lags)", bg[0], bg[1], "H0: no serial correlation"))
    except Exception:
        pass
    try:
        bp = het_breuschpagan(e, res.model.exog)
        rows.append(test_row("Breusch-Pagan heteroskedasticity", bp[0], bp[1], "H0: constant variance"))
    except Exception:
        pass
    try:
        ar = het_arch(e, nlags=lags)
        rows.append(test_row(f"ARCH LM ({lags} lags)", ar[0], ar[1], "H0: no ARCH effects"))
    except Exception:
        pass
    jb = jarque_bera(e)
    rows.append(test_row("Jarque-Bera normality", jb[0], jb[1], "H0: normal errors"))
    try:
        rs = linear_reset(res, power=3, use_f=True)
        rows.append(test_row("Ramsey RESET (powers 2-3)", rs.fvalue, rs.pvalue, "H0: correct functional form"))
    except Exception:
        pass
    rows.append(["Durbin-Watson", f(durbin_watson(e), 4), None, "about 2 means no first-order autocorrelation"])
    return rows


def _cov_kw(cov, hac_lags):
    if cov in ("nonrobust", None, ""):
        return {}
    if cov == "HAC":
        return {"cov_type": "HAC", "cov_kwds": {"maxlags": hac_lags}}
    if cov in ("HC0", "HC1", "HC2", "HC3"):
        return {"cov_type": cov}
    raise OESError(f"Unknown covariance option {cov}.")


# ------------------------------------------------------------------------------------------------ OLS
def ols(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "explanatory variables"))
    distinct([y], x)
    d, labels = _series_frame(df, spec, [y] + x)
    k = len(x) + 1
    if len(d) < k + 3:
        raise OESError("Too few complete observations for this regression.")
    cov = spec.get("cov", "HC1")
    hac = int_option(spec, "hac_lags", 4, 0, 48, "HAC lags")
    X = sm.add_constant(d[x].astype(float), has_constant="add")
    base = sm.OLS(d[y].astype(float), X).fit()
    res = sm.OLS(d[y].astype(float), X).fit(**_cov_kw(cov, hac))
    robust = cov not in ("nonrobust", None, "")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = [[v, f(variance_inflation_factor(X.values, i + 1), 3)] for i, v in enumerate(x)] if len(x) > 1 else []
    cov_label = {"nonrobust": "conventional", "HAC": f"Newey-West HAC ({hac} lags)"}.get(cov, f"heteroskedasticity-robust {cov}")
    tables = [coef_table(["const"] + x, res.params, res.bse, dist="z" if robust else "t", df=res.df_resid,
                         note=f"Standard errors: {cov_label}."),
              tests_table(_diagnostics(base))]
    if vif:
        tables.append(table("Variance inflation factors", ["Variable", "VIF"], vif, "Values above 10 indicate strong multicollinearity."))
    summary = [stat("Observations", int(res.nobs)), stat("R-squared", res.rsquared), stat("Adj. R-squared", res.rsquared_adj),
               stat("F-statistic", base.fvalue), stat("Prob (F)", base.f_pvalue), stat("AIC", res.aic, 3), stat("BIC", res.bic, 3)]
    charts = [chart("Actual and fitted values", labels, [(y, d[y]), ("fitted", base.fittedvalues)]),
              chart("Residuals", labels, [("residual", base.resid)])]
    return result("ols", f"OLS: {y}", res.nobs, summary, tables, charts)


# ------------------------------------------------------------------------------------------------ ARDL family
def _uecm_tables(uecm, y, regs, case):
    """Bounds test, long-run coefficients, error-correction form and diagnostics of a fitted UECM."""
    import statsmodels.api as sm
    bt = uecm.bounds_test(case=case)
    cv = bt.crit_vals
    bounds_rows = []
    for level, row in cv.iterrows():
        sig = 100 - float(level)
        verdict = "long-run relation" if bt.stat > row["upper"] else ("no long-run relation" if bt.stat < row["lower"] else "inconclusive")
        bounds_rows.append([f"{sig:g}%", f(row["lower"], 3), f(row["upper"], 3), verdict])
    bounds = table(f"Bounds test (Pesaran, Shin and Smith 2001), case {case}: F = {f(bt.stat, 3)}",
                   ["Significance", "I(0) bound", "I(1) bound", "Conclusion"], bounds_rows,
                   "H0: no long-run relationship. Reject when F is above the I(1) bound.")
    ci = pd.Series(uecm.ci_params)
    cib = pd.Series(uecm.ci_bse)
    cip = pd.Series(uecm.ci_pvalues)
    lr_names = [n for n in ci.index if n != y]
    long_run = coef_table(lr_names, -ci[lr_names].values, cib[lr_names].values, dist="z", p=cip[lr_names].values,
                          title="Long-run coefficients", note=f"Long run: {y} = sum of coefficient x variable. Delta-method standard errors.")
    params = pd.Series(uecm.params)
    bse = pd.Series(uecm.bse)
    ec_name = f"{y}.L1"
    sr = coef_table(list(params.index), params.values, bse.values, df=uecm.df_resid if hasattr(uecm, "df_resid") else None,
                    title="Error-correction form (unrestricted ECM)",
                    note=f"{ec_name} is the speed of adjustment; it should be negative. Its t-ratio is not standard: use the bounds test.")
    rho = float(params[ec_name])
    ols_ = sm.OLS(uecm.model._y, uecm.model._x).fit()
    half = np.log(0.5) / np.log(1 + rho) if -1 < rho < 0 else np.nan
    return bt, rho, half, [bounds, long_run, sr, tests_table(_diagnostics(ols_))], ols_


def _ardl_core(d, y, regs, spec, model_id, title_prefix, notes):
    from statsmodels.tsa.ardl import ardl_select_order, UECM
    maxlag = int_option(spec, "max_lag", 4, 1, 24, "Maximum lag of the dependent variable")
    maxorder = int_option(spec, "max_order", 4, 0, 24, "Maximum lag of the regressors")
    ic = spec.get("ic", "aic")
    if ic not in ("aic", "bic", "hqic"):
        raise OESError("The information criterion must be aic, bic or hqic.")
    case = int_option(spec, "case", 3, 1, 5, "Bounds-test case")
    trend = {1: "n", 2: "c", 3: "c", 4: "ct", 5: "ct"}[case]
    n = len(d)
    if n < 30:
        raise OESError(f"ARDL needs at least 30 observations; {n} were found.")
    Y, X = d[y].astype(float), d[regs].astype(float)
    sel = ardl_select_order(Y, maxlag=maxlag, exog=X, maxorder=maxorder, ic=ic, trend=trend)
    p = max(1, max(sel.model.ar_lags or [1]))
    order = {}
    dropped = []
    for v in regs:
        lags = sel.model.dl_lags.get(v) if isinstance(sel.model.dl_lags, dict) else None
        q = max(lags) if lags else 0
        if not lags:
            dropped.append(v)
        order[v] = max(1, q)
    if dropped:
        notes.append("The lag search left out " + ", ".join(dropped) + "; it is kept with one lag so that its long-run "
                     "coefficient is reported.")
    uecm = UECM(Y, lags=p, exog=X, order=order, trend=trend).fit()
    bt, rho, half, tabs, ols_ = _uecm_tables(uecm, y, regs, case)
    spec_txt = f"ARDL({p}, " + ", ".join(str(order[v]) for v in regs) + ")"
    sel_rows = [[spec_txt, ic.upper(), f"max lag {maxlag} for {y}, {maxorder} for the regressors"]]
    tabs.insert(0, table("Lag selection", ["Selected model", "Criterion", "Search"], sel_rows))
    summary = [stat("Observations", int(uecm.nobs)), stat("Model", spec_txt), stat("Bounds F", bt.stat, 3),
               stat("Speed of adjustment", rho), stat("Half-life (periods)", half, 2), stat("AIC", uecm.aic, 3)]
    return uecm, ols_, tabs, summary, p, order, case


def ardl(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    distinct([y], x)
    d, labels = _series_frame(df, spec, [y] + x)
    notes = ["The ARDL bounds test is valid for any mixture of I(0) and I(1) regressors, but not for I(2) variables."]
    uecm, ols_, tabs, summary, p, order, case = _ardl_core(d, y, x, spec, "ardl", "ARDL", notes)
    lab = labels[len(labels) - len(ols_.resid):]
    charts = [chart("Residuals of the error-correction regression", lab, [("residual", ols_.resid)])]
    return result("ardl", f"ARDL bounds test: {y}", uecm.nobs, summary, tabs, charts, notes)


def nardl(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "asymmetric regressor(s)"))
    ctrl = as_list(spec.get("controls"))
    distinct([y], x, ctrl)
    d, labels = _series_frame(df, spec, [y] + x + ctrl)
    regs = []
    for v in x:
        dv = d[v].astype(float).diff().fillna(0.0)
        d[v + "_pos"] = dv.clip(lower=0).cumsum()
        d[v + "_neg"] = dv.clip(upper=0).cumsum()
        regs += [v + "_pos", v + "_neg"]
    regs += ctrl
    notes = ["Partial sums (Shin, Yu and Greenwood-Nimmo 2014): x_pos accumulates increases and x_neg decreases of x.",
             "The bounds test counts each partial sum as a regressor."]
    uecm, ols_, tabs, summary, p, order, case = _ardl_core(d, y, regs, spec, "nardl", "NARDL", notes)
    params, V = pd.Series(uecm.params), pd.DataFrame(uecm.cov_params(), index=uecm.params.index, columns=uecm.params.index)
    rows = []
    for v in x:
        pos, neg = f"{v}_pos.L1", f"{v}_neg.L1"
        rows.append(_wald(params, V, {pos: 1, neg: -1}, f"Long-run symmetry of {v}", "H0: same long-run effect of increases and decreases"))
        sp = [n for n in params.index if n.startswith(f"D.{v}_pos")]
        sn = [n for n in params.index if n.startswith(f"D.{v}_neg")]
        if sp and sn:
            w = {**{n: 1 for n in sp}, **{n: -1 for n in sn}}
            rows.append(_wald(params, V, w, f"Short-run symmetry of {v} (sum of lags)", "H0: same short-run effect"))
    tabs.insert(3, tests_table(rows, "Asymmetry (Wald) tests"))
    charts = [chart("Partial sums", labels, [(c, d[c]) for c in regs if c.endswith(("_pos", "_neg"))])]
    return result("nardl", f"NARDL: {y}", uecm.nobs, summary, tabs, charts, notes)


def _wald(params, V, weights, name, detail):
    r = np.array([weights.get(n, 0.0) for n in params.index], float)
    val = float(r @ params.values)
    var = float(r @ V.values @ r)
    W = val ** 2 / var
    return [name, f(W, 4), f(stats.chi2.sf(W, 1), 4), detail + f"; difference = {f(val, 4)}"]


# ------------------------------------------------------------------------------------------------ VAR
def _irf_charts(irf, names, horizon, ortho=True, max_charts=9):
    out = []
    resp = irf.orth_irfs if ortho else irf.irfs
    try:
        se = irf.stderr(orth=ortho)
    except Exception:
        se = None
    x = list(range(horizon + 1))
    for j, imp in enumerate(names):
        for i, r in enumerate(names):
            if len(out) >= max_charts:
                return out
            y = resp[:, i, j]
            band = (y - 1.96 * se[:, i, j], y + 1.96 * se[:, i, j]) if se is not None else None
            out.append(chart(f"Response of {r} to a shock in {imp}", x, [("response", y)], "horizon", r, band=band))
    return out


def var(df, spec):
    from statsmodels.tsa.api import VAR
    names = as_list(need(spec, "vars", "variables (at least two)"))
    if len(names) < 2:
        raise OESError("A VAR needs at least two variables.")
    distinct(names)
    d, labels = _series_frame(df, spec, names)
    data = d[names].astype(float)
    if spec.get("transform", "levels") == "diff":
        data = data.diff().dropna()
        labels = labels[1:]
    maxl = int_option(spec, "max_lags", 8, 1, 24, "Maximum lag")
    ic = spec.get("ic", "aic")
    horizon = int_option(spec, "horizon", 12, 1, 60, "IRF horizon")
    model = VAR(data.reset_index(drop=True))
    maxl = min(maxl, max(1, (len(data) - 10) // (len(names) + 1)))
    so = model.select_order(maxl)
    p = spec.get("lags")
    p = int_option(spec, "lags", 1, 1, 24, "VAR lag order") if p not in (None, "", "auto") else max(1, int(getattr(so, ic)))
    res = model.fit(p)
    sel_rows = []
    for lag in range(len(so.ics["aic"])):
        sel_rows.append([lag] + [f(so.ics[c][lag], 4) for c in ("aic", "bic", "hqic", "fpe")])
    tabs = [table("Lag-order selection", ["Lags", "AIC", "BIC", "HQIC", "FPE"], sel_rows,
                  f"Selected by {ic.upper()}: {p} lag(s).")]
    for eq in names:
        tabs.append(coef_table(list(res.params.index), res.params[eq].values, res.stderr[eq].values, dist="t", df=res.df_resid,
                               title=f"Equation: {eq}"))
    gc = []
    for caused in names:
        for causing in names:
            if caused == causing:
                continue
            t = res.test_causality(caused, [causing], kind="f")
            gc.append([f"{causing} -> {caused}", f(t.test_statistic, 4), f(t.pvalue, 4), f"F({t.df[0]}, {t.df[1]}); H0: {causing} does not Granger-cause {caused}"])
    tabs.append(tests_table(gc, "Granger causality tests"))
    diag = []
    try:
        w = res.test_whiteness(nlags=max(p + 1, min(12, len(data) // 5)))
        diag.append(test_row(f"Portmanteau whiteness ({w.df} df)", w.test_statistic, w.pvalue, "H0: no residual autocorrelation"))
    except Exception:
        pass
    try:
        nm = res.test_normality()
        diag.append(test_row("Jarque-Bera normality (system)", nm.test_statistic, nm.pvalue, "H0: normal residuals"))
    except Exception:
        pass
    roots = np.abs(res.roots)
    diag.append(["Stability: smallest root modulus", f(roots.min(), 4), None, "stable if every root lies outside the unit circle (modulus > 1)"])
    tabs.append(tests_table(diag, "Residual diagnostics and stability"))
    irf = res.irf(horizon)
    fe = res.fevd(horizon).decomp  # (neqs, horizon, neqs)
    fe_rows = [[r] + [f(fe[i, -1, j] * 100, 2) for j in range(len(names))] for i, r in enumerate(names)]
    tabs.append(table(f"Forecast-error variance decomposition at horizon {horizon} (%)", ["Variable"] + names, fe_rows,
                      "Rows: variable forecast; columns: share explained by each shock (Cholesky, in the order the variables were chosen)."))
    summary = [stat("Observations", int(res.nobs)), stat("Lags", p), stat("Variables", len(names)), stat("AIC", res.aic, 4),
               stat("BIC", res.bic, 4), stat("Stable", "yes" if res.is_stable() else "no")]
    notes = ["Orthogonalised impulse responses use the Cholesky ordering of the variable list; bands are ±1.96 asymptotic standard errors."]
    if spec.get("transform", "levels") == "diff":
        notes.append("The variables were first-differenced before estimation.")
    return result("var", "Vector autoregression", res.nobs, summary, tabs, _irf_charts(irf, names, horizon), notes)


# ------------------------------------------------------------------------------------------------ VECM
def vecm(df, spec):
    from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
    names = as_list(need(spec, "vars", "variables (at least two)"))
    if len(names) < 2:
        raise OESError("A VECM needs at least two variables.")
    distinct(names)
    d, labels = _series_frame(df, spec, names)
    data = d[names].astype(float).reset_index(drop=True)
    det = spec.get("deterministic", "ci")
    if det not in ("n", "co", "ci", "lo", "li"):
        raise OESError("Deterministic terms must be one of n, co, ci, lo, li.")
    maxl = int_option(spec, "max_lags", 8, 1, 24, "Maximum lag")
    maxl = min(maxl, max(1, (len(data) - 10) // (len(names) + 1)))
    so = select_order(data, maxlags=maxl, deterministic=det)
    k_ar_diff = spec.get("lags")
    k_ar_diff = int_option(spec, "lags", 1, 0, 24, "Lags in differences") if k_ar_diff not in (None, "", "auto") else int(so.aic)
    det_order = {"n": -1, "co": 0, "ci": 0, "lo": 1, "li": 1}[det]
    rk = select_coint_rank(data, det_order, k_ar_diff, method="trace", signif=0.05)
    jrows = []
    for i in range(len(rk.test_stats)):
        jrows.append([f"r <= {i}", f(rk.test_stats[i], 3), f(rk.crit_vals[i], 3), "reject" if rk.test_stats[i] > rk.crit_vals[i] else "do not reject"])
    rank = spec.get("rank")
    rank = int_option(spec, "rank", 1, 1, len(names) - 1, "Cointegration rank") if rank not in (None, "", "auto") else int(rk.rank)
    notes = []
    if rank == 0:
        notes.append("The trace test finds no cointegration at 5%; the VECM is estimated with rank 1 for illustration. "
                     "With no cointegration, use a VAR in first differences instead.")
        rank = 1
    if rank >= len(names):
        rank = len(names) - 1
    res = VECM(data, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic=det).fit()
    tabs = [table(f"Johansen trace test (lags in differences = {k_ar_diff}, deterministic = '{det}')",
                  ["H0", "Trace statistic", "5% critical value", "Decision"], jrows,
                  f"Selected rank: {rk.rank}.")]
    for r in range(rank):
        tabs.append(coef_table(names, res.beta[:, r], res.stderr_beta[:, r] if res.stderr_beta is not None else np.full(len(names), np.nan),
                               dist="z", title=f"Cointegrating vector {r + 1} (beta, normalised on {names[r]})",
                               note="Normalised coefficients have zero standard error by construction."))
    tabs.append(table("Adjustment coefficients (alpha)", ["Equation"] + [f"alpha{r + 1}" for r in range(rank)] + [f"p{r + 1}" for r in range(rank)],
                      [[n] + [f(res.alpha[i, r]) for r in range(rank)] + [f(res.pvalues_alpha[i, r], 4) for r in range(rank)] for i, n in enumerate(names)],
                      "A negative, significant alpha means the variable moves back towards the long-run relation."))
    if k_ar_diff > 0:
        gnames = [f"L{l + 1}.D.{v}" for l in range(k_ar_diff) for v in names]
        for i, n in enumerate(names):
            tabs.append(coef_table(gnames, res.gamma[i], res.stderr_gamma[i], dist="z", title=f"Short-run coefficients: D.{n}"))
    horizon = int_option(spec, "horizon", 12, 1, 60, "IRF horizon")
    irf = res.irf(horizon)
    charts = []
    for j, imp in enumerate(names):
        for i, rsp in enumerate(names):
            if len(charts) < 9:
                charts.append(chart(f"Response of {rsp} to a shock in {imp}", list(range(horizon + 1)), [("response", irf.orth_irfs[:, i, j])], "horizon", rsp))
    summary = [stat("Observations", int(res.nobs)), stat("Lags in differences", k_ar_diff), stat("Cointegration rank", rank),
               stat("Deterministic", det), stat("Log-likelihood", res.llf, 3)]
    notes.append("Orthogonalised impulse responses (Cholesky, variable order as chosen), without confidence bands.")
    return result("vecm", "Vector error-correction model", res.nobs, summary, tabs, charts, notes)


# ------------------------------------------------------------------------------------------------ 2SLS / GMM
def _hac(U, L):
    """Bartlett-kernel long-run covariance of the rows of U (already products of instruments and residuals)."""
    S = U.T @ U
    for j in range(1, L + 1):
        w = 1 - j / (L + 1)
        G = U[j:].T @ U[:-j]
        S += w * (G + G.T)
    return S


def iv_fit(y, X, Z, method="2sls", cov="robust", L=0):
    """2SLS or two-step efficient GMM. Covariances follow linearmodels: robust = HC0, kernel = Bartlett(L) (for GMM the
    weight matrix and the covariance both use the chosen option, as IVGMM(weight_type=..., cov_type=...)), unadjusted."""
    n = len(y)
    ZZi = np.linalg.pinv(Z.T @ Z)
    Xh = Z @ ZZi @ (Z.T @ X)
    b = np.linalg.solve(Xh.T @ X, Xh.T @ y)
    e = y - X @ b
    if method == "2sls":
        A = np.linalg.inv(Xh.T @ Xh)
        if cov == "unadjusted":
            V = A * (e @ e) / n
        else:
            U = Xh * e[:, None]
            S = _hac(U, L) if cov == "kernel" else U.T @ U
            V = A @ S @ A
        J = None
    else:
        U = Z * e[:, None]
        S = (_hac(U, L) if cov == "kernel" else U.T @ U) / n
        W = np.linalg.pinv(S)
        XZ = X.T @ Z
        b = np.linalg.solve(XZ @ W @ XZ.T, XZ @ W @ (Z.T @ y))
        e = y - X @ b
        U = Z * e[:, None]
        S = (_hac(U, L) if cov == "kernel" else U.T @ U) / n
        A = np.linalg.inv(XZ @ W @ XZ.T)
        V = A @ (XZ @ W @ S @ W @ XZ.T) @ A * n
        gbar = Z.T @ e / n
        J = float(n * gbar @ W @ gbar)          # Hansen J with the weight matrix used in estimation (as linearmodels)
    return b, V, e, J


def iv_gmm(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "dependent variable")
    endog = as_list(need(spec, "endog", "endogenous regressor(s)"))
    inst = as_list(need(spec, "instruments", "excluded instrument(s)"))
    exog = as_list(spec.get("exog"))
    distinct([y], endog, inst, exog)
    if len(inst) < len(endog):
        raise OESError("The model is not identified: it needs at least as many excluded instruments as endogenous regressors.")
    d, labels = _series_frame(df, spec, [y] + endog + inst + exog)
    method = spec.get("method", "2sls")
    cov = spec.get("cov", "robust")
    L = int_option(spec, "bandwidth", 4, 0, 48, "HAC bandwidth")
    n = len(d)
    one = np.ones((n, 1))
    W_ = np.column_stack([one, d[exog].to_numpy(float)]) if exog else one
    X = np.column_stack([W_, d[endog].to_numpy(float)])
    Z = np.column_stack([W_, d[inst].to_numpy(float)])
    Y = d[y].to_numpy(float)
    names = ["const"] + exog + endog
    b, V, e, J = iv_fit(Y, X, Z, method, cov, L)
    se = np.sqrt(np.diag(V))
    cov_label = {"robust": "heteroskedasticity-robust", "kernel": f"Bartlett HAC (bandwidth {L})", "unadjusted": "conventional"}[cov]
    tabs = [coef_table(names, b, se, dist="z", title=f"{'2SLS' if method == '2sls' else 'Two-step efficient GMM'} estimates",
                       note=f"Standard errors: {cov_label}.")]
    ob = sm.OLS(Y, X).fit()
    tabs.append(coef_table(names, ob.params, ob.bse, dist="t", df=ob.df_resid, title="OLS for comparison (conventional standard errors)"))
    rows = []
    fs_rows = []
    for j, en in enumerate(endog):
        full = sm.OLS(d[en].to_numpy(float), Z).fit()
        rest = sm.OLS(d[en].to_numpy(float), W_).fit()
        q = len(inst)
        F = ((rest.ssr - full.ssr) / q) / (full.ssr / full.df_resid)
        R = np.zeros((q, Z.shape[1])); R[:, W_.shape[1]:] = np.eye(q)
        rob = sm.OLS(d[en].to_numpy(float), Z).fit(cov_type="HC0")
        Fr = float(rob.wald_test(R, scalar=True).statistic) / q
        fs_rows.append([en, f(F, 3), f(stats.f.sf(F, q, full.df_resid), 4), f(Fr, 3), f(full.rsquared - rest.rsquared, 4)])
    tabs.append(table("First stage: strength of the excluded instruments", ["Endogenous", "F (conventional)", "p-value", "F (robust)", "Partial R-squared"],
                      fs_rows, "Rule of thumb: F below 10 signals weak instruments (Staiger and Stock 1997)."))
    # Durbin-Wu-Hausman endogeneity test by control function: first-stage residuals added to the structural equation
    V1 = np.column_stack([d[en].to_numpy(float) - sm.OLS(d[en].to_numpy(float), Z).fit().fittedvalues for en in endog])
    cf = sm.OLS(Y, np.column_stack([X, V1])).fit(cov_type="HC0")
    Rr = np.zeros((len(endog), X.shape[1] + len(endog))); Rr[:, X.shape[1]:] = np.eye(len(endog))
    wt = cf.wald_test(Rr, scalar=True)
    rows.append(test_row("Endogeneity (Durbin-Wu-Hausman, robust)", float(wt.statistic), float(wt.pvalue), "H0: regressors are exogenous; OLS is consistent"))
    over = len(inst) - len(endog)
    if over > 0:
        if J is None:
            _, _, _, J = iv_fit(Y, X, Z, "gmm", cov, L)
        rows.append(test_row(f"Hansen J overidentification ({over} df)", J, stats.chi2.sf(J, over), "H0: instruments are valid"))
    else:
        rows.append(["Hansen J overidentification", None, None, "exactly identified: not testable"])
    tabs.append(tests_table(rows))
    ssr = float(e @ e)
    summary = [stat("Observations", n), stat("Estimator", "2SLS" if method == "2sls" else "GMM (two-step)"),
               stat("Instruments", len(inst)), stat("Endogenous", len(endog)), stat("Min first-stage F", min(float(r[1]) for r in fs_rows), 3),
               stat("R-squared", 1 - ssr / float(((Y - Y.mean()) ** 2).sum()))]
    return result("iv_gmm", f"Instrumental variables: {y}", n, summary, tabs,
                  [chart("Residuals", labels, [("residual", e)])])
