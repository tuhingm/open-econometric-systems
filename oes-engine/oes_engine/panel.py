"""Panel models of the Panel-Data Research Guide.

Static panels (pooled OLS, fixed effects, random effects, first differences, fixed effects with Driscoll-Kraay errors)
are written in NumPy and reproduce linearmodels (the package used in the guide) with its default small-sample scaling.
Difference GMM, mean group, CCE mean group, pooled mean group and CS-ARDL use the guide's own helper files.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import _diff_gmm, _panel_core, _panel_tests
from .core import (OESError, as_list, chart, coef_table, distinct, f, int_option, need, numeric, result, stat, table,
                   test_row, tests_table)


# ------------------------------------------------------------------------------------------------ data
def _panel(df, spec, cols, extra=()):
    """Numeric panel sorted by unit and time; rows missing any of cols are dropped, columns in extra are kept as they are."""
    unit = need(spec, "unit", "unit (entity) identifier")
    time = need(spec, "time", "time identifier")
    distinct([unit, time], cols, list(extra))
    d = numeric(df, cols, keep=[unit, time] + list(extra))
    for c in extra:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_t"] = pd.to_numeric(d[time], errors="coerce")
    if d["_t"].isna().any():
        d["_t"] = pd.to_datetime(d[time], errors="coerce").rank(method="dense")
    if d["_t"].isna().any():
        raise OESError("The time identifier must be numeric (a year or period number) or a date.")
    d = d.sort_values([unit, "_t"]).reset_index(drop=True)
    if d.duplicated([unit, "_t"]).any():
        raise OESError("Some unit appears more than once in the same period. Each unit-period must be one row.")
    d["_u"] = pd.factorize(d[unit])[0]
    d["_ti"] = pd.factorize(d["_t"], sort=True)[0]
    N, T = d["_u"].nunique(), d["_ti"].nunique()
    if N < 2:
        raise OESError("A panel needs at least two units.")
    return d, unit, time, N, T


def _demean(A, codes):
    A = np.asarray(A, float)
    m = pd.DataFrame(A).groupby(codes).transform("mean").to_numpy()
    return A - m


def _two_way(A, u, t, tol=1e-12, maxit=1000):
    """Remove unit and time means (alternating projections; exact in one pass for balanced panels)."""
    A = np.asarray(A, float).copy()
    for _ in range(maxit):
        old = A.copy()
        A = _demean(_demean(A, u), t)
        if np.max(np.abs(A - old)) < tol:
            break
    return A


def _sandwich(X, e, groups=None, time=None, kind="unadjusted", bw=None, extra_df=0):
    """linearmodels-compatible covariance with debiased=True: scale n / (n - k - extra_df)."""
    n, k = X.shape
    XXi = np.linalg.pinv(X.T @ X)
    scale = n / (n - k - extra_df)
    if kind == "unadjusted":
        return XXi * float(e @ e) / n * scale
    U = X * e[:, None]
    if kind == "robust":
        S = U.T @ U
    elif kind == "clustered":
        G = pd.DataFrame(U).groupby(groups).sum().to_numpy()
        S = G.T @ G
    elif kind == "dk":
        H = pd.DataFrame(U).groupby(time).sum().sort_index().to_numpy()
        Tn = H.shape[0]
        L = int(np.floor(4 * (Tn / 100) ** (2 / 9))) if bw is None else int(bw)
        S = H.T @ H
        for j in range(1, L + 1):
            w = 1 - j / (L + 1)
            Gj = H[j:].T @ H[:-j]
            S += w * (Gj + Gj.T)
    else:
        raise OESError(f"Unknown covariance option {kind}.")
    return XXi @ S @ XXi * scale


COV_LABEL = {"unadjusted": "conventional", "robust": "heteroskedasticity-robust", "clustered": "clustered by unit",
             "dk": "Driscoll-Kraay (robust to cross-sectional and serial correlation)"}


def _cov_args(spec, default="clustered"):
    kind = spec.get("cov", default) or default
    if kind not in COV_LABEL:
        raise OESError("Standard errors must be unadjusted, robust, clustered or dk.")
    bw = spec.get("bandwidth")
    bw = None if bw in (None, "", "auto") else int_option(spec, "bandwidth", 3, 0, 50, "Driscoll-Kraay bandwidth")
    return kind, bw


def _cd_residuals(d, e):
    W = pd.DataFrame({"u": d["_u"].values, "t": d["_ti"].values, "e": e}).pivot(index="t", columns="u", values="e")
    cd, p, r = _panel_tests._cd(W.to_numpy(float))
    return test_row("Pesaran CD on residuals", cd, p, f"H0: no cross-sectional dependence; mean |rho| = {f(r, 3)}") if cd is not None else None


# ------------------------------------------------------------------------------------------------ static models
def _within(d, y, x, time_effects):
    u, t = d["_u"].to_numpy(), d["_ti"].to_numpy()
    A = d[[y] + x].to_numpy(float)
    Ad = _two_way(A, u, t) if time_effects else _demean(A, u)
    return Ad[:, 0], Ad[:, 1:]


def _fe_fit(d, y, x, N, T, time_effects, kind, bw):
    yd, Xd = _within(d, y, x, time_effects)
    keep = np.abs(Xd).max(0) > 1e-10
    if not keep.all():
        raise OESError("These regressors do not vary within units and are absorbed by the fixed effects: "
                       + ", ".join(v for v, k in zip(x, keep) if not k) + ". Use random effects or pooled OLS for them.")
    b = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    e = yd - Xd @ b
    extra = (N - 1) + ((T - 1) if time_effects else 0) + 1        # absorbed effects (the constant counted with them)
    # linearmodels does not count one-way effects that are nested in the clusters (only the constant remains)
    extra_cov = 1 if (kind == "clustered" and not time_effects) else extra
    V = _sandwich(Xd, e, d["_u"].to_numpy(), d["_ti"].to_numpy(), kind, bw, extra_df=extra_cov)
    return b, V, e, yd, extra


def pols(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    d, unit, time, N, T = _panel(df, spec, [y] + x)
    kind, bw = _cov_args(spec)
    X = np.column_stack([np.ones(len(d)), d[x].to_numpy(float)])
    Y = d[y].to_numpy(float)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ b
    V = _sandwich(X, e, d["_u"].to_numpy(), d["_ti"].to_numpy(), kind, bw)
    n, k = X.shape
    names = ["const"] + x
    # Breusch-Pagan LM test for unit effects and the F test of fixed effects against pooling
    eb = pd.Series(e).groupby(d["_u"].values)
    Ti = eb.size().to_numpy()
    LM = n ** 2 / (2 * float((Ti * (Ti - 1)).sum())) * (float((eb.sum() ** 2).sum()) / float(e @ e) - 1) ** 2
    bf, Vf, ef, _, extra = _fe_fit(d, y, x, N, T, False, "unadjusted", None)
    ssr_p, ssr_f = float(e @ e), float(ef @ ef)
    df2 = n - N - len(x)
    F = ((ssr_p - ssr_f) / (N - 1)) / (ssr_f / df2)
    rows = [test_row("Breusch-Pagan LM for unit effects", LM, stats.chi2.sf(LM, 1), "H0: no unit effects (pooled OLS is fine)"),
            test_row(f"F test of unit fixed effects F({N - 1}, {df2})", F, stats.f.sf(F, N - 1, df2), "H0: all unit intercepts equal")]
    cd = _cd_residuals(d, e)
    if cd:
        rows.append(cd)
    r2 = 1 - ssr_p / float(((Y - Y.mean()) ** 2).sum())
    tabs = [coef_table(names, b, np.sqrt(np.diag(V)), dist="t", df=n - k, note=f"Standard errors: {COV_LABEL[kind]}."), tests_table(rows)]
    summary = [stat("Observations", n), stat("Units", N), stat("Periods", T), stat("R-squared", r2)]
    return result("pols", f"Pooled OLS: {y}", n, summary, tabs)


def fe(df, spec, model_id="fe", title="Fixed effects"):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    d, unit, time, N, T = _panel(df, spec, [y] + x)
    te = str(spec.get("time_effects", "no")).lower() in ("yes", "true", "1")
    kind, bw = _cov_args(spec, "dk" if model_id == "fedk" else "clustered")
    b, V, e, yd, extra = _fe_fit(d, y, x, N, T, te, kind, bw)
    n = len(d)
    r2w = 1 - float(e @ e) / float(yd @ yd)
    # F test that the unit effects are jointly zero
    Y = d[y].to_numpy(float)
    X = np.column_stack([np.ones(n), d[x].to_numpy(float)])
    if te:
        X = np.column_stack([X, pd.get_dummies(d["_ti"], drop_first=True).to_numpy(float)])
    ep = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    ssr_f = float(e @ e)
    df2 = n - extra - len(x)
    F = ((float(ep @ ep) - ssr_f) / (N - 1)) / (ssr_f / df2)
    rows = [test_row(f"F test of unit effects F({N - 1}, {df2})", F, stats.f.sf(F, N - 1, df2), "H0: no unit effects (pooled OLS)")]
    if model_id == "fe" and not te:
        h = _hausman_fe_re(d, y, x, N)
        if h:
            rows.append(h)
    cd = _cd_residuals(d, e)
    if cd:
        rows.append(cd)
    L = None
    if kind == "dk":
        L = int(np.floor(4 * (T / 100) ** (2 / 9))) if bw is None else bw
    note = f"Standard errors: {COV_LABEL[kind]}" + (f", bandwidth {L}" if L is not None else "") + "."
    eff = "unit and time" if te else "unit"
    tabs = [coef_table(x, b, np.sqrt(np.diag(V)), dist="t", df=df2, note=note + f" {eff.capitalize()} effects absorbed."), tests_table(rows)]
    summary = [stat("Observations", n), stat("Units", N), stat("Periods", T), stat("Within R-squared", r2w),
               stat("Effects", eff)]
    return result(model_id, f"{title}: {y}", n, summary, tabs)


def fedk(df, spec):
    spec = dict(spec)
    spec.setdefault("cov", "dk")
    return fe(df, spec, "fedk", "Fixed effects with Driscoll-Kraay errors")


def _re_fit(d, y, x, N):
    """Swamy-Arora random effects as in linearmodels.RandomEffects."""
    Y = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d[x].to_numpy(float)])
    u = d["_u"].to_numpy()
    n, k = X.shape
    yd, Xd = _demean(Y[:, None], u)[:, 0] + Y.mean(), _demean(X, u) + X.mean(0)
    bw_ = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    ew = yd - Xd @ bw_
    sigma2_e = float(ew @ ew) / (n - k - N + 1)
    gb = pd.DataFrame(np.column_stack([Y, X])).groupby(u).mean().to_numpy()
    bb = np.linalg.lstsq(gb[:, 1:], gb[:, 0], rcond=None)[0]
    eb = gb[:, 0] - gb[:, 1:] @ bb
    Ti = np.bincount(u)
    tbar = N / (1.0 / Ti).sum()
    sigma2_u = max(0.0, float(eb @ eb) / (N - k) - sigma2_e / tbar)
    theta = 1 - np.sqrt(sigma2_e / (Ti * sigma2_u + sigma2_e))
    th = theta[u]
    ys = Y - th * gb[u, 0]
    Xs = X - th[:, None] * gb[u, 1:]
    b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    es = ys - Xs @ b
    return b, Xs, es, sigma2_u, sigma2_e, theta


def _hausman_fe_re(d, y, x, N):
    try:
        bf, Vf, _, _, _ = _fe_fit(d, y, x, N, d["_ti"].nunique(), False, "unadjusted", None)
        br, Xs, es, *_ = _re_fit(d, y, x, N)
        Vr = _sandwich(Xs, es, kind="unadjusted")[1:, 1:]
        dlt = bf - br[1:]
        H = float(dlt @ np.linalg.pinv(Vf - Vr) @ dlt)
        if H < 0:
            return [f"Hausman FE vs RE ({len(x)} df)", f(H, 4), None,
                    "variance difference not positive definite in this sample: rely on the Mundlak test"]
        return test_row(f"Hausman FE vs RE ({len(x)} df)", H, stats.chi2.sf(H, len(x)), "H0: RE consistent (effects uncorrelated with regressors)")
    except Exception:
        return None


def re(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    d, unit, time, N, T = _panel(df, spec, [y] + x)
    kind, bw = _cov_args(spec)
    b, Xs, es, s2u, s2e, theta = _re_fit(d, y, x, N)
    V = _sandwich(Xs, es, d["_u"].to_numpy(), d["_ti"].to_numpy(), kind, bw)
    n = len(d)
    rows = []
    h = _hausman_fe_re(d, y, x, N)
    if h:
        rows.append(h)
    # Mundlak test: unit means of the time-varying regressors added to a pooled regression, clustered Wald test
    tv = [v for v in x if d.groupby("_u")[v].std().fillna(0).max() > 1e-10]
    if tv:
        M = d[tv].groupby(d["_u"]).transform("mean").to_numpy(float)
        Xm = np.column_stack([np.ones(n), d[x].to_numpy(float), M])
        Ym = d[y].to_numpy(float)
        bm = np.linalg.lstsq(Xm, Ym, rcond=None)[0]
        Vm = _sandwich(Xm, Ym - Xm @ bm, d["_u"].to_numpy(), kind="clustered")
        idx = slice(1 + len(x), None)
        W = float(bm[idx] @ np.linalg.pinv(Vm[idx, idx]) @ bm[idx])
        rows.append(test_row(f"Mundlak test ({len(tv)} df, clustered)", W, stats.chi2.sf(W, len(tv)), "H0: unit means irrelevant; RE is consistent"))
    cd = _cd_residuals(d, es)
    if cd:
        rows.append(cd)
    tabs = [coef_table(["const"] + x, b, np.sqrt(np.diag(V)), dist="t", df=n - len(x) - 1, note=f"Standard errors: {COV_LABEL[kind]}."),
            tests_table(rows)]
    summary = [stat("Observations", n), stat("Units", N), stat("Periods", T), stat("sigma2 unit effect", s2u),
               stat("sigma2 idiosyncratic", s2e), stat("rho", s2u / (s2u + s2e)), stat("theta (median)", float(np.median(theta)))]
    return result("re", f"Random effects (GLS): {y}", n, summary, tabs)


def fd(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    d, unit, time, N, T = _panel(df, spec, [y] + x)
    kind, bw = _cov_args(spec, "dk")
    g = d.groupby("_u")
    D = d[[y] + x].astype(float) - g[[y] + x].shift(1).astype(float)
    consecutive = (d["_ti"] - g["_ti"].shift(1)) == 1
    D = D[consecutive]
    dd = d.loc[D.index]
    Y = D[y].to_numpy(float)
    X = np.column_stack([np.ones(len(D)), D[x].to_numpy(float)])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ b
    V = _sandwich(X, e, dd["_u"].to_numpy(), dd["_ti"].to_numpy(), kind, bw)
    names = ["const"] + ["D." + v for v in x]
    rows = []
    sub = dd.assign(_e=e)
    W = sub.pivot(index="_ti", columns="_u", values="_e")
    cd, p, r = _panel_tests._cd(W.to_numpy(float))
    if cd is not None:
        rows.append(test_row("Pesaran CD on residuals", cd, p, "H0: no cross-sectional dependence"))
    L = int(np.floor(4 * (T / 100) ** (2 / 9))) if (kind == "dk" and bw is None) else bw
    tabs = [coef_table(names, b, np.sqrt(np.diag(V)), dist="t", df=len(Y) - X.shape[1],
                       note=f"Standard errors: {COV_LABEL[kind]}" + (f", bandwidth {L}" if kind == "dk" else "") + ". The constant is the average trend."),
            tests_table(rows)]
    r2 = 1 - float(e @ e) / float(((Y - Y.mean()) ** 2).sum())
    return result("fd", f"First differences: D.{y}", len(Y), [stat("Observations", len(Y)), stat("Units", N), stat("R-squared", r2)], tabs)


# ------------------------------------------------------------------------------------------------ dynamic and heterogeneous panels
def _df_table(tb, title, note=None):
    return coef_table(list(tb.index), tb["coef"].values, tb["std err"].values, dist="z", title=title, note=note)


def diff_gmm(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(spec.get("x"))
    # only y must be present: rows with a missing regressor (such as the first year of a lag) still supply
    # lagged levels of y as instruments, and diff_gmm itself skips them as equations
    d, unit, time, N, T = _panel(df, spec, [y], extra=x)
    lags = int_option(spec, "lags_y", 1, 1, 4, "Lags of the dependent variable")
    first = int_option(spec, "gmm_first", 2, 2, 10, "First instrument lag")
    last = spec.get("gmm_last")
    last = None if last in (None, "", "all") else int_option(spec, "gmm_last", 4, first, 30, "Last instrument lag")
    collapse = str(spec.get("collapse", "no")).lower() in ("yes", "true", "1")
    dd = d.assign(_tt=d["_ti"].astype(int))
    r = _diff_gmm.diff_gmm(dd, unit="_u", time="_tt", y=y, lags_y=lags, x=x, gmm_lags=(first, last), collapse=collapse)
    tb = r["table"]
    keep = [i for i in tb.index if not str(i).startswith("year_")]
    h, dfh, ph = r["hansen"]
    rows = [test_row("Arellano-Bond AR(1)", r["ar1"][0], r["ar1"][1], "expected to reject"),
            test_row("Arellano-Bond AR(2)", r["ar2"][0], r["ar2"][1], "H0: no second-order serial correlation (needed)"),
            test_row(f"Hansen J ({dfh} df)", h, ph, "H0: instruments valid; p near 1 is a sign of too many instruments")]
    notes = ["One-step difference GMM (Arellano and Bond 1991) with year dummies; standard errors robust to heteroskedasticity "
             "and within-unit correlation; the Hansen test comes from the two-step estimator."]
    if r["instruments"] > r["units"]:
        notes.append(f"There are more instruments ({r['instruments']}) than units ({r['units']}): consider collapsing the instruments or limiting the lags.")
    tabs = [_df_table(tb.loc[keep], "Difference GMM estimates (one-step, robust)", "Year dummies included but not shown."), tests_table(rows)]
    summary = [stat("Observations", r["nobs"]), stat("Units", r["units"]), stat("Instruments", r["instruments"])]
    return result("diff_gmm", f"Difference GMM: {y}", r["nobs"], summary, tabs, notes=notes)


def _long_panel_tests(d, unit, y, x):
    rows = []
    for v in [y] + x:
        W = d.pivot(index="_ti", columns="_u", values=v)
        cd, p, r = _panel_tests._cd(W.to_numpy(float))
        if cd is not None:
            rows.append(test_row(f"Pesaran CD: {v}", cd, p, f"mean |rho| = {f(r, 3)}"))
    try:
        s = _panel_tests.slope_test(d.assign(_e=d["_u"]), "_e", y, x)
        if isinstance(s, dict) and s.get("delta_adj") is not None:
            rows.append(test_row("Pesaran-Yamagata slope homogeneity (adjusted delta)", s["delta_adj"], s.get("delta_adj_p"), "H0: homogeneous slopes"))
    except Exception:
        pass
    return rows


def mg(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    d, unit, time, N, T = _panel(df, spec, [y] + x)
    method = spec.get("method", "mg")
    dd = d.assign(_tt=d["_ti"])
    if method == "cce":
        r = _panel_core.cce_mg(dd, "_u", "_tt", y, x)
        title = "CCE mean group (Pesaran 2006)"
    else:
        r = _panel_core.mean_group(dd, "_u", "_tt", y, x, trend=str(spec.get("trend", "no")).lower() in ("yes", "true", "1"))
        title = "Mean group (Pesaran and Smith 1995)"
    if r["n"] < 2:
        raise OESError("Too few units have enough periods for unit-by-unit regressions.")
    U = r["units"]
    dist = [[v, f(U[v].mean()), f(U[v].median()), f(U[v].std()), f(U[v].min()), f(U[v].max()), f((U[v] > 0).mean(), 3)] for v in x]
    tabs = [_df_table(r["table"], title, "Nonparametric mean-group standard errors: s.d. of unit slopes / sqrt(N)."),
            table("Distribution of unit slopes", ["Variable", "Mean", "Median", "S.d.", "Min", "Max", "Share > 0"], dist),
            tests_table(_long_panel_tests(d, unit, y, x), "Cross-sectional dependence and slope heterogeneity")]
    names = d.drop_duplicates("_u").set_index("_u")[unit].astype(str)
    s = U[x[0]].sort_values()
    charts = [chart(f"Unit slopes of {x[0]} (sorted)", [names.get(i, str(i)) for i in s.index], [(x[0], s.values)], "unit", "slope", kind="bar")]
    return result("mg", f"{title}: {y}", len(d), [stat("Units used", r["n"]), stat("Periods", T), stat("Observations", len(d))], tabs, charts)


def pmg(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    d, unit, time, N, T = _panel(df, spec, [y] + x)
    dd = d.assign(_tt=d["_ti"])
    res = {m: _panel_core.ardl_ec(dd, "_u", "_tt", y, x, method=m) for m in ("pmg", "mg", "dfe")}
    lab = {"pmg": "Pooled mean group", "mg": "Mean group", "dfe": "Dynamic fixed effects"}
    tabs = []
    main = spec.get("method", "pmg")
    order = [main] + [m for m in ("pmg", "mg", "dfe") if m != main]
    for m in order:
        r = res[m]
        tabs.append(_df_table(r["long_run"], f"{lab[m]}: long-run coefficients"))
        tabs.append(_df_table(pd.concat([r["phi"], r["short_run"]]), f"{lab[m]}: error correction and short run",
                              "ec is the speed of adjustment; it should be negative."))
    H, k, p = _panel_core.hausman(res["pmg"]["theta"], res["pmg"]["V"], res["mg"]["theta"], res["mg"]["V"])
    H2, k2, p2 = _panel_core.hausman(res["dfe"]["theta"], res["dfe"]["V"], res["mg"]["theta"], res["mg"]["V"])
    tabs.insert(2, tests_table([test_row(f"Hausman PMG vs MG ({k} df)", H, p, "H0: long-run homogeneity; PMG is efficient"),
                                test_row(f"Hausman DFE vs MG ({k2} df)", H2, p2, "H0: DFE consistent")], "Choosing between PMG, MG and DFE"))
    phi = res["pmg"]["phi"]["coef"].iloc[0]
    notes = ["Panel ARDL(1,1) in error-correction form. PMG pools the long-run coefficients and lets the short run and the speed "
             "of adjustment differ by unit (Pesaran, Shin and Smith 1999)."]
    summary = [stat("Units", res["pmg"]["n"]), stat("Periods", T), stat("PMG speed of adjustment", float(phi)),
               stat("PMG iterations", res["pmg"].get("iterations", 0))]
    return result("pmg", f"Panel ARDL (PMG / MG / DFE): {y}", len(d), summary, tabs, notes=notes)


def csardl(df, spec):
    y = need(spec, "y", "dependent variable")
    x = as_list(need(spec, "x", "regressors"))
    d, unit, time, N, T = _panel(df, spec, [y] + x)
    pt = spec.get("pt")
    pt = None if pt in (None, "", "auto") else int_option(spec, "pt", 1, 0, 6, "Lags of the cross-sectional averages")
    dd = d.assign(_tt=d["_ti"])
    r = _panel_core.cs_ardl(dd, "_u", "_tt", y, x, pt=pt)
    pm = _panel_core.ardl_ec(dd, "_u", "_tt", y, x, method="pmg")
    tabs = [_df_table(r["long_run"], f"CS-ARDL(1,1) long-run coefficients (mean group), {r['pt']} lag(s) of the averages"),
            _df_table(r["phi"], "Speed of adjustment (mean group)"),
            _df_table(pm["long_run"], "PMG long-run coefficients without cross-sectional averages (for comparison)"),
            tests_table(_long_panel_tests(d, unit, y, x), "Cross-sectional dependence and slope heterogeneity")]
    notes = ["Chudik, Mohaddes, Pesaran and Raissi (2016): unit-by-unit ARDL(1,1) augmented with current and lagged cross-sectional "
             "averages of all variables, which absorb common shocks; mean-group standard errors."]
    return result("csardl", f"CS-ARDL: {y}", len(d), [stat("Units used", r["n"]), stat("Periods", T), stat("Average lags", r["pt"])], tabs, notes=notes)
