"""Local projections (Jordà 2005): a time-series LP (optionally with an instrument, LP-IV) and a panel LP.

For every horizon h = 0, 1, ..., H the dependent variable at t + h is regressed on the shock at t and controls dated
t or earlier. The coefficient on the shock is the impulse response at horizon h, so the response function is the
sequence of H + 1 separate regressions. Standard errors are Newey-West (time series) or clustered / Driscoll-Kraay
(panel), because the horizon-h errors overlap and are serially correlated by construction.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import panel as _panel
from .core import (OESError, as_list, chart, check_columns, distinct, f, int_option, need, result, stat, table,
                   test_row, tests_table)
from .timeseries import _series_frame, iv_fit

OUTCOMES = {"level": "y(t+h)", "diff": "y(t+h) - y(t-1)", "cumsum": "y(t) + ... + y(t+h)"}


def _outcome(y, h, kind):
    """Dependent variable of the horizon-h regression; y is a Series (a unit's series in the panel case)."""
    if kind == "level":
        return y.shift(-h)
    if kind == "diff":
        return y.shift(-h) - y.shift(1)
    if kind == "cumsum":
        return sum(y.shift(-j) for j in range(h + 1))
    raise OESError("The outcome must be level, diff or cumsum.")


def _check_outcome(spec):
    kind = spec.get("outcome", "level") or "level"
    if kind not in OUTCOMES:
        raise OESError("The outcome must be level, diff or cumsum.")
    return kind


def _irf_rows(hs, b, se, nobs):
    z = np.asarray(b) / np.asarray(se)
    return [[int(h), f(bi), f(si), f(2 * stats.norm.sf(abs(zi)), 4), f(bi - 1.96 * si), f(bi + 1.96 * si), int(n)]
            for h, bi, si, zi, n in zip(hs, b, se, z, nobs)]


IRF_COLS = ["Horizon", "Response", "Std. err.", "p-value", "95% CI low", "95% CI high", "Obs."]


def _summary_and_charts(model_title, y, shock, hs, b, se, sd, kind, scale):
    b, se = np.asarray(b, float), np.asarray(se, float)
    k = int(np.nanargmax(np.abs(b)))
    unit = "one standard deviation" if scale == "sd" else "one unit"
    ch = [chart(f"Response of {y} to {unit} of {shock}", hs, [("response", b)], "horizon", y, band=(b - 1.96 * se, b + 1.96 * se))]
    return k, ch


# ------------------------------------------------------------------------------------------------ time series
def lp(df, spec):
    import statsmodels.api as sm
    y = need(spec, "y", "response variable")
    shock = need(spec, "shock", "shock variable")
    contemp = as_list(spec.get("contemp"))
    controls = as_list(spec.get("controls"))
    inst = as_list(spec.get("instrument"))
    distinct([y], [shock], contemp, controls, inst)
    p = int_option(spec, "lags", 4, 0, 12, "Number of lags")
    H = int_option(spec, "horizon", 12, 1, 40, "Horizon")
    kind = _check_outcome(spec)
    scale = spec.get("scale", "unit") or "unit"
    aug = (spec.get("lag_augment", "no") or "no") == "yes"
    cov = "robust" if aug else (spec.get("cov", "hac") or "hac")
    if cov not in ("hac", "robust"):
        raise OESError("Standard errors must be hac or robust.")
    bw = spec.get("bandwidth")
    bw = None if bw in (None, "", "auto") else int_option(spec, "bandwidth", 4, 0, 60, "HAC bandwidth")
    d, labels = _series_frame(df, spec, [y, shock] + contemp + controls + inst)
    n = len(d)
    if n < 30 + p:
        raise OESError("A local projection needs at least 30 observations after the lags.")
    sd = float(d[shock].std())
    sh = d[shock] / sd if scale == "sd" else d[shock]
    plag = p + (1 if aug else 0)
    lagged = [y, shock] + contemp + controls
    base = pd.DataFrame({shock: sh})
    for c in contemp:
        base[c] = d[c]
    for k in range(1, plag + 1):
        for c in lagged:
            base[f"{c}_l{k}"] = (sh if c == shock else d[c]).shift(k)
    Zexo = [c for c in base.columns if c != shock]
    hs, bs, ses, ns, fs = [], [], [], [], None
    for h in range(H + 1):
        Y = _outcome(d[y], h, kind)
        frame = pd.concat([Y.rename("_y"), base] + ([d[inst]] if inst else []), axis=1).dropna()
        if len(frame) < len(base.columns) + 12:
            break
        L = (h + 1) if bw is None else bw
        yv = frame["_y"].to_numpy(float)
        if inst:
            W = np.column_stack([np.ones(len(frame)), frame[Zexo].to_numpy(float)])
            X = np.column_stack([W, frame[shock].to_numpy(float)])
            Z = np.column_stack([W, frame[inst].to_numpy(float)])
            b, V, e, _ = iv_fit(yv, X, Z, "2sls", "kernel" if cov == "hac" else "robust", L)
            bh, sh_ = float(b[-1]), float(np.sqrt(V[-1, -1]))
            if h == 0:
                full = sm.OLS(frame[shock].to_numpy(float), Z).fit()
                rest = sm.OLS(frame[shock].to_numpy(float), W).fit()
                q = len(inst)
                F0 = ((rest.ssr - full.ssr) / q) / (full.ssr / full.df_resid)
                R = np.zeros((q, Z.shape[1])); R[:, W.shape[1]:] = np.eye(q)
                rob = sm.OLS(frame[shock].to_numpy(float), Z).fit(cov_type="HC0")
                fs = (F0, float(rob.wald_test(R, scalar=True).statistic) / q)
        else:
            X = sm.add_constant(frame[[shock] + Zexo].to_numpy(float), has_constant="add")
            fit = sm.OLS(yv, X).fit(cov_type="HAC", cov_kwds=dict(maxlags=L, use_correction=True)) if cov == "hac" \
                else sm.OLS(yv, X).fit(cov_type="HC1")
            bh, sh_ = float(fit.params[1]), float(fit.bse[1])
        hs.append(h); bs.append(bh); ses.append(sh_); ns.append(len(frame))
    if len(hs) < 2:
        raise OESError("Too few observations for the chosen lags and horizon. Reduce the number of lags or the horizon.")
    k, charts = _summary_and_charts("", y, shock, hs, bs, ses, sd, kind, scale)
    unit = "one standard deviation of the shock" if scale == "sd" else "a one-unit increase in the shock"
    tabs = [table("Impulse response by horizon", IRF_COLS, _irf_rows(hs, bs, ses, ns),
                  f"Response of {y} to {unit}. Outcome: {OUTCOMES[kind].replace('y', y)}. Each row is a separate regression.")]
    if inst:
        tabs.append(table("First stage at horizon 0: strength of the instrument(s)", ["Statistic", "Value"],
                          [["F (conventional)", f(fs[0], 3)], ["F (robust)", f(fs[1], 3)]],
                          "Rule of thumb: an F below 10 signals a weak instrument (Stock and Yogo 2005). Montiel Olea and Pflueger (2013) give a robust version."))
    diag = []
    sig = [h for h, b_, s_ in zip(hs, bs, ses) if abs(b_ / s_) > 1.96]
    diag.append(["Horizons with a response significant at 5%", len(sig), None, ", ".join(map(str, sig)) if sig else "none"])
    diag.append(["Peak response", f(bs[k]), None, f"at horizon {hs[k]}"])
    tabs.append(tests_table(diag, "Reading the response"))
    summary = [stat("Observations (h = 0)", ns[0]), stat("Horizons", hs[-1]), stat("Lags", p), stat("Shock std. dev.", sd, 4),
               stat("Peak response", bs[k], 4), stat("Peak horizon", hs[k]),
               stat("Standard errors", "Newey-West" if cov == "hac" else "robust (HC1)"), stat("Estimator", "LP-IV (2SLS)" if inst else "OLS")]
    notes = ["Standard errors at each horizon are " + ("Newey-West with bandwidth h + 1 (the horizon-h errors are a moving average of order h)."
             if (cov == "hac" and bw is None) else (f"Newey-West with bandwidth {bw}." if cov == "hac" else "heteroskedasticity-robust (HC1)."))]
    if aug:
        notes.append("Lag augmentation: one extra lag is included and not interpreted, so heteroskedasticity-robust standard errors are valid for stationary and persistent data alike (Montiel Olea and Plagborg-Moller 2021).")
    if contemp:
        notes.append("Contemporaneous controls (" + ", ".join(contemp) + ") are the variables ordered before the shock: they are held fixed at date t, which is the same recursive (Cholesky) identification as in a VAR.")
    else:
        notes.append("No contemporaneous controls: the shock variable is treated as exogenous given its own lags and the controls. If it responds to other variables within the period, add them as contemporaneous controls or use an instrument.")
    if len(hs) < H + 1:
        notes.append(f"The horizon was cut at {hs[-1]} because too few observations remain beyond it.")
    if kind == "level":
        notes.append("The outcome is the level of the response variable at t + h. If it is in logs this is the response in log points; for a variable in growth rates use the cumulative sum outcome to obtain the response of the level.")
    return result("lp", f"Local projection: {y}", ns[0], summary, tabs, charts, notes)


# ------------------------------------------------------------------------------------------------ panel
def plp(df, spec):
    y = need(spec, "y", "response variable")
    shock = need(spec, "shock", "shock variable")
    uc = need(spec, "unit", "unit identifier")
    tc = need(spec, "time", "time identifier")
    contemp = as_list(spec.get("contemp"))
    controls = as_list(spec.get("controls"))
    distinct([y], [shock], contemp, controls, [uc], [tc])
    p = int_option(spec, "lags", 2, 0, 8, "Number of lags")
    H = int_option(spec, "horizon", 8, 1, 30, "Horizon")
    kind = _check_outcome(spec)
    scale = spec.get("scale", "unit") or "unit"
    eff = spec.get("effects", "unit") or "unit"
    if eff not in ("unit", "twoway", "none"):
        raise OESError("Effects must be unit, twoway or none.")
    cov = spec.get("cov", "clustered") or "clustered"
    if cov not in ("clustered", "dk"):
        raise OESError("Standard errors must be clustered or dk (Driscoll-Kraay).")
    bw = spec.get("bandwidth")
    bw = None if bw in (None, "", "auto") else int_option(spec, "bandwidth", 2, 0, 20, "Driscoll-Kraay bandwidth")
    cols = [y, shock] + contemp + controls
    check_columns(df, [uc, tc] + cols)
    d = df[[uc, tc] + cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    tnum = pd.to_numeric(d[tc], errors="coerce")
    if tnum.notna().all():
        tv = tnum
    else:
        dt = pd.to_datetime(d[tc], errors="coerce")
        tv = dt if dt.notna().all() else d[tc].astype(str).astype("category").cat.codes
    d["_t"] = tv.rank(method="dense").astype(int) - 1
    d["_u"] = d[uc].astype("category").cat.codes
    if d.duplicated(["_u", "_t"]).any():
        raise OESError("Some unit-time pairs appear more than once. Each unit needs one row per period.")
    G, T = d["_u"].nunique(), int(d["_t"].max()) + 1
    full = pd.MultiIndex.from_product([range(G), range(T)], names=["_u", "_t"])
    d = d.set_index(["_u", "_t"]).reindex(full)
    if G < 5 or T < 12:
        raise OESError("A panel local projection needs at least 5 units and 12 periods.")
    sd = float(d[shock].std())
    S = d[shock] / sd if scale == "sd" else d[shock]
    gb = lambda s, k: s.groupby(level=0).shift(k)
    base = pd.DataFrame({shock: S})
    for c in contemp:
        base[c] = d[c]
    for k in range(1, p + 1):
        for c in [y, shock] + contemp + controls:
            base[f"{c}_l{k}"] = gb(S if c == shock else d[c], k)
    names_x = list(base.columns)
    hs, bs, ses, ns = [], [], [], []
    for h in range(H + 1):
        Y = (gb(d[y], -h) if kind == "level" else
             gb(d[y], -h) - gb(d[y], 1) if kind == "diff" else
             sum(gb(d[y], -j) for j in range(h + 1)))
        frame = pd.concat([Y.rename("_y"), base], axis=1).dropna()
        if len(frame) < 10 * (len(names_x) + 1) or frame.index.get_level_values(0).nunique() < 5:
            break
        u = frame.index.get_level_values(0).to_numpy()
        t = frame.index.get_level_values(1).to_numpy()
        A = frame.to_numpy(float)
        extra = 0
        if eff == "unit":
            A = _panel._demean(A, u)
        elif eff == "twoway":
            A = _panel._two_way(A, u, t)
            extra = len(np.unique(t)) - 1
        else:
            A = np.column_stack([A, np.ones(len(A))])  # constant last: columns are _y, regressors, const
        Yw, Xw = A[:, 0], A[:, 1:]
        if cov == "dk":
            extra += len(np.unique(u)) if eff != "none" else 0
        b = np.linalg.lstsq(Xw, Yw, rcond=None)[0]
        e = Yw - Xw @ b
        V = _panel._sandwich(Xw, e, groups=u, time=t, kind=cov, bw=bw, extra_df=extra)
        hs.append(h); bs.append(float(b[0])); ses.append(float(np.sqrt(V[0, 0]))); ns.append(len(frame))
    if len(hs) < 2:
        raise OESError("Too few observations for the chosen lags and horizon. Reduce the number of lags or the horizon.")
    k, charts = _summary_and_charts("", y, shock, hs, bs, ses, sd, kind, scale)
    unit = "one standard deviation of the shock" if scale == "sd" else "a one-unit increase in the shock"
    cov_label = _panel.COV_LABEL[cov] + (f", bandwidth {bw}" if (cov == "dk" and bw is not None) else "")
    tabs = [table("Impulse response by horizon", IRF_COLS, _irf_rows(hs, bs, ses, ns),
                  f"Response of {y} to {unit}. Outcome: {OUTCOMES[kind].replace('y', y)}. Each row is a separate regression. Standard errors: {cov_label}.")]
    sig = [h for h, b_, s_ in zip(hs, bs, ses) if abs(b_ / s_) > 1.96]
    tabs.append(tests_table([["Horizons with a response significant at 5%", len(sig), None, ", ".join(map(str, sig)) if sig else "none"],
                             ["Peak response", f(bs[k]), None, f"at horizon {hs[k]}"]], "Reading the response"))
    summary = [stat("Observations (h = 0)", ns[0]), stat("Units", G), stat("Periods", T), stat("Horizons", hs[-1]), stat("Lags", p),
               stat("Peak response", bs[k], 4), stat("Peak horizon", hs[k]),
               stat("Fixed effects", {"unit": "Unit", "twoway": "Unit and time", "none": "None"}[eff])]
    notes = [f"Standard errors: {cov_label}. Clustering by unit allows any serial correlation within a unit, including the overlap between horizons.",
             "Each regression uses the within transformation, so the response is identified from changes over time within units."]
    if eff == "twoway":
        notes.append("Time effects remove everything that is common to all units, including the average response of the economy to the shock. The response is identified from differences across units.")
    if eff != "none":
        notes.append("With fixed effects and lagged dependent-variable controls, a small-T panel has Nickell bias; local projections are less affected than a dynamic panel regression, but check that the response is stable to the number of lags.")
    if len(hs) < H + 1:
        notes.append(f"The horizon was cut at {hs[-1]} because too few observations remain beyond it.")
    return result("plp", f"Panel local projection: {y}", ns[0], summary, tabs, charts, notes)
