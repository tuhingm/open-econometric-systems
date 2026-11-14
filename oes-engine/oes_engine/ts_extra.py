"""ARIMA/SARIMA, GARCH family, quantile regression, cointegration tests and wavelet coherence.
Pure NumPy, SciPy and statsmodels (the `arch` package is not available in the browser). Same result format as the other models (core.py)."""
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .core import (OESError, as_list, chart, distinct, f, int_option, float_option, need, result, stat, table, test_row, tests_table)
from .timeseries import _series_frame


def _nums(v, default, label):
    if v in (None, ""):
        return list(default)
    try:
        return [float(s) for s in (v.split(",") if isinstance(v, str) else v)]
    except ValueError:
        raise OESError(f"{label} must be a comma-separated list of numbers.")


# ---------------------------------------------------------------------------------------------------------------
# ARIMA / SARIMA
# ---------------------------------------------------------------------------------------------------------------
def arima(df, spec):
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    y = need(spec, "y", "series")
    d, labels = _series_frame(df, spec, [y])
    x = d[y].to_numpy(float); n = len(x)
    s = int_option(spec, "season", 0, 0, 52, "Seasonal period")
    maxp = int_option(spec, "max_p", 3, 0, 5, "Maximum p"); maxq = int_option(spec, "max_q", 3, 0, 5, "Maximum q")
    h = int_option(spec, "holdout", 12, 4, 60, "Hold-out periods")
    dsel = spec.get("d") or "auto"
    if n < max(60, 3 * h):
        raise OESError("The series is too short for this hold-out. Use at least 60 observations.")
    # unit-root tests
    rows = []
    def ur(z, lab):
        a = adfuller(z, autolag="AIC"); k = kpss(z, regression="c", nlags="auto")
        rows.append([lab, f(a[0], 3), f(a[1], 4), f(k[0], 3), f(k[1], 4)]); return a[1], k[1]
    pa0, pk0 = ur(x, "level"); pa1, pk1 = ur(np.diff(x), "first difference")
    dd = int(dsel) if dsel != "auto" else (0 if pa0 < .10 else 1)
    train, test = x[:-h], x[-h:]
    seas = (1, 1 if spec.get("sdiff") == "yes" else 0, 0, s) if s > 1 else (0, 0, 0, 0)
    trend = "c" if dd == 0 and seas[1] == 0 else "n"
    def fit(p, q, data):
        return SARIMAX(data, order=(p, dd, q), seasonal_order=seas, trend=trend).fit(disp=False, maxiter=200)
    grid = []
    for p in range(maxp + 1):
        for q in range(maxq + 1):
            try:
                r = fit(p, q, train); grid.append((p, q, r.aic, r.bic))
            except Exception:
                continue
    if not grid:
        raise OESError("No ARIMA model could be estimated.")
    g = pd.DataFrame(grid, columns=["p", "q", "AIC", "BIC"]).sort_values("BIC")
    p, q = int(g.iloc[0].p), int(g.iloc[0].q)
    m = fit(p, q, train)
    names = list(m.param_names)
    se = np.asarray(m.bse); pr = np.asarray(m.params)
    lb = acorr_ljungbox(m.resid[max(dd + p, 1):], lags=[6, 12, 24], return_df=True)
    res_ = m.resid[max(dd + p, 1):]
    tests = [test_row(f"Ljung-Box serial correlation ({L} lags)", lb.lb_stat.iloc[i], lb.lb_pvalue.iloc[i], "H0: no serial correlation") for i, L in enumerate((6, 12, 24))]
    ar = het_arch(res_, nlags=6); tests.append(test_row("ARCH LM (6 lags)", ar[0], ar[1], "H0: no ARCH effects"))
    jb = stats.jarque_bera(res_); tests.append(test_row("Jarque-Bera normality", jb[0], jb[1], "H0: normal residuals"))
    # rolling one-step forecasts with the selected order against a random-walk / mean benchmark
    fc, bench = [], []
    for i in range(h):
        data = x[:n - h + i]
        try:
            r = fit(p, q, data)
            fc.append(float(r.forecast(1)[0]))
        except Exception:
            fc.append(np.nan)
        bench.append(float(data[-1]))
    fc, bench = np.array(fc), np.array(bench)
    e1, e0 = test - fc, test - bench
    rmse1, rmse0 = float(np.sqrt(np.nanmean(e1 ** 2))), float(np.sqrt(np.mean(e0 ** 2)))
    dm = e1 ** 2 - e0 ** 2; ok = np.isfinite(dm)
    dmstat = dmp = np.nan
    if ok.sum() > 4:
        v = dm[ok]; bw = max(int(round(ok.sum() ** (1 / 3))), 1)
        u = v - v.mean(); lrv = u @ u / len(v) + 2 * sum((1 - l / (bw + 1)) * (u[l:] @ u[:-l]) / len(v) for l in range(1, bw + 1))
        dmstat = v.mean() / np.sqrt(max(lrv, 1e-12) / len(v)); dmp = 2 * stats.norm.sf(abs(dmstat))
    # multi-step forecast with intervals from the full-sample fit
    full = fit(p, q, x); fo = full.get_forecast(h); fm = np.asarray(fo.predicted_mean); ci = np.asarray(fo.conf_int(alpha=.05))
    tabs = [table("Unit-root tests", ["Series", "ADF stat", "ADF p", "KPSS stat", "KPSS p"], rows,
                  "ADF: H0 unit root. KPSS: H0 stationary. A disagreement usually signals a shifting mean; differencing is then judged by forecasts."),
            table("Order selection (training sample)", ["p", "q", "AIC", "BIC"], [[int(a), int(b), f(c, 2), f(e, 2)] for a, b, c, e in g.head(6).values],
                  f"Best six by BIC out of {len(g)} models, d = {dd}" + (f", seasonal period {s}" if s > 1 else "") + "."),
            table(f"ARIMA({p},{dd},{q}) estimates", ["Parameter", "Coef.", "Std. err.", "z", "p-value"],
                  [[nm, f(b), f(a), f(b / a, 3), f(2 * stats.norm.sf(abs(b / a)), 4)] for nm, b, a in zip(names, pr, se)]),
            tests_table(tests, "Residual diagnostics"),
            table("Hold-out forecast accuracy (rolling one-step)", ["Forecast", "RMSE"], [[f"ARIMA({p},{dd},{q})", f(rmse1, 4)], ["Random walk (last value)", f(rmse0, 4)]],
                  f"Diebold-Mariano test of equal squared errors: statistic {f(dmstat, 3)}, p = {f(dmp, 4)} ({int(ok.sum())} forecasts).")]
    xs = list(range(n)); lab = labels
    ch = [chart(f"{y}: observed and {h}-step forecast", lab + [f"+{i + 1}" for i in range(h)], [("Observed", list(x) + [None] * h), ("Forecast", [None] * (n - 1) + [x[-1]] + list(fm))], "", y,
                band=([None] * n + list(ci[:, 0]), [None] * n + list(ci[:, 1])))]
    summary = [stat("Observations", n), stat("Selected p", p), stat("Selected d", dd), stat("Selected q", q), stat("Hold-out RMSE ratio vs random walk", rmse1 / rmse0, 3)]
    notes = ["Order chosen by BIC on the training sample (all but the last hold-out periods); the hold-out is forecast one step ahead with re-estimation at each origin. Constant included when no differencing is used; the reported intercept is the constant of the equation, so the process mean is intercept / (1 - sum of AR coefficients)."]
    return result("arima", f"ARIMA / SARIMA: {y}", n, summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# GARCH family with normal, Student-t and Hansen skewed-t errors, estimated by maximum likelihood
# ---------------------------------------------------------------------------------------------------------------
def _skewt_logpdf(z, nu, lam):
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - .5 * np.log(np.pi * (nu - 2))
    a = 4 * lam * np.exp(c) * (nu - 2) / (nu - 1); b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    w = b * z + a
    return np.log(b) + c - (nu + 1) / 2 * np.log1p((w / (1 + lam * np.sign(w))) ** 2 / (nu - 2))


def _logdens(z, dist, sh):
    if dist == "normal":
        return -.5 * np.log(2 * np.pi) - .5 * z ** 2
    nu = sh[0]
    if dist == "t":
        return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - .5 * np.log(np.pi * (nu - 2)) - (nu + 1) / 2 * np.log1p(z ** 2 / (nu - 2)))
    return _skewt_logpdf(z, nu, sh[1])


def _recursion(e, vol, par, var0):
    n = len(e); s2 = np.empty(n)
    if vol == "const":
        s2[:] = par[0]; return s2
    if vol == "egarch":
        om, al, ga, be = par; lv = np.log(var0); s2[0] = var0
        for t in range(1, n):
            z = e[t - 1] / np.sqrt(s2[t - 1])
            lv = om + al * abs(z) + ga * z + be * lv
            lv = min(max(lv, -50), 50); s2[t] = np.exp(lv)
        return s2
    om, al, ga, be = par; s2[0] = var0
    for t in range(1, n):
        s2[t] = om + (al + (ga if e[t - 1] < 0 else 0.0) * (vol == "gjr")) * e[t - 1] ** 2 + be * s2[t - 1]
    return s2


def _unpack(theta, vol, dist):
    mu = theta[0]
    k = {"const": 1, "garch": 4, "gjr": 4, "egarch": 4}[vol]
    vp = theta[1:1 + k]; sh = theta[1 + k:]
    if dist != "normal":
        sh = np.array([2.05 + np.exp(sh[0])] + ([np.tanh(sh[1])] if dist == "skewt" else []))
    return mu, vp, sh


def _negll(theta, r, vol, dist, var0):
    mu, vp, sh = _unpack(theta, vol, dist)
    if vol in ("garch", "gjr"):
        if vp[0] <= 0 or vp[1] < 0 or vp[3] < 0 or vp[1] + (vp[2] / 2 if vol == "gjr" else 0) + vp[3] >= 0.9999 or (vol == "garch" and vp[2] != 0):
            return 1e10
        if vol == "gjr" and vp[1] + vp[2] < 0:
            return 1e10
    if vol == "egarch" and abs(vp[3]) >= .9999:
        return 1e10
    e = r - mu
    s2 = _recursion(e, vol, vp, var0)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e10
    z = e / np.sqrt(s2)
    ll = _logdens(z, dist, sh) - .5 * np.log(s2)
    v = -ll.sum()
    return v if np.isfinite(v) else 1e10


def _hessian(fn, x, eps=1e-4):
    k = len(x); H = np.zeros((k, k)); h = eps * np.maximum(np.abs(x), 1e-2)
    f0 = fn(x)
    for i in range(k):
        for j in range(i, k):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = H[j, i] = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * h[i] * h[j])
    return H


def _fit_garch(r, vol, dist):
    n = len(r); var0 = float(r.var()); mu0 = float(r.mean())
    if vol == "const":
        v0 = [mu0, var0]
    elif vol == "egarch":
        v0 = [mu0, (1 - .9) * np.log(var0), .1, -.05, .9]
    elif vol == "gjr":
        v0 = [mu0, var0 * .05, .03, .08, .88]
    else:
        v0 = [mu0, var0 * .05, .08, 0.0, .88]
    s0 = [] if dist == "normal" else ([np.log(6.0)] if dist == "t" else [np.log(6.0), 0.0])
    x0 = np.array(v0 + s0)
    fixed = None
    if vol == "garch":
        # the leverage slot is held at zero in the plain GARCH
        free = [i for i in range(len(x0)) if i != 3]
    else:
        free = list(range(len(x0)))
    def obj(z):
        th = x0.copy(); th[free] = z; return _negll(th, r, vol, dist, var0)
    best = None
    starts = [x0[free]]
    if vol != "const":
        alt = x0.copy(); alt[2] *= 2; alt[-len(s0) - 1 if s0 else -1] = .8 if vol != "egarch" else .7; starts.append(alt[free])
    for st in starts:
        try:
            o = optimize.minimize(obj, st, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8, adaptive=True))
            o = optimize.minimize(obj, o.x, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9, adaptive=True))
        except Exception:
            continue
        if best is None or o.fun < best.fun:
            best = o
    if best is None or best.fun >= 1e9:
        raise OESError("The likelihood could not be maximised for this specification.")
    th = x0.copy(); th[free] = best.x
    try:
        H = _hessian(obj, best.x); V = np.linalg.inv(H); se = np.sqrt(np.clip(np.diag(V), 0, None)); se[~np.isfinite(se)] = np.nan
    except Exception:
        se = np.full(len(free), np.nan)
    return dict(theta=th, free=free, se=se, ll=-best.fun, k=len(free), var0=var0, vol=vol, dist=dist)


_PNAMES = {"const": ["mean", "variance"], "garch": ["mean", "omega", "alpha", "gamma (leverage)", "beta"], "gjr": ["mean", "omega", "alpha", "gamma (leverage)", "beta"],
           "egarch": ["mean", "omega", "alpha (size)", "gamma (sign)", "beta"]}


def _garch_view(fit, r):
    th = fit["theta"]; vol, dist = fit["vol"], fit["dist"]
    mu, vp, sh = _unpack(th, vol, dist)
    e = r - mu; s2 = _recursion(e, vol, vp, fit["var0"])
    return mu, vp, sh, e, s2


def garch(df, spec):
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
    y = need(spec, "y", "return series")
    d, labels = _series_frame(df, spec, [y])
    r = d[y].to_numpy(float); n = len(r)
    if n < 150:
        raise OESError("GARCH models need at least 150 observations.")
    scale = float(r.std()); rs = r / scale if scale > 0 else r      # fit on a standardised scale for stable optimisation
    # tests for ARCH effects
    e0 = rs - rs.mean()
    arch_rows = [test_row(f"ARCH LM ({L} lags)", *het_arch(e0, nlags=L)[:2], "H0: no ARCH effects") for L in (1, 3, 6, 12)]
    lbq = acorr_ljungbox(e0 ** 2, lags=[12], return_df=True)
    arch_rows.append(test_row("Ljung-Box on squared deviations (12)", lbq.lb_stat.iloc[0], lbq.lb_pvalue.iloc[0], "H0: no clustering"))
    specs = [("Constant variance, t", "const", "t"), ("GARCH, normal", "garch", "normal"), ("GARCH, t", "garch", "t"), ("GJR, t", "gjr", "t"),
             ("EGARCH, t", "egarch", "t"), ("GJR, skewed t", "gjr", "skewt"), ("EGARCH, skewed t", "egarch", "skewt")]
    fits = {}
    for name, v, dist in specs:
        try:
            fits[name] = _fit_garch(rs, v, dist)
        except OESError:
            continue
    if "GARCH, t" not in fits:
        raise OESError("The GARCH(1,1) model with t errors could not be estimated on these data.")
    rows = []
    for name, ft in fits.items():
        k = ft["k"]; ll = ft["ll"] - n * np.log(scale)                  # log-likelihood on the original scale
        rows.append([name, k, f(ll, 2), f(-2 * ll + 2 * k, 2), f(-2 * ll + k * np.log(n), 2)])
    tab = pd.DataFrame(rows, columns=["Model", "k", "ll", "AIC", "BIC"])
    pick = spec.get("model") or "best"
    bname = tab.loc[tab.BIC.astype(float).idxmin(), "Model"] if pick in ("best", "") else pick
    if bname not in fits:
        bname = tab.loc[tab.BIC.astype(float).idxmin(), "Model"]
    ft = fits[bname]; mu, vp, sh, e, s2 = _garch_view(ft, rs)
    z = e / np.sqrt(s2)
    # parameter table in original units
    names = _PNAMES[ft["vol"]] + (["df (nu)"] if ft["dist"] != "normal" else []) + (["skew (lambda)"] if ft["dist"] == "skewt" else [])
    val = np.concatenate([[mu * scale], vp, sh]).astype(float)
    if ft["vol"] in ("garch", "gjr"):
        val[1] *= scale ** 2
        if ft["vol"] == "gjr":
            pass
    elif ft["vol"] == "egarch":
        val[1] += (1 - vp[3]) * 2 * np.log(scale)
    elif ft["vol"] == "const":
        val[1] *= scale ** 2
    ses = np.full(len(val), np.nan)
    # standard errors are on the internal parametrisation: report them only for mean and the volatility parameters (delta method skipped for shape)
    idx = ft["free"]; se_int = dict(zip(idx, ft["se"]))
    raw_to_pos = {0: 0}
    kv = {"const": 1, "garch": 4, "gjr": 4, "egarch": 4}[ft["vol"]]
    for pos in range(1 + kv):
        if pos in se_int:
            ses[pos] = se_int[pos] * (scale if pos == 0 else (scale ** 2 if (pos == 1 and ft["vol"] != "egarch") else 1))
    prow = []
    for i, nm in enumerate(names):
        if ft["vol"] == "garch" and nm == "gamma (leverage)":
            continue
        b = val[i]; s = ses[i] if i < len(ses) else np.nan
        prow.append([nm, f(b, 5), f(s, 5), f(b / s, 2) if np.isfinite(s) and s > 0 else None])
    pers = None
    if ft["vol"] == "garch":
        pers = vp[1] + vp[3]
    elif ft["vol"] == "gjr":
        pers = vp[1] + vp[2] / 2 + vp[3]
    elif ft["vol"] == "egarch":
        pers = vp[3]
    # standardised residual diagnostics
    zd = [test_row("Ljung-Box on z (12 lags)", *[acorr_ljungbox(z, lags=[12], return_df=True).iloc[0, i] for i in (0, 1)], "H0: no serial correlation"),
          test_row("Ljung-Box on z^2 (12 lags)", *[acorr_ljungbox(z ** 2, lags=[12], return_df=True).iloc[0, i] for i in (0, 1)], "H0: variance fully modelled")]
    neg = (z[:-1] < 0).astype(float); X = np.column_stack([np.ones(n - 1), neg, neg * z[:-1], (1 - neg) * z[:-1]])
    yy = z[1:] ** 2; bb = np.linalg.lstsq(X, yy, rcond=None)[0]; resid = yy - X @ bb
    s2e = resid @ resid / (n - 1 - 4); Vb = s2e * np.linalg.inv(X.T @ X); R = np.zeros((3, 4)); R[0, 1] = R[1, 2] = R[2, 3] = 1
    wald = float((R @ bb) @ np.linalg.solve(R @ Vb @ R.T, R @ bb))
    zd.append(test_row("Engle-Ng sign-bias (joint)", wald, stats.chi2.sf(wald, 3), "H0: no leverage left in z^2"))
    # value at risk, in-sample 1-step with the fitted model
    qfun = (lambda p: stats.norm.ppf(p)) if ft["dist"] == "normal" else (lambda p: float(np.quantile(z, p)))
    var_rows = []
    for a in (.05, .01):
        q = qfun(a) if ft["dist"] == "normal" else float(np.quantile(z, a))
        thr = mu + q * np.sqrt(s2); viol = (rs < thr).sum(); p0 = a
        phat = viol / n
        lr = -2 * ((n - viol) * np.log(1 - p0) + viol * np.log(p0)) + 2 * ((n - viol) * np.log(1 - phat) + viol * np.log(phat)) if 0 < viol < n else np.nan
        var_rows.append([f"{int(a * 100)}% VaR", int(viol), f(phat, 4), f(lr, 3), f(stats.chi2.sf(lr, 1), 4) if np.isfinite(lr) else None])
    vol = np.sqrt(s2) * scale
    tabs = [tests_table(arch_rows, "Tests for time-varying volatility", "Constant-variance residuals from the sample mean."),
            table("Specification comparison (original scale)", ["Model", "Parameters", "Log-likelihood", "AIC", "BIC"], rows,
                  "Maximum likelihood on the standardised series; log-likelihoods are converted back to the original units."),
            table(f"Estimates: {bname}", ["Parameter", "Estimate", "Std. err.", "z"], prow,
                  ("Persistence: " + str(f(pers, 4)) + ". " if pers is not None else "") + "Standard errors are numerical and refer to the internal parametrisation for shape parameters (not shown)."),
            tests_table(zd, "Standardised-residual diagnostics"),
            table("In-sample value at risk and Kupiec test", ["Level", "Violations", "Share", "LR (Kupiec)", "p-value"], var_rows,
                  "Violation share should match the nominal level; the quantile uses the empirical quantile of standardised residuals for t-type errors.")]
    ch = [chart(f"Conditional volatility ({bname})", labels, [("Conditional s.d.", vol), ("|deviation from mean|", np.abs(r - mu * scale))], "", "Percent / units of the series")]
    summary = [stat("Observations", n), stat("Selected model (BIC)", bname), stat("Persistence", pers if pers is not None else float("nan"), 3), stat("Log-likelihood", float(tab.set_index("Model").loc[bname, "ll"]), 1)]
    notes = ["Models are estimated by Nelder-Mead maximum likelihood in NumPy (no external GARCH package), on the series divided by its standard deviation. The OES version omits the multivariate DCC step shown in the guide."]
    return result("garch", f"GARCH family: {y}", n, summary, tabs, ch, notes)


# ---------------------------------------------------------------------------------------------------------------
# Quantile regression with a block bootstrap
# ---------------------------------------------------------------------------------------------------------------
def quantile(df, spec):
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "explanatory variables")); distinct([y], xs)
    taus = _nums(spec.get("taus"), [.05, .25, .5, .75, .95], "Quantiles")
    if not taus or any(not 0 < t < 1 for t in taus):
        raise OESError("Quantiles must lie strictly between 0 and 1.")
    taus = sorted(set(taus))
    d, labels = _series_frame(df, spec, [y] + xs)
    yv = d[y].to_numpy(float); n = len(d)
    X = np.column_stack([np.ones(n), d[xs].to_numpy(float)]); names = ["const"] + xs
    L = int_option(spec, "block", 1, 1, 40, "Block length"); R = int_option(spec, "boot", 200, 50, 1000, "Bootstrap replications")
    if n < 8 * X.shape[1]:
        raise OESError("Too few observations for this number of regressors.")
    est = {t: QuantReg(yv, X).fit(q=t, max_iter=5000).params for t in taus}
    ols = sm.OLS(yv, X).fit()
    rng = np.random.default_rng(int(spec.get("seed") or 0)); nb = int(np.ceil(n / L)); draws = {t: [] for t in taus}
    for _ in range(R):
        st = rng.integers(0, n - L + 1, nb); idx = np.concatenate([np.arange(s, s + L) for s in st])[:n]
        for t in taus:
            try:
                draws[t].append(QuantReg(yv[idx], X[idx]).fit(q=t, max_iter=5000).params)
            except Exception:
                draws[t].append(np.full(X.shape[1], np.nan))
    draws = {t: np.array(v) for t, v in draws.items()}
    coef_rows = []
    for k, nm in enumerate(names):
        for t in taus:
            b = est[t][k]; sd = np.nanstd(draws[t][:, k], ddof=1); lo, hi = np.nanpercentile(draws[t][:, k], [5, 95])
            coef_rows.append([nm, t, f(b, 4), f(sd, 4), f(b / sd, 2) if sd > 0 else None, f(lo, 4), f(hi, 4), f(ols.params[k], 4)])
    tabs = [table("Quantile regression coefficients", ["Variable", "Quantile", "Coef.", "Bootstrap s.e.", "t", "90% low", "90% high", "OLS coef."], coef_rows,
                  f"{R} bootstrap replications" + (f" (moving blocks of length {L})" if L > 1 else " (pairs bootstrap)") + "; 90% intervals are bootstrap percentiles.")]
    k1 = 1
    pairs = [(taus[0], taus[len(taus) // 2]), (taus[-1], taus[len(taus) // 2]), (taus[0], taus[-1])]
    drows = []
    for kk, nm in enumerate(names[1:], start=1):
        for a, b in pairs:
            if a == b: continue
            e_ = est[a][kk] - est[b][kk]; dd = draws[a][:, kk] - draws[b][:, kk]
            cen = dd - np.nanmean(dd); p = np.nanmean(np.abs(cen) >= abs(e_)); lo, hi = np.nanpercentile(dd, [2.5, 97.5])
            drows.append([nm, f"q{a:g} - q{b:g}", f(e_, 4), f(lo, 4), f(hi, 4), f(p, 4)])
    tabs.append(table("Does the effect differ across quantiles?", ["Variable", "Difference", "Estimate", "95% low", "95% high", "p-value (bootstrap)"], drows,
                      "Bootstrap distribution recentred at zero for the p-value."))
    fit_q = np.column_stack([X @ est[t] for t in taus]); cross = (np.diff(fit_q, axis=1) < 0).any(axis=1)
    cov = [[t, f((yv < fit_q[:, i]).mean(), 3)] for i, t in enumerate(taus)]
    tabs.append(table("In-sample coverage of the fitted quantiles", ["Quantile", "Share of outcomes below the fitted quantile"], cov, f"Quantile curves cross in {int(cross.sum())} of {n} observations."))
    ch = []
    for kk, nm in enumerate(names[1:], start=1):
        lo, hi = zip(*[np.nanpercentile(draws[t][:, kk], [5, 95]) for t in taus])
        ch.append(chart(f"Coefficient on {nm} by quantile", taus, [("Quantile regression", [est[t][kk] for t in taus]), ("OLS", [ols.params[kk]] * len(taus))], "Quantile", "Coefficient", band=(lo, hi)))
    lo5, hi5 = taus[0], taus[-1]
    fq = np.sort(fit_q, axis=1)
    ch.append(chart("Observed outcome and fitted quantiles (rearranged)", labels, [("Outcome", yv), (f"q{lo5:g}", fq[:, 0]), (f"q{taus[len(taus) // 2]:g}", fq[:, len(taus) // 2]), (f"q{hi5:g}", fq[:, -1])]))
    summary = [stat("Observations", n), stat("Quantiles", len(taus)), stat("Bootstrap replications", R), stat("Crossing observations", int(cross.sum()))]
    return result("quantile", f"Quantile regression: {y}", n, summary, tabs, ch, ["Fitted quantiles can cross; the final chart rearranges them by sorting (Chernozhukov, Fernandez-Val and Galichon)."])


# ---------------------------------------------------------------------------------------------------------------
# Cointegration: unit roots, Engle-Granger, Johansen, Gregory-Hansen, DOLS
# ---------------------------------------------------------------------------------------------------------------
def _adf_stat(u, maxlag=8):
    import statsmodels.api as sm
    best = None; du = np.diff(u); n = len(du)
    for k in range(0, maxlag + 1):
        Y = du[maxlag:]
        cols = [u[maxlag:-1]] + [du[maxlag - j:n - j] for j in range(1, k + 1)]
        res = sm.OLS(Y, np.column_stack(cols)).fit()
        ic = len(Y) * np.log(res.ssr / len(Y)) + 2 * (k + 1)
        if best is None or ic < best[0]:
            best = (ic, res.tvalues[0], k)
    return best[1], best[2]


def _gh_design(x, tb, model):
    T = x.shape[0]; tt = np.arange(T); D = (tt >= tb).astype(float)
    cols = [np.ones(T), D]
    if model == "C/T":
        cols.append(tt.astype(float))
    cols += [x[:, j] for j in range(x.shape[1])]
    if model == "C/S":
        cols += [x[:, j] * D for j in range(x.shape[1])]
    return np.column_stack(cols)


def _adf0(u):
    du = np.diff(u); xx = u[:-1]; b = (xx @ du) / (xx @ xx); e = du - b * xx
    return b / np.sqrt(e @ e / (len(du) - 1) / (xx @ xx))


def cointegration(df, spec):
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    y = need(spec, "y", "dependent variable"); xs = as_list(need(spec, "x", "regressors")); distinct([y], xs)
    d, labels = _series_frame(df, spec, [y] + xs)
    n = len(d)
    if n < 60:
        raise OESError("Cointegration tests need at least 60 observations.")
    yv = d[y].to_numpy(float); xm = d[xs].to_numpy(float)
    urows = []
    for nm in [y] + xs:
        s = d[nm].to_numpy(float)
        for lab, z in (("level", s), ("first difference", np.diff(s))):
            a = adfuller(z, regression="c", autolag="AIC"); k = kpss(z, regression="c", nlags="auto")
            urows.append([nm, lab, f(a[0], 3), f(a[1], 4), f(k[0], 3), f(k[1], 4)])
    tabs = [table("Unit-root tests", ["Series", "Form", "ADF stat", "ADF p", "KPSS stat", "KPSS p"], urows, "The tests are applied to levels and first differences; cointegration needs I(1) series.")]
    eg_rows = []
    jobs = [(y, yv, xm, "the regressors")]
    if len(xs) == 1:
        jobs.append((xs[0], xm[:, 0], yv[:, None], y))
    for dep, dv, regs, other in jobs:
        r = coint(dv, regs, trend="c", maxlag=8, autolag="aic")
        ols = sm.OLS(dv, sm.add_constant(regs)).fit()
        eg_rows.append([f"{dep} on {other}", f(r[0], 3), f(r[1], 4), f(r[2][1], 3), "; ".join(f"{b:.3f}" for b in ols.params[1:])])
    tabs.append(table("Engle-Granger test", ["Regression", "Statistic", "p-value", "5% critical value", "Long-run slope(s)"], eg_rows, "H0: no cointegration. The test is not symmetric in the choice of dependent variable."))
    if len(xs) >= 1:
        jrows = []
        data = d[[y] + xs].to_numpy(float)
        for k in (1, 2, 3, 4):
            j = coint_johansen(data, det_order=0, k_ar_diff=k)
            for r in range(min(len(xs) + 1, 3)):
                jrows.append([k, f"r <= {r}" if r else "r = 0", f(j.lr1[r], 2), f(j.cvt[r, 1], 2), f(j.lr2[r], 2), f(j.cvm[r, 1], 2)])
        tabs.append(table("Johansen test (constant in the cointegrating space)", ["VAR lag differences", "H0 rank", "Trace", "Trace 5% cv", "Max-eigenvalue", "Max-eig 5% cv"], jrows,
                          "Reject when the statistic exceeds the critical value."))
    # Gregory-Hansen with simulated critical values
    trim = .15; res = {}
    for m in ("C", "C/T", "C/S"):
        out = []
        for tb in range(int(trim * n), int((1 - trim) * n)):
            u = sm.OLS(yv, _gh_design(xm, tb, m)).fit().resid
            st, k = _adf_stat(u, 8); out.append((st, tb, k))
        res[m] = (min(out), out)
    Rsim = int_option(spec, "sims", 100, 20, 500, "Simulation replications"); rng = np.random.default_rng(int(spec.get("seed") or 0))
    sim = {m: [] for m in res}
    for _ in range(Rsim):
        ys = np.cumsum(rng.standard_normal(n)); xsim = np.cumsum(rng.standard_normal((n, len(xs))), axis=0)
        for m in sim:
            best = 1e9
            for tb in range(int(trim * n), int((1 - trim) * n), 2):
                Dm = _gh_design(xsim, tb, m); u = ys - Dm @ np.linalg.lstsq(Dm, ys, rcond=None)[0]
                best = min(best, _adf0(u))
            sim[m].append(best)
    cv = {m: np.percentile(v, [1, 5, 10]) for m, v in sim.items()}
    grows = []
    for m in res:
        (st, tb, k), _ = res[m]
        grows.append([m, f(st, 3), labels[tb], int(k), f(cv[m][0], 2), f(cv[m][1], 2), f(cv[m][2], 2), "yes" if st < cv[m][1] else "no"])
    tabs.append(table("Gregory-Hansen test with one structural break", ["Model", "Statistic", "Break date", "ADF lags", "1% cv", "5% cv", "10% cv", "Rejects at 5%"], grows,
                      f"Critical values simulated for independent random walks without break ({Rsim} replications, T = {n}, break grid every second date). Models: C = level shift, C/T = level shift with trend, C/S = regime shift (intercept and slopes)."))
    # DOLS and slopes before/after the C/S break
    tb = res["C/S"][0][1]
    Dm = _gh_design(xm, tb, "C/S"); ols = sm.OLS(yv, Dm).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
    nx = len(xs); b = ols.params; V = ols.cov_params(); srows = []
    for j in range(nx):
        i0 = 2 + j; i1 = 2 + nx + j
        w = np.zeros(len(b)); w[[i0, i1]] = 1; sea = float(np.sqrt(w @ V @ w))
        srows.append([xs[j], f(b[i0], 3), f(np.sqrt(V[i0, i0]), 3), f(b[i0] + b[i1], 3), f(sea, 3)])
    tabs.append(table(f"Long-run slopes before and after the break ({labels[tb]})", ["Regressor", "Before", "HAC s.e.", "After", "HAC s.e."], srows))
    def dols(yy, xx, lags=2):
        T = len(yy); dx = np.diff(xx, axis=0); dx = np.vstack([np.zeros((1, xx.shape[1])), dx])
        idx = np.arange(lags, T - lags)
        cols = [np.ones(len(idx))] + [xx[idx, j] for j in range(xx.shape[1])]
        for l in range(-lags, lags + 1):
            for j in range(xx.shape[1]):
                cols.append(dx[idx - l, j])
        r = sm.OLS(yy[idx], np.column_stack(cols)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        return r.params[1:1 + xx.shape[1]], r.bse[1:1 + xx.shape[1]], len(idx)
    drows = []
    for lab, sl in (("whole sample", slice(None)), ("before break", slice(0, tb)), ("after break", slice(tb, None))):
        if len(yv[sl]) < 25:
            continue
        pb, ps, nn = dols(yv[sl], xm[sl])
        for j in range(nx):
            drows.append([lab, xs[j], f(pb[j], 3), f(ps[j], 3), nn])
    tabs.append(table("Dynamic OLS (2 leads and 2 lags)", ["Sample", "Regressor", "Slope", "s.e. (HAC)", "n"], drows, "DOLS corrects for endogeneity of the regressors in the long-run relation."))
    ch = []
    for m in ("C", "C/T", "C/S"):
        path = res[m][1]
        ch.append(chart(f"Gregory-Hansen: ADF statistic by break date ({m})", [labels[b] for _, b, _ in path], [("ADF statistic", [s for s, _, _ in path]), ("5% critical value", [cv[m][1]] * len(path))], "Break date", "ADF statistic of the residual"))
    summary = [stat("Observations", n), stat("Engle-Granger p", float(eg_rows[0][2]), 3), stat("Break date (C/S)", labels[tb]), stat("Rejects at 5% (C/S)", grows[2][-1])]
    return result("cointegration", f"Cointegration: {y} and {', '.join(xs)}", n, summary, tabs, ch,
                  ["Phillips-Ouliaris tests, available in the guide through the `arch` package, are not part of the OES version."])


# ---------------------------------------------------------------------------------------------------------------
# Wavelet coherence
# ---------------------------------------------------------------------------------------------------------------
def _morlet_cwt(x, scales, w0=6.0):
    m = len(x); N = 1 << int(np.ceil(np.log2(2 * m)))
    xf = np.fft.fft(x - x.mean(), N); om = 2 * np.pi * np.fft.fftfreq(N)
    W = np.empty((len(scales), m), complex)
    for k, s in enumerate(scales):
        psi = np.pi ** -.25 * np.exp(-.5 * (s * om - w0) ** 2) * (om > 0) * np.sqrt(2 * np.pi * s)
        W[k] = np.fft.ifft(xf * psi)[:m]
    return W


def _smooth(A, scales):
    from scipy.ndimage import uniform_filter1d
    out = np.empty_like(A)
    for k, s in enumerate(scales):
        w = max(int(round(2 * s)), 1)
        out[k] = uniform_filter1d(A[k].real, w, mode="nearest") + 1j * uniform_filter1d(A[k].imag, w, mode="nearest")
    return uniform_filter1d(out.real, 3, axis=0, mode="nearest") + 1j * uniform_filter1d(out.imag, 3, axis=0, mode="nearest")


def _wtc(x, y, scales):
    Wx, Wy = _morlet_cwt(x, scales), _morlet_cwt(y, scales)
    Sxy = _smooth(Wx * np.conj(Wy) / scales[:, None], scales)
    Sx = _smooth(np.abs(Wx) ** 2 / scales[:, None], scales).real
    Sy = _smooth(np.abs(Wy) ** 2 / scales[:, None], scales).real
    return np.abs(Sxy) ** 2 / (Sx * Sy), np.angle(Sxy)


def wavelet(df, spec):
    xn = need(spec, "x", "first series"); yn = need(spec, "y", "second series"); distinct([xn], [yn])
    d, labels = _series_frame(df, spec, [xn, yn])
    n = len(d)
    if n < 100:
        raise OESError("Wavelet coherence needs at least 100 observations.")
    x = d[xn].to_numpy(float); y = d[yn].to_numpy(float)
    pmin = float_option(spec, "pmin", 16, "Shortest period"); pmax = float_option(spec, "pmax", min(512, n // 2), "Longest period")
    if pmin < 4 or pmax <= pmin or pmax > n:
        raise OESError("Choose periods with 4 <= shortest < longest <= number of observations.")
    nj = int(np.floor(8 * np.log2(pmax / pmin))) + 1
    period = pmin * 2 ** (np.arange(nj) / 8)
    scales = period / (4 * np.pi / (6 + np.sqrt(2 + 36)))
    R, ph = _wtc(x, y, scales)
    coi = np.minimum(np.arange(n), n - 1 - np.arange(n)); mask = coi[None, :] >= (np.sqrt(2) * scales)[:, None]
    if mask.sum() == 0:
        raise OESError("No part of the time-period plane lies inside the cone of influence; reduce the longest period.")
    rng = np.random.default_rng(int(spec.get("seed") or 0)); Rs = int_option(spec, "sims", 100, 20, 500, "Simulation replications")
    def ar1(v):
        v = v - v.mean(); a = (v[1:] @ v[:-1]) / (v[:-1] @ v[:-1]); return a, np.std(v[1:] - a * v[:-1])
    (a1, s1), (a2, s2) = ar1(x), ar1(y)
    def sim(a, s):
        e = rng.standard_normal(n + 100) * s; u = np.zeros(n + 100)
        for j in range(1, n + 100): u[j] = a * u[j - 1] + e[j]
        return u[100:]
    sims = np.array([_wtc(sim(a1, s1), sim(a2, s2), scales)[0] for _ in range(Rs)])
    c95 = np.percentile(sims, 95, axis=0); sig = (R > c95) & mask
    fa = ((sims > c95[None]) & mask[None]).sum((1, 2)) / mask.sum()
    # by period
    pick = [p for p in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512) if pmin <= p <= pmax]
    prow = []
    for p_ in pick:
        k = int(np.argmin(abs(period - p_))); m_ = mask[k]
        if m_.sum() == 0: continue
        prow.append([p_, f(R[k, m_].mean(), 3), f(c95[k, m_].mean(), 3), f((R[k, m_] > c95[k, m_]).mean(), 3), f(np.angle(np.exp(1j * ph[k, m_]).mean()), 2)])
    band = (period >= min(32, pmax)) & (period <= min(128, pmax))
    seg = np.array_split(np.arange(n), 6); trow = []; tmean = []
    for ix in seg:
        vals = R[np.ix_(band, ix)]; ok = mask[np.ix_(band, ix)]
        if ok.sum() == 0:
            trow.append([f"{labels[ix[0]]} to {labels[ix[-1]]}", None, None]); tmean.append(np.nan); continue
        trow.append([f"{labels[ix[0]]} to {labels[ix[-1]]}", f(vals[ok].mean(), 3), f(sig[np.ix_(band, ix)].sum() / ok.sum(), 3)]); tmean.append(vals[ok].mean())
    ss = sig[band]; pp = ph[band][ss]
    inph = float((np.abs(pp) < np.pi / 2).mean()) if len(pp) else float("nan")
    cm = float(np.angle(np.exp(1j * pp).mean())) if len(pp) else float("nan")
    tabs = [table("Coherence by period", ["Period", "Mean coherence", "Mean 95% critical value", "Share significant", "Mean phase (rad)"], prow,
                  "Averaged over dates inside the cone of influence. Phase near 0 = in phase, near +/-pi = opposite directions; a positive phase means the second series lags the first."),
            table("Coherence over time (periods 32 to 128, or the available range)", ["Segment", "Mean coherence", "Share significant"], trow),
            table("Overall", ["Item", "Value"], [["Share of the plane (inside the cone) above the 95% critical value", f(sig.sum() / mask.sum(), 4)], ["Same share in simulated null series, mean", f(fa.mean(), 4)],
                                                 ["Same share in simulated null series, 95th percentile", f(np.percentile(fa, 95), 4)], ["Significant points in the band: share in phase (|phase| < 90 degrees)", f(inph, 3)],
                                                 ["Circular mean phase in the band (radians)", f(cm, 3)], ["AR(1) coefficients (first, second series)", f"{a1:.3f}, {a2:.3f}"]],
                  "The null is two independent AR(1) processes fitted to the series. The share above the critical value should be compared with the simulated null share, because the 5% pointwise level overstates joint significance.")]
    ch = [chart("Mean squared coherence by period", [f(p, 1) for p in period], [("Coherence", [R[k, mask[k]].mean() if mask[k].any() else np.nan for k in range(nj)]),
                                                                                  ("95% critical value", [c95[k, mask[k]].mean() if mask[k].any() else np.nan for k in range(nj)])], "Period", "Squared coherence"),
          chart("Mean coherence over time (band 32 to 128 periods)", [f"{labels[ix[0]]}" for ix in seg], [("Coherence", tmean)], "", "Squared coherence", kind="bar")]
    summary = [stat("Observations", n), stat("Periods", f"{period[0]:.0f} to {period[-1]:.0f}"), stat("Share significant", float(sig.sum() / mask.sum()), 3), stat("Null share (mean)", float(fa.mean()), 3)]
    return result("wavelet", f"Wavelet coherence: {xn} and {yn}", n, summary, tabs, ch,
                  ["Morlet wavelet (w0 = 6), smoothing over time and three neighbouring scales (Torrence and Webster 1999); significance by Monte Carlo from AR(1) nulls. The coherence heat map in the guide is shown here as averages by period and by sub-period."])
