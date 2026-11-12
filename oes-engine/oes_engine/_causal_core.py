"""Design selection and diagnostics for the Causal Inference Research Guide.

analyze(csv_text, roles_json) reads a CSV and a JSON dict of column roles:
    outcome (required), treat, unit, time, running, cutoff, instruments (list), controls (list)
and identifies the research design from the roles and the data:

* running variable and cutoff                  -> regression discontinuity (sharp, or fuzzy if treat is given and
                                                  not determined by the cutoff): local linear estimate with the
                                                  Imbens-Kalyanaraman bandwidth, McCrary density test, covariate
                                                  continuity, placebo cutoffs and bandwidth sensitivity
* instrument(s)                                -> instrumental variables: OLS vs 2SLS, first-stage F,
                                                  Anderson-Rubin test, endogeneity test, Hansen J if overidentified
* treatment varying over time (unit and time) -> the number of treated units, the comparison units and the timing
                                                  decide: interrupted time series, synthetic control, DiD with a
                                                  common date, or staggered DiD; an event-study pre-trend test
                                                  sends designs with non-parallel pre-trends to generalized
                                                  synthetic control (interactive fixed effects)

Returns a JSON string. The same file runs in the browser (Pyodide) for uploaded data and offline for the examples.
All estimators are written out in NumPy / SciPy so they run without extra packages.
"""
import io, json, math, warnings
import numpy as np
import pandas as pd

FEW_TREATED = 3      # this many treated units or fewer: synthetic control / interrupted time series
MIN_DONORS = 5       # synthetic control needs at least this many untreated comparison units
BOOT = 199           # bootstrap replications (cluster bootstrap over units)


def _f(x, nd=4):
    try:
        x = float(x)
    except Exception:
        return None
    return None if not np.isfinite(x) else round(x, nd)


def _ncdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _p2(z):
    return 2 * (1 - _ncdf(abs(z)))


def _chi2_sf(x, df):
    from scipy.stats import chi2
    return float(chi2.sf(x, df))


def _f_sf(x, d1, d2):
    from scipy.stats import f
    return float(f.sf(x, d1, d2))


def _ols(y, X, w=None):
    if w is not None:
        sw = np.sqrt(w)
        Xw, yw = X * sw[:, None], y * sw
    else:
        Xw, yw = X, y
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    b = XtX_inv @ (Xw.T @ yw)
    return b, y - X @ b, XtX_inv


def _hc1(X, e, XtX_inv, w=None):
    n, k = X.shape
    s = X * (e * (w if w is not None else 1.0))[:, None]
    return XtX_inv @ (s.T @ s) @ XtX_inv * n / max(n - k, 1)


def _cluster(X, e, XtX_inv, g):
    n, k = X.shape
    G = len(np.unique(g))
    order = np.argsort(g, kind="stable")
    Xs, es, gs = X[order], e[order], g[order]
    cut = np.flatnonzero(np.diff(gs)) + 1
    meat = np.zeros((k, k))
    for Xg, eg in zip(np.split(Xs, cut), np.split(es, cut)):
        s = Xg.T @ eg
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / max(n - k, 1)) if G > 1 else 1.0
    return c * XtX_inv @ meat @ XtX_inv


def _err(msg):
    return json.dumps({"error": msg})


# ================================================================ reading

def _read(csv_text, roles):
    try:
        raw = pd.read_csv(io.StringIO(csv_text), sep=None, engine="python")
    except Exception as e:
        return None, f"Could not read the file as CSV ({e})."
    raw.columns = [str(c).strip() for c in raw.columns]
    need = [roles.get(k) for k in ("outcome", "treat", "unit", "time", "running") if roles.get(k)]
    need += list(roles.get("instruments") or []) + list(roles.get("controls") or [])
    miss = [c for c in need if c not in raw.columns]
    if miss:
        return None, "Column(s) not found in the file: " + ", ".join(miss)
    if not roles.get("outcome"):
        return None, "Choose the outcome variable."
    return raw, None


def _time_order(s):
    t = pd.to_numeric(s, errors="coerce")
    if t.notna().all():
        return t.astype(float), "numeric"
    d = pd.to_datetime(s, errors="coerce")
    if d.notna().all():
        return (d - d.min()).dt.days.astype(float), "date"
    return s.astype(str).map({v: i for i, v in enumerate(sorted(s.astype(str).unique()))}).astype(float), "text"


# ================================================================ regression discontinuity

def _ll_fit(x, y, h, c=0.0, extra=None):
    """Local linear regression on both sides of c with a triangular kernel; returns jump, robust se, n used."""
    xc = x - c
    w = np.clip(1 - np.abs(xc) / h, 0, None)
    m = w > 0
    xc, y, w = xc[m], y[m], w[m]
    D = (xc >= 0).astype(float)
    X = np.column_stack([np.ones(len(xc)), D, xc, D * xc])
    if extra is not None:
        X = np.column_stack([X, extra[m]])
    if len(y) < X.shape[1] + 5 or D.sum() < 3 or (1 - D).sum() < 3:
        return None, None, int(m.sum())
    b, e, I = _ols(y, X, w)
    V = _hc1(X, e, I, w)
    return float(b[1]), float(math.sqrt(max(V[1, 1], 0))), int(m.sum())


def _ik_bandwidth(x, y, c=0.0):
    """Imbens and Kalyanaraman (2012) MSE-optimal bandwidth for local linear RD with a triangular kernel."""
    x = x - c
    N = len(x)
    sx = np.std(x, ddof=1)
    h1 = 1.84 * sx * N ** (-1 / 5)
    L, R = (x < 0) & (x >= -h1), (x >= 0) & (x <= h1)
    if L.sum() < 5 or R.sum() < 5:
        return None
    s2l, s2r = np.var(y[L], ddof=1), np.var(y[R], ddof=1)
    fc = (L.sum() + R.sum()) / (2 * N * h1)
    ml, mr = np.median(x[x < 0]), np.median(x[x >= 0])
    W = (x >= ml) & (x <= mr)
    Xc = np.column_stack([np.ones(W.sum()), (x[W] >= 0), x[W], x[W] ** 2, x[W] ** 3])
    b, _, _ = _ols(y[W], Xc)
    m3 = 6 * b[4]
    if abs(m3) < 1e-12:
        m3 = 1e-12
    Nl, Nr = (x < 0).sum(), (x >= 0).sum()
    h2l = 3.56 * (s2l / (fc * m3 ** 2)) ** (1 / 7) * Nl ** (-1 / 7)
    h2r = 3.56 * (s2r / (fc * m3 ** 2)) ** (1 / 7) * Nr ** (-1 / 7)
    def m2(side, h2):
        S = (x < 0) & (x >= -h2) if side == "l" else (x >= 0) & (x <= h2)
        if S.sum() < 6:
            return 0.0, int(S.sum())
        bb, _, _ = _ols(y[S], np.column_stack([np.ones(S.sum()), x[S], x[S] ** 2]))
        return 2 * bb[2], int(S.sum())
    m2l, nl2 = m2("l", h2l)
    m2r, nr2 = m2("r", h2r)
    rl = 2160 * s2l / (max(nl2, 1) * h2l ** 4)
    rr = 2160 * s2r / (max(nr2, 1) * h2r ** 4)
    h = 3.4375 * ((s2l + s2r) / (fc * ((m2r - m2l) ** 2 + rl + rr))) ** (1 / 5) * N ** (-1 / 5)
    return float(h)


def _plot_mask(x):
    """Exclude extreme mass points (e.g. vote shares of exactly 0 or 1) and the outer 2% from plots only."""
    x = np.asarray(x, float)
    keep = np.ones(len(x), bool)
    for v in (x.min(), x.max()):
        if (x == v).mean() > 0.01:
            keep &= x != v
    q = np.quantile(x[keep], [0.02, 0.98])
    return keep & (x >= q[0]) & (x <= q[1])


def _mccrary(x, c=0.0):
    """McCrary (2008) density discontinuity test: log difference of the density just right and left of c."""
    x = np.asarray(x, float) - c
    n = len(x)
    b = 2 * np.std(x, ddof=1) * n ** (-1 / 2)
    lo, hi = np.floor(x.min() / b) * b, np.ceil(x.max() / b) * b
    edges = np.arange(lo, hi + b / 2, b)
    if len(edges) < 8:
        return None
    cnt, _ = np.histogram(x, edges)
    mid = (edges[:-1] + edges[1:]) / 2
    fy = cnt / (n * b)
    # rule-of-thumb bandwidth from a 4th-order polynomial on each side
    hs = []
    for side in (mid < 0, mid >= 0):
        if side.sum() < 6:
            return None
        P = np.column_stack([mid[side] ** k for k in range(5)])
        bb, e, _ = _ols(fy[side], P)
        f2 = 2 * bb[2] + 6 * bb[3] * mid[side] + 12 * bb[4] * mid[side] ** 2
        s2 = float(e @ e) / max(side.sum() - 5, 1)
        rng = mid[side].max() - mid[side].min()
        hs.append(3.348 * (s2 * rng / max(float((f2 ** 2).sum()), 1e-12)) ** (1 / 5))
    h = float(np.mean(hs))
    def side_fit(S):
        xs, ys = mid[S], fy[S]
        w = np.clip(1 - np.abs(xs) / h, 0, None)
        k = w > 0
        if k.sum() < 3:
            return None
        bb, _, _ = _ols(ys[k], np.column_stack([np.ones(k.sum()), xs[k]]), w[k])
        return float(bb[0])
    fl, fr = side_fit(mid < 0), side_fit(mid >= 0)
    if not fl or not fr or fl <= 0 or fr <= 0:
        return None
    theta = math.log(fr) - math.log(fl)
    se = math.sqrt((1 / (n * h)) * (24 / 5) * (1 / fr + 1 / fl))
    pm = _plot_mask(x + c)
    lo_, hi_ = (x + c)[pm].min() - c, (x + c)[pm].max() - c
    sel = (mid >= lo_) & (mid <= hi_)
    hist = dict(mid=[_f(v + c, 5) for v in mid[sel]], f=[_f(v, 6) for v in fy[sel]])
    return dict(theta=_f(theta, 3), se=_f(se, 3), z=_f(theta / se, 2), p=_f(_p2(theta / se), 4), h=_f(h, 4), hist=hist)


def _rdd(df, roles):
    y, r = roles["outcome"], roles["running"]
    c = float(roles.get("cutoff") or 0)
    tr = roles.get("treat")
    cols = [y, r] + ([tr] if tr else [])
    d = df[cols + [k for k in df.columns if k not in cols]].copy()
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=cols)
    x, Y = d[r].to_numpy(float), d[y].to_numpy(float)
    if (x < c).sum() < 20 or (x >= c).sum() < 20:
        return {"error": "The RD design needs at least 20 observations on each side of the cutoff."}
    out = {"design": "RDD", "n": int(len(d)), "running": r, "cutoff": c, "outcome": y,
           "n_left": int((x < c).sum()), "n_right": int((x >= c).sum())}
    h = _ik_bandwidth(x, Y, c)
    if h is None or not np.isfinite(h) or h <= 0:
        h = 1.0 * np.std(x) * len(x) ** (-1 / 5)
    out["h"] = _f(h, 4)
    # sharp or fuzzy
    fuzzy = False
    if tr:
        D = pd.to_numeric(d[tr], errors="coerce").to_numpy(float)
        sharp_rule = (x >= c).astype(float)
        agree = np.nanmean(D == sharp_rule)
        agree_rev = np.nanmean(D == 1 - sharp_rule)
        fuzzy = max(agree, agree_rev) < 0.99
        out["treat_agree"] = _f(max(agree, agree_rev), 3)
    out["fuzzy"] = bool(fuzzy)
    est, se, nn = _ll_fit(x, Y, h, c)
    out["rd"] = dict(est=_f(est), se=_f(se), z=_f(est / se if se else None, 2), p=_f(_p2(est / se) if se else None, 4), n=nn)
    if fuzzy:
        fs, fse, _ = _ll_fit(x, D, h, c)
        out["first_stage"] = dict(est=_f(fs), se=_f(fse), z=_f(fs / fse if fse else None, 2))
        if fs and abs(fs) > 1e-8:
            # local Wald estimate with a delta-method standard error (ignoring the covariance of the two jumps)
            w = est / fs
            sew = abs(w) * math.sqrt((se / est) ** 2 + (fse / fs) ** 2) if est else se / abs(fs)
            out["wald"] = dict(est=_f(w), se=_f(sew), p=_f(_p2(w / sew), 4))
    # bandwidth sensitivity
    sens = []
    for mlt in (0.5, 0.75, 1.0, 1.5, 2.0):
        e2, s2, n2 = _ll_fit(x, Y, h * mlt, c)
        sens.append(dict(mult=mlt, h=_f(h * mlt, 4), est=_f(e2), se=_f(s2), n=n2))
    out["sensitivity"] = sens
    # placebo cutoffs at the medians on each side (using only that side's data)
    pl = []
    for side, S in (("left", x < c), ("right", x >= c)):
        pc = float(np.median(x[S]))
        e2, s2, _ = _ll_fit(x[S], Y[S], h, pc)
        pl.append(dict(side=side, cutoff=_f(pc, 4), est=_f(e2), se=_f(s2), p=_f(_p2(e2 / s2) if s2 else None, 4)))
    out["placebo"] = pl
    out["density"] = _mccrary(x, c)
    # covariate continuity: every other numeric column
    cov = []
    skip = set(cols) | set(roles.get("instruments") or [])
    for k in d.columns:
        if k in skip:
            continue
        v = pd.to_numeric(d[k], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        if ok.mean() < 0.8 or np.nanstd(v) == 0 or len(np.unique(v[ok])) < 2:
            continue
        e2, s2, _ = _ll_fit(x[ok], v[ok], h, c)
        if s2:
            cov.append(dict(var=k, est=_f(e2), se=_f(s2), p=_f(_p2(e2 / s2), 4)))
    out["covariates"] = cov[:8]
    # binned scatter for the plot
    keep = _plot_mask(x)
    nb = 20
    bins = []
    for S in (keep & (x < c), keep & (x >= c)):
        if S.sum() < nb:
            continue
        edges = np.quantile(x[S], np.linspace(0, 1, nb + 1))
        idx = np.clip(np.searchsorted(edges, x[S], side="right") - 1, 0, nb - 1)
        for k in range(nb):
            m = idx == k
            if m.any():
                bins.append([_f(x[S][m].mean(), 5), _f(Y[S][m].mean(), 5)])
    out["bins"] = bins
    # fitted local linear lines within the bandwidth
    fit = []
    for sgn in (-1, 1):
        S = ((x - c) * sgn >= 0) & (np.abs(x - c) <= h)
        if S.sum() > 3:
            bb, _, _ = _ols(Y[S], np.column_stack([np.ones(S.sum()), x[S] - c]), np.clip(1 - np.abs(x[S] - c) / h, 0, None))
            ends = [c, c + sgn * h]
            fit.append([[_f(e, 5), _f(bb[0] + bb[1] * (e - c), 5)] for e in ends])
    out["fit"] = fit
    flags = []
    if out["density"] and out["density"]["p"] is not None and out["density"]["p"] < 0.05:
        flags.append("density")
    if any(cv["p"] is not None and cv["p"] < 0.05 for cv in cov):
        flags.append("covariates")
    out["flags"] = flags
    out["model"] = "RDD"
    return out


# ================================================================ instrumental variables

def _iv(df, roles):
    y, dvar = roles["outcome"], roles.get("treat")
    Z = list(roles.get("instruments") or [])
    W = [c for c in (roles.get("controls") or []) if c not in (y, dvar, *Z)]
    if not dvar:
        return {"error": "For an IV design, choose the treatment (endogenous) variable as well as the instrument."}
    cols = [y, dvar] + Z + W
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)
    if n < 30:
        return {"error": f"Only {n} complete rows were found; IV needs many more."}
    Y, D = d[y].to_numpy(float), d[dvar].to_numpy(float)
    Zm, Wm = d[Z].to_numpy(float), (d[W].to_numpy(float) if W else np.zeros((n, 0)))
    C = np.column_stack([np.ones(n), Wm])
    k = C.shape[1]
    out = {"design": "IV", "n": int(n), "outcome": y, "treat": dvar, "instruments": Z, "controls": W}
    # OLS
    X = np.column_stack([D, C])
    b, e, I = _ols(Y, X)
    V = _hc1(X, e, I)
    out["ols"] = dict(est=_f(b[0]), se=_f(math.sqrt(V[0, 0])))
    # first stage
    F1 = np.column_stack([Zm, C])
    g, v, I1 = _ols(D, F1)
    V1 = _hc1(F1, v, I1)
    q = len(Z)
    gz, Vz = g[:q], V1[:q, :q]
    Fst = float(gz @ np.linalg.pinv(Vz) @ gz) / q
    out["first_stage"] = dict(coefs=[dict(var=z, est=_f(gz[i]), se=_f(math.sqrt(Vz[i, i]))) for i, z in enumerate(Z)],
                              F=_f(Fst, 2), p=_f(_f_sf(Fst, q, n - F1.shape[1]), 4), partial_r2=None)
    # partial R2 of the instruments
    _, v0, _ = _ols(D, C)
    out["first_stage"]["partial_r2"] = _f(1 - float(v @ v) / float(v0 @ v0), 4)
    # 2SLS
    Dhat = F1 @ g
    Xh = np.column_stack([Dhat, C])
    Xs = np.column_stack([D, C])
    Ih = np.linalg.pinv(Xh.T @ Xh)
    bs = Ih @ (Xh.T @ Y)
    es = Y - Xs @ bs
    Vs = _hc1(Xh, es, Ih)
    out["tsls"] = dict(est=_f(bs[0]), se=_f(math.sqrt(Vs[0, 0])), p=_f(_p2(bs[0] / math.sqrt(Vs[0, 0])), 4))
    # reduced form
    rf, er, Ir = _ols(Y, F1)
    Vr = _hc1(F1, er, Ir)
    out["reduced_form"] = [dict(var=z, est=_f(rf[i]), se=_f(math.sqrt(Vr[i, i]))) for i, z in enumerate(Z)]
    # endogeneity: control-function test
    Xc = np.column_stack([D, C, v])
    bc, ec, Ic = _ols(Y, Xc)
    Vc = _hc1(Xc, ec, Ic)
    tcf = bc[-1] / math.sqrt(Vc[-1, -1])
    out["endog"] = dict(t=_f(tcf, 2), p=_f(_p2(tcf), 4))
    # Anderson-Rubin test of beta = 0 and a 95% confidence set by grid
    def ar_stat(b0):
        yy = Y - b0 * D
        a, ea, Ia = _ols(yy, F1)
        Va = _hc1(F1, ea, Ia)
        return float(a[:q] @ np.linalg.pinv(Va[:q, :q]) @ a[:q]) / q
    ar0 = ar_stat(0.0)
    out["ar"] = dict(F=_f(ar0, 2), p=_f(_f_sf(ar0, q, n - F1.shape[1]), 4))
    se2 = math.sqrt(Vs[0, 0])
    grid = np.linspace(bs[0] - 8 * se2, bs[0] + 8 * se2, 401)
    from scipy.stats import f as fdist
    crit = float(fdist.ppf(0.95, q, n - F1.shape[1]))
    acc = [g0 for g0 in grid if ar_stat(g0) <= crit]
    if acc:
        bounded = acc[0] > grid[0] and acc[-1] < grid[-1]
        out["ar"]["ci"] = [_f(acc[0]), _f(acc[-1])]
        out["ar"]["bounded"] = bool(bounded)
    # overidentification
    if q > 1:
        a, ej, Ij = _ols(es, F1)
        J = n * (1 - float(ej @ ej) / float(((es - es.mean()) ** 2).sum()))
        out["overid"] = dict(J=_f(J, 2), df=q - 1, p=_f(_chi2_sf(J, q - 1), 4))
    out["weak"] = bool(Fst < 10)
    out["flags"] = (["weak"] if Fst < 10 else [])
    out["model"] = "IV"
    return out


# ================================================================ designs over time

def _prep_panel(df, roles):
    y, tr, u, t = roles["outcome"], roles.get("treat"), roles.get("unit"), roles.get("time")
    if not tr or not t:
        return None, "For a policy that starts at a point in time, choose the treatment indicator and the time variable."
    d = df[[c for c in (u, t, y, tr) if c]].copy()
    if not u:
        d["_unit"] = "all"; u = "_unit"
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d[tr] = pd.to_numeric(d[tr], errors="coerce")
    d = d.dropna(subset=[y, tr, t])
    tt, kind = _time_order(d[t])
    d["_t"] = tt
    periods = sorted(d["_t"].unique())
    d["_ti"] = d["_t"].map({p: i for i, p in enumerate(periods)}).astype(int)
    labels = d.drop_duplicates("_ti").set_index("_ti")[t].sort_index().astype(str).tolist()
    d["_u"] = d[u].astype(str)
    if d.duplicated(["_u", "_ti"]).any():
        return None, "Some unit appears more than once in the same period. Each unit-period must be one row."
    d["_d"] = (d[tr] > 0).astype(int)
    d = d.sort_values(["_u", "_ti"]).reset_index(drop=True)
    return dict(d=d, y=y, labels=labels, T=len(periods), unit=roles.get("unit"), time=t, treat=tr), None


def _adoption(d):
    g = d[d["_d"] == 1].groupby("_u")["_ti"].min()
    units = d["_u"].unique()
    G = pd.Series(np.inf, index=units)
    G.loc[g.index] = g.values
    # absorbing? (once treated, stays treated)
    after = d.merge(G.rename("_g"), left_on="_u", right_index=True)
    switch_off = bool(((after["_ti"] >= after["_g"]) & (after["_d"] == 0)).any())
    return G, switch_off


def _att_gt(d, y, G, base="universal"):
    """Callaway-Sant'Anna style group-time ATTs with the never-treated (or, if none, not-yet-treated) units as controls,
    universal base period g-1. Returns dict (g, t) -> att and the group sizes."""
    W = d.pivot(index="_u", columns="_ti", values=y)
    T = W.shape[1]
    never = G.index[~np.isfinite(G.values)]
    groups = sorted(set(G[np.isfinite(G.values)].astype(int)))
    res, size = {}, {}
    for g in groups:
        if g < 1:
            continue
        gu = G.index[G.values == g]
        size[g] = len(gu)
        for t in range(T):
            if t == g - 1:
                continue
            if len(never):
                cu = never
            else:
                cu = G.index[G.values > max(t, g - 1)]
            if len(cu) == 0:
                continue
            a = W.loc[gu, t] - W.loc[gu, g - 1]
            b = W.loc[cu, t] - W.loc[cu, g - 1]
            if a.notna().sum() == 0 or b.notna().sum() == 0:
                continue
            res[(g, t)] = float(a.mean() - b.mean())
    return res, size


def _event_agg(res, size, emin=-5, emax=5):
    agg = {}
    for (g, t), v in res.items():
        e = t - g
        if e < emin or e > emax:
            continue
        agg.setdefault(e, []).append((v, size[g]))
    out = {e: sum(v * w for v, w in L) / sum(w for _, w in L) for e, L in agg.items()}
    post = [(v, size[g]) for (g, t), v in res.items() if t >= g]
    overall = sum(v * w for v, w in post) / sum(w for _, w in post) if post else None
    return out, overall


def _twfe(d, y):
    codes_u, _ = pd.factorize(d["_u"])
    codes_t = d["_ti"].to_numpy()
    nu, nt = codes_u.max() + 1, codes_t.max() + 1
    yv = d[y].to_numpy(float); D = d["_d"].to_numpy(float)
    # within transformation for two-way FE on a possibly unbalanced panel: dummies for time, demean by unit
    T = (codes_t[:, None] == np.arange(1, nt)[None, :]).astype(float)
    def dm(a):
        a = np.asarray(a, float)
        if a.ndim == 1:
            return a - (np.bincount(codes_u, a, nu) / np.bincount(codes_u, minlength=nu))[codes_u]
        return np.column_stack([dm(a[:, j]) for j in range(a.shape[1])])
    X = np.column_stack([dm(D), dm(T)])
    b, e, I = _ols(dm(yv), X)
    V = _cluster(X, e, I, codes_u)
    return float(b[0]), float(math.sqrt(max(V[0, 0], 0)))


def _did(P, G, staggered):
    d, y = P["d"], P["y"]
    res, size = _att_gt(d, y, G)
    if not res:
        return {"error": "Could not form any comparison between treated and untreated units."}
    ev, overall = _event_agg(res, size)
    # cluster bootstrap over units
    rng = np.random.default_rng(12345)
    units = d["_u"].unique()
    by = {u: g for u, g in d.groupby("_u")}
    es = sorted(ev)
    boots_e, boots_o = [], []
    for _ in range(BOOT):
        pick = rng.choice(units, len(units), replace=True)
        parts = []
        for j, u0 in enumerate(pick):
            g = by[u0].copy(); g["_u"] = f"{u0}#{j}"; parts.append(g)
        db = pd.concat(parts, ignore_index=True)
        Gb = pd.Series([G[u0] for u0 in pick], index=[f"{u0}#{j}" for j, u0 in enumerate(pick)])
        if not np.isfinite(Gb.values).any() or np.isfinite(Gb.values).all() and not staggered:
            continue
        rb, sb = _att_gt(db, y, Gb)
        if not rb:
            continue
        eb, ob = _event_agg(rb, sb)
        boots_e.append([eb.get(e, np.nan) for e in es]); boots_o.append(ob)
    B = np.array(boots_e, float)
    se_e = np.nanstd(B, axis=0, ddof=1)
    se_o = float(np.nanstd(np.array(boots_o, float), ddof=1))
    out = {"event": [dict(e=int(e), est=_f(ev[e]), se=_f(s)) for e, s in zip(es, se_e)],
           "att": dict(est=_f(overall), se=_f(se_o), p=_f(_p2(overall / se_o) if se_o else None, 4))}
    # joint pre-trend test on the leads e = -5..-2 (the event window; e = -1 is the base period)
    pre = [i for i, e in enumerate(es) if e <= -2]
    if pre:
        v = np.array([ev[es[i]] for i in pre])
        Bp = B[:, pre]
        Bp = Bp[~np.isnan(Bp).any(axis=1)]
        if len(Bp) > len(pre) + 5:
            S = np.cov(Bp.T, ddof=1).reshape(len(pre), len(pre))
            W = float(v @ np.linalg.pinv(S) @ v)
            out["pretrend"] = dict(stat=_f(W, 2), df=len(pre), p=_f(_chi2_sf(W, len(pre)), 4))
    if "pretrend" not in out:
        out["pretrend"] = None
    tw, tws = _twfe(d, y)
    out["twfe"] = dict(est=_f(tw), se=_f(tws), p=_f(_p2(tw / tws) if tws else None, 4))
    return out


def _gsc(P, G, rmax=4):
    """Generalized synthetic control (Xu 2017) without covariates: Y(0)_it = a_i + xi_t + l_i' f_t. Factors are
    estimated on the never-treated units; each treated unit's loadings on its pre-treatment periods. r by leave-one-out
    cross-validation on the treated units' pre-treatment periods. Standard errors by bootstrapping units."""
    d, y = P["d"], P["y"]
    W = d.pivot(index="_ti", columns="_u", values=y)
    ctrl = [u for u in W.columns if not np.isfinite(G[u])]
    trt = [u for u in W.columns if np.isfinite(G[u])]
    Yc = W[ctrl].to_numpy(float)
    if np.isnan(Yc).any() or len(ctrl) < 5:
        return None
    T = Yc.shape[0]

    def factors(Yc, r):
        a = Yc.mean(0)
        R = Yc - a
        xi = R.mean(1)
        R = R - xi[:, None]
        if r == 0:
            return a, xi, np.zeros((T, 0))
        for _ in range(30):
            U, s, Vt = np.linalg.svd(R, full_matrices=False)
            F = U[:, :r] * math.sqrt(T)
            L = (R.T @ F) / T
            E = Yc - a - xi[:, None] - F @ L.T
            # refresh a and xi given the factor part
            Z = Yc - F @ L.T
            a = Z.mean(0); xi = (Z - a).mean(1)
            R = Yc - a - xi[:, None]
        return a, xi, F

    def fit_unit(yv, xi, F, pre, drop=None):
        idx = [t for t in pre if t != drop]
        X = np.column_stack([np.ones(len(idx)), F[idx]])
        b, *_ = np.linalg.lstsq(X, yv[idx] - xi[idx], rcond=None)
        return lambda tt: b[0] + F[tt] @ b[1:] + xi[tt]

    def cv(r):
        a, xi, F = factors(Yc, r)
        errs = []
        for u in trt:
            yv = W[u].to_numpy(float); g = int(G[u]); pre = [t for t in range(g) if np.isfinite(yv[t])]
            if len(pre) <= r + 2:
                return np.inf
            for t0 in pre:
                f = fit_unit(yv, xi, F, pre, drop=t0)
                errs.append((yv[t0] - f(t0)) ** 2)
        return float(np.mean(errs))

    scores = {r: cv(r) for r in range(rmax + 1)}
    r = min(scores, key=scores.get)

    def estimate(Wm, Gm, trt_u, ctrl_u):
        Yc_ = Wm[ctrl_u].to_numpy(float)
        a, xi, F = factors(Yc_, r) if Yc_.shape[1] > r + 1 else (None, None, None)
        if F is None:
            return None
        gaps = {}
        for u in trt_u:
            yv = Wm[u].to_numpy(float); g = int(Gm[u]); pre = [t for t in range(g) if np.isfinite(yv[t])]
            if len(pre) <= r + 1:
                continue
            f = fit_unit(yv, xi, F, pre)
            for t in range(T):
                if np.isfinite(yv[t]):
                    gaps.setdefault(t - g, []).append(yv[t] - f(t))
        ev = {e: float(np.mean(v)) for e, v in gaps.items()}
        post = [v for e, L in gaps.items() if e >= 0 for v in L]
        return ev, (float(np.mean(post)) if post else None)

    base = estimate(W, G, trt, ctrl)
    if base is None:
        return None
    ev, att = base
    # bootstrap units
    rng = np.random.default_rng(7)
    be, bo = [], []
    for _ in range(BOOT):
        cs = list(rng.choice(ctrl, len(ctrl), replace=True)); ts = list(rng.choice(trt, len(trt), replace=True))
        cols, Gb, tl, cl = {}, {}, [], []
        for j, u in enumerate(cs):
            k = f"c{j}"; cols[k] = W[u]; Gb[k] = np.inf; cl.append(k)
        for j, u in enumerate(ts):
            k = f"t{j}"; cols[k] = W[u]; Gb[k] = G[u]; tl.append(k)
        r_ = estimate(pd.DataFrame(cols), pd.Series(Gb), tl, cl)
        if r_ is None:
            continue
        be.append(r_[0]); bo.append(r_[1])
    es = sorted(ev)
    se_e = [float(np.nanstd([b.get(e, np.nan) for b in be], ddof=1)) for e in es]
    se_o = float(np.nanstd(np.array(bo, float), ddof=1))
    pre_rmspe = math.sqrt(np.mean([ev[e] ** 2 for e in es if e < 0])) if any(e < 0 for e in es) else None
    return dict(r=int(r), cv={str(k): _f(v, 5) for k, v in scores.items()}, att=dict(est=_f(att), se=_f(se_o), p=_f(_p2(att / se_o) if se_o else None, 4)),
                event=[dict(e=int(e), est=_f(ev[e]), se=_f(s)) for e, s in zip(es, se_e) if -10 <= e <= 10], pre_rmspe=_f(pre_rmspe))


def _synth_weights(y1, Y0):
    """Nonnegative weights summing to one that minimise ||y1 - Y0 w||, via NNLS with a heavily weighted adding-up row."""
    from scipy.optimize import nnls
    k = 1e3 * max(1.0, float(np.abs(Y0).max()))
    A = np.vstack([Y0, k * np.ones((1, Y0.shape[1]))])
    b = np.concatenate([y1, [k]])
    w, _ = nnls(A, b)
    return w / w.sum() if w.sum() > 0 else w


def _scm(P, G):
    d, y = P["d"], P["y"]
    W = d.pivot(index="_ti", columns="_u", values=y)
    donors = [u for u in W.columns if not np.isfinite(G[u]) and W[u].notna().all()]
    trt = [u for u in W.columns if np.isfinite(G[u])]
    T = W.shape[0]
    units = []
    for u in trt:
        g = int(G[u])
        if g < 3 or W[u].isna().any():
            continue
        Y0 = W[donors].to_numpy(float)
        y1 = W[u].to_numpy(float)
        w = _synth_weights(y1[:g], Y0[:g])
        synth = Y0 @ w
        gap = y1 - synth
        pre = math.sqrt(np.mean(gap[:g] ** 2)); post = math.sqrt(np.mean(gap[g:] ** 2))
        # placebo in space: each donor in turn as the treated unit, the other donors as its pool
        ratios = []
        for j, dj in enumerate(donors):
            others = [k for k in range(len(donors)) if k != j]
            wj = _synth_weights(Y0[:g, j], Y0[:g][:, others])
            gj = Y0[:, j] - Y0[:, others] @ wj
            pr = math.sqrt(np.mean(gj[:g] ** 2)); po = math.sqrt(np.mean(gj[g:] ** 2))
            ratios.append(po / pr if pr > 0 else np.nan)
        ratio = post / pre if pre > 0 else np.nan
        rank = 1 + sum(1 for r_ in ratios if np.isfinite(r_) and r_ >= ratio)
        top = sorted([(donors[k], float(w[k])) for k in range(len(donors)) if w[k] > 0.005], key=lambda z: -z[1])[:8]
        units.append(dict(unit=u, g=g, g_label=P["labels"][g], weights=[dict(unit=k, w=_f(v, 3)) for k, v in top],
                          actual=[_f(v) for v in y1], synth=[_f(v) for v in synth], gap=[_f(v) for v in gap],
                          pre_rmspe=_f(pre), post_rmspe=_f(post), ratio=_f(ratio, 2), att=_f(float(np.mean(gap[g:]))),
                          perm_p=_f(rank / (len(ratios) + 1), 4), n_placebo=len(ratios),
                          mean_pre=_f(float(np.mean(y1[:g])))))
    return dict(units=units, donors=len(donors)) if units else None


def _its(P, G):
    d, y = P["d"], P["y"]
    if d["_u"].nunique() > 1:
        tu = [u for u in d["_u"].unique() if np.isfinite(G[u])]
        s = d[d["_u"].isin(tu)].groupby("_ti")[y].mean()
        g = int(min(G[u] for u in tu))
    else:
        s = d.set_index("_ti")[y].sort_index()
        g = int(G.iloc[0])
    T = len(s)
    if g < 8 or T - g < 4:
        return {"error": "Interrupted time series needs at least 8 periods before and 4 after the intervention."}
    yv = s.to_numpy(float)
    t = np.arange(T, dtype=float)
    D = (t >= g).astype(float)
    # seasonality: if the time labels look monthly or quarterly
    labs = P["labels"]
    tv = pd.to_numeric(pd.Series(labs), errors="coerce")
    season = None
    if tv.notna().all():
        frac = np.round((tv - np.floor(tv)) * 12).astype(int) % 12
        if frac.nunique() == 12:
            season = frac.to_numpy()
        elif np.round((tv - np.floor(tv)) * 4).astype(int).nunique() == 4 and frac.nunique() == 4:
            season = np.round((tv - np.floor(tv)) * 4).astype(int).to_numpy()
    X = [np.ones(T), t, D, (t - g) * D]
    names = ["const", "trend", "level change", "slope change"]
    if season is not None:
        for k in sorted(set(season))[1:]:
            X.append((season == k).astype(float)); names.append(f"season {k}")
    X = np.column_stack(X)
    b, e, I = _ols(yv, X)
    # Newey-West standard errors
    L = int(math.floor(4 * (T / 100) ** (2 / 9))) + 1
    def nw(X, e, I):
        S = (X * e[:, None]).T @ (X * e[:, None])
        for l in range(1, L + 1):
            G_ = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
            S += (1 - l / (L + 1)) * (G_ + G_.T)
        return I @ S @ I
    V = nw(X, e, I)
    coefs = [dict(var=n, est=_f(b[i]), se=_f(math.sqrt(max(V[i, i], 0))), p=_f(_p2(b[i] / math.sqrt(V[i, i])) if V[i, i] > 0 else None, 4)) for i, n in enumerate(names)]
    # counterfactual: pre-period model extended
    pre = X[:g][:, [0, 1] + list(range(4, X.shape[1]))]
    bp, _, _ = _ols(yv[:g], pre)
    cf = X[:, [0, 1] + list(range(4, X.shape[1]))] @ bp
    # residual autocorrelation (Durbin-Watson, Ljung-Box at lag 12 or T/4)
    dw = float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))
    # placebo interventions at earlier dates (pre-period only)
    plac = []
    for gp in (g // 2, (2 * g) // 3):
        if gp < 6 or g - gp < 4:
            continue
        tp = t[:g]; Dp = (tp >= gp).astype(float)
        Xp = [np.ones(g), tp, Dp, (tp - gp) * Dp] + ([(season[:g] == k).astype(float) for k in sorted(set(season))[1:]] if season is not None else [])
        Xp = np.column_stack(Xp)
        bb, ee, II = _ols(yv[:g], Xp)
        VV = nw(Xp, ee, II)
        plac.append(dict(at=labs[gp], est=_f(bb[2]), se=_f(math.sqrt(VV[2, 2])), p=_f(_p2(bb[2] / math.sqrt(VV[2, 2])), 4)))
    out = dict(g=g, g_label=labs[g], T=T, series=[_f(v) for v in yv], counterfactual=[_f(v) for v in cf], labels=labs,
               coefs=coefs, seasonal=season is not None, dw=_f(dw, 3), nw_lags=L, placebo=plac,
               mean_pre=_f(float(yv[:g].mean())), avg_effect=_f(float(np.mean(yv[g:] - cf[g:]))))
    return out


def _over_time(df, roles):
    P, err = _prep_panel(df, roles)
    if err:
        return {"error": err}
    d, y = P["d"], P["y"]
    G, switch_off = _adoption(d)
    n_units = int(len(G)); treated = [u for u in G.index if np.isfinite(G[u])]
    n_tr = len(treated); n_never = n_units - n_tr
    if n_tr == 0:
        return {"error": "No observation has the treatment switched on. The treatment column should be 1 for treated unit-periods and 0 otherwise."}
    if any(G[u] == 0 for u in treated) and all(G[u] == 0 for u in treated):
        return {"error": "Every treated unit is treated from the first period, so there are no pre-treatment observations to build a counterfactual from."}
    adopt = sorted(set(int(G[u]) for u in treated))
    out = {"design": "TIME", "n": int(len(d)), "N": n_units, "T": P["T"], "n_treated": n_tr, "n_never": n_never,
           "adoption": [P["labels"][g] for g in adopt], "n_cohorts": len(adopt), "switch_off": switch_off,
           "outcome": y, "first": P["labels"][0], "last": P["labels"][-1], "has_unit": bool(roles.get("unit"))}
    # chart: treated and untreated means
    m = d.assign(_grp=np.where(np.isfinite(G.reindex(d["_u"]).values), "treated", "untreated")).groupby(["_grp", "_ti"])[y].mean().unstack(0)
    out["means"] = {k: [_f(v) for v in m[k].reindex(range(P["T"])).values] for k in m.columns}
    out["labels"] = P["labels"]
    out["g0"] = adopt[0]
    if n_units == 1 or (n_tr <= FEW_TREATED and n_never < MIN_DONORS):
        out["route"] = "few_none"
        r = _its(P, G)
        if "error" in r:
            return r
        out["its"] = r
        out["model"] = "ITS"
        return out
    if n_tr <= FEW_TREATED:
        out["route"] = "few_donors"
        r = _scm(P, G)
        if r is None:
            return {"error": "Synthetic control needs a balanced panel for the treated and comparison units and at least 3 pre-treatment periods."}
        out["scm"] = r
        out["model"] = "SCM"
        return out
    staggered = len(adopt) > 1
    out["route"] = "many"
    out["timing"] = "staggered" if staggered else "common"
    r = _did(P, G, staggered)
    if "error" in r:
        return r
    out["did"] = r
    pt = r.get("pretrend")
    fail = pt is not None and pt["p"] is not None and pt["p"] < 0.05
    out["pretrend_fail"] = bool(fail)
    if fail:
        g = _gsc(P, G)
        if g is not None:
            out["gsc"] = g
        out["model"] = "GSC"
    else:
        out["model"] = "STAG" if staggered else "DID"
    return out


# ================================================================ main entry point

def analyze(csv_text, roles_json):
    warnings.simplefilter("ignore")
    roles = json.loads(roles_json) if isinstance(roles_json, str) else dict(roles_json)
    raw, err = _read(csv_text, roles)
    if err:
        return _err(err)
    try:
        if roles.get("running"):
            if roles.get("cutoff") in (None, ""):
                return _err("Enter the cutoff value of the running variable.")
            r = _rdd(raw, roles)
        elif roles.get("instruments"):
            r = _iv(raw, roles)
        else:
            r = _over_time(raw, roles)
    except Exception as e:  # report rather than crash the page
        return _err(f"The analysis failed: {e}")
    if "error" in r:
        return _err(r["error"])
    r["roles"] = roles
    return json.dumps(r)


# ---------------------------------------------------------------- public helpers for your own scripts
def rd_estimate(x, y, cutoff=0.0, h=None):
    """Local linear RD estimate with a triangular kernel; bandwidth by Imbens-Kalyanaraman if h is None."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = h or _ik_bandwidth(x, y, cutoff)
    est, se, n = _ll_fit(x, y, h, cutoff)
    return dict(est=est, se=se, h=h, n=n)


def mccrary_test(x, cutoff=0.0):
    """McCrary (2008) test for a discontinuity in the density of the running variable at the cutoff."""
    r = _mccrary(np.asarray(x, float), cutoff)
    if r:
        r.pop("hist", None)
    return r


def synth_weights(y_treated_pre, Y_donors_pre):
    """Synthetic control weights: nonnegative, summing to one, matching the pre-treatment outcome path."""
    return _synth_weights(np.asarray(y_treated_pre, float), np.asarray(Y_donors_pre, float))


def event_study(df, unit, time, y, treat):
    """Event-study estimates relative to adoption (base period -1), never-treated controls, cluster-bootstrap errors,
    a joint test of the leads -5..-2, the heterogeneity-robust ATT and the two-way fixed-effects estimate."""
    P, err = _prep_panel(df, dict(outcome=y, treat=treat, unit=unit, time=time))
    if err:
        raise ValueError(err)
    G, _ = _adoption(P["d"])
    return _did(P, G, len(set(G[np.isfinite(G.values)])) > 1)


def gsc(df, unit, time, y, treat):
    """Generalized synthetic control (interactive fixed effects, Xu 2017): number of factors chosen by
    cross-validation, ATT and event-time effects with bootstrap standard errors, pre-treatment fit."""
    P, err = _prep_panel(df, dict(outcome=y, treat=treat, unit=unit, time=time))
    if err:
        raise ValueError(err)
    G, _ = _adoption(P["d"])
    return _gsc(P, G)
