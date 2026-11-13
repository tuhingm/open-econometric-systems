"""Smooth-transition regression (Terasvirta 1994; LSTAR and ESTAR): the coefficients move gradually between two regimes.

y_t = x_t' phi + (x_t' theta) G(s_t; gamma, c) + e_t

Logistic (LSTAR): G = 1 / (1 + exp(-gamma (s - c) / sd(s))). The two regimes differ and the move is monotone in s.
Exponential (ESTAR): G = 1 - exp(-gamma (s - c)^2 / sd(s)^2). The outer regimes are alike and differ from the middle.

Linearity is tested with Terasvirta's third-order Taylor approximation, which also chooses between the logistic and the
exponential form. Given gamma and c the model is linear, so the sum of squares is concentrated on (gamma, c): a grid search gives
starting values and a simplex search polishes them. Standard errors use the analytic Jacobian with a Newey-West covariance.
"""
import numpy as np
from scipy import optimize, stats

from .core import (OESError, as_list, chart, check_columns, f, int_option, need, numeric, result, stat, table, test_row, time_order)
from .tar import _hac

GAMMA_LO, GAMMA_HI = 0.1, 500.0


def _G(s, gamma, c, sd, kind):
    z = (s - c) / sd
    if kind == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(gamma * z, -50, 50)))
    return 1.0 - np.exp(-np.minimum(gamma * z * z, 50.0))


def _dG(s, gamma, c, sd, kind):
    z = (s - c) / sd
    if kind == "logistic":
        G = _G(s, gamma, c, sd, kind)
        gp = G * (1 - G)
        return gp * z, -gp * gamma / sd
    E = np.exp(-np.minimum(gamma * z * z, 50.0))
    return E * z * z, -E * 2 * gamma * z / sd


def _fit_given(y, X, G):
    Z = np.hstack([X, X * G[:, None]])
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    e = y - Z @ b
    return b, float(e @ e), Z, e


def _independent(W):
    """Indices of a maximal set of linearly independent columns of W, scanning left to right."""
    keep = []
    for j in range(W.shape[1]):
        cand = keep + [j]
        if np.linalg.matrix_rank(W[:, cand], tol=1e-9 * max(np.abs(W).max(), 1)) == len(cand):
            keep.append(j)
    return keep


def _nested_F(y, A, Bm):
    """F test that the columns Bm add nothing to A."""
    T = len(y)
    if Bm.shape[1] == 0:
        return np.nan, np.nan, 0
    W = np.hstack([A, Bm])
    ea = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    ew = y - W @ np.linalg.lstsq(W, y, rcond=None)[0]
    ssa, ssw = float(ea @ ea), float(ew @ ew)
    m = Bm.shape[1]
    dfr = T - W.shape[1]
    if dfr <= 0 or ssw <= 0:
        return np.nan, np.nan, m
    F = ((ssa - ssw) / m) / (ssw / dfr)
    return F, float(stats.f.sf(F, m, dfr)), m


def _linearity(y, X, s):
    sc = (s - s.mean()) / s.std()
    blocks = [X * (sc ** k)[:, None] for k in (1, 2, 3)]
    W = np.hstack([X] + blocks)
    keep = _independent(W)
    q = X.shape[1]
    idx = np.arange(W.shape[1])
    owner = np.concatenate([np.zeros(q, int)] + [np.full(q, k) for k in (1, 2, 3)])
    keep = [j for j in keep]
    cols = {k: [j for j in keep if owner[j] == k] for k in range(4)}
    base = W[:, cols[0]]
    out = {}
    out["H0"] = _nested_F(y, base, W[:, cols[1] + cols[2] + cols[3]])
    out["H04"] = _nested_F(y, W[:, cols[0] + cols[1] + cols[2]], W[:, cols[3]])
    out["H03"] = _nested_F(y, W[:, cols[0] + cols[1]], W[:, cols[2]])
    out["H02"] = _nested_F(y, base, W[:, cols[1]])
    return out


def _grid_search(y, X, s, sd, kind, trim):
    qs = np.quantile(s, np.linspace(trim, 1 - trim, 31))
    cs = np.unique(qs)
    gs = np.geomspace(0.3, 200.0, 18)
    best = []
    for g in gs:
        for c in cs:
            _, S, _, _ = _fit_given(y, X, _G(s, g, c, sd, kind))
            best.append((S, g, c))
    best.sort(key=lambda t: t[0])
    return best[:4], (cs.min(), cs.max())


def _polish(y, X, s, sd, kind, starts, crange):
    lo_c, hi_c = crange

    def obj(v):
        g = np.exp(v[0])
        c = v[1]
        if g < GAMMA_LO or g > GAMMA_HI or c < lo_c or c > hi_c:
            return 1e300
        return _fit_given(y, X, _G(s, g, c, sd, kind))[1]
    best = None
    for S0, g0, c0 in starts:
        r = optimize.minimize(obj, [np.log(g0), c0], method="Nelder-Mead",
                              options=dict(xatol=1e-6, fatol=1e-10, maxiter=800, initial_simplex=np.array([[np.log(g0), c0], [np.log(g0) + 0.3, c0], [np.log(g0), c0 + 0.1 * sd]])))
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0])), float(best.x[1]), float(best.fun)


def smooth(df, spec):
    y = need(spec, "y", "dependent variable")
    xs = as_list(spec.get("x"))
    p = int_option(spec, "ar_lags", 1, 0, 12, "Lags of the dependent variable")
    sv = spec.get("transition") or None
    if y in xs:
        raise OESError("The dependent variable cannot also be a regressor; use the lags option for its lags.")
    if sv == y:
        sv = None
    if p == 0 and not xs:
        raise OESError("Add lags of the dependent variable or other regressors.")
    delay = int_option(spec, "delay", 1, 0 if sv else 1, 12, "Delay")
    kind_opt = spec.get("form") or "auto"
    if kind_opt not in ("auto", "logistic", "exponential"):
        raise OESError("Choose the transition function.")
    trim = float(spec.get("trim", 0.10) or 0.10)
    if not 0.05 <= trim <= 0.4:
        raise OESError("Trimming must be between 0.05 and 0.40.")
    cols = list(dict.fromkeys([y] + xs + ([sv] if sv else [])))
    tcol = spec.get("time")
    dframe = df
    if tcol:
        check_columns(df, [tcol])
        dframe = time_order(df, tcol)
    d = numeric(dframe, cols, keep=[tcol] if tcol else [])
    labels = d[tcol].astype(str).tolist() if tcol else [str(i + 1) for i in range(len(d))]
    yy = d[y].to_numpy(float)
    n = len(yy)
    start = max(p, delay)
    rows = np.arange(start, n)
    T = len(rows)
    names = ["const"] + [f"{y}(-{j})" for j in range(1, p + 1)] + xs
    X = np.column_stack([np.ones(T)] + [yy[rows - j] for j in range(1, p + 1)] + [d[c].to_numpy(float)[rows] for c in xs])
    Y = yy[rows]
    qsrc = d[sv].to_numpy(float) if sv else yy
    sname = f"{sv or y}(-{delay})" if delay else sv
    s = qsrc[rows - delay]
    lab = [labels[i] for i in rows]
    q = X.shape[1]
    if T < 40 or T < 6 * q:
        raise OESError("Too few observations for a smooth-transition model with this many regressors.")
    if np.std(s) == 0:
        raise OESError("The transition variable does not vary.")
    sd = float(np.std(s))
    # linearity tests and choice of the transition function
    lin = _linearity(Y, X, s)
    p4, p3, p2 = lin["H04"][1], lin["H03"][1], lin["H02"][1]
    ps = [p4, p3, p2]
    chosen = "exponential" if (np.isfinite(ps).all() and p3 == min(ps)) else "logistic"
    kind = chosen if kind_opt == "auto" else kind_opt
    lin_rows = [["H0: linear (all nonlinear terms zero)", f(lin["H0"][0], 3), f(lin["H0"][1], 4), lin["H0"][2], ""],
                ["H04: third-order terms zero", f(lin["H04"][0], 3), f(p4, 4), lin["H04"][2], ""],
                ["H03: second-order terms zero, given third zero", f(lin["H03"][0], 3), f(p3, 4), lin["H03"][2], ""],
                ["H02: first-order terms zero, given higher zero", f(lin["H02"][0], 3), f(p2, 4), lin["H02"][2], ""]]
    lin_note = ("Terasvirta's (1994) sequence: if the second-order test (H03) has the smallest p-value among the three, the exponential form is chosen; otherwise the logistic form. "
                f"Here the rule chooses {chosen}" + (f"; you asked for {kind_opt}." if kind_opt != 'auto' else "."))
    starts, crange = _grid_search(Y, X, s, sd, kind, trim)
    gam, c, S = _polish(Y, X, s, sd, kind, starts, crange)
    G = _G(s, gam, c, sd, kind)
    b, S, Z, e = _fit_given(Y, X, G)
    phi, theta = b[:q], b[q:]
    # Jacobian of the fitted values with respect to (phi, theta, gamma, c)
    dg, dc = _dG(s, gam, c, sd, kind)
    xth = X @ theta
    Jf = np.column_stack([Z, xth * dg, xth * dc])
    V = _hac(Jf, e)
    se = np.sqrt(np.maximum(np.diag(V), 0))
    npar = 2 * q + 2
    dfres = T - npar
    s2 = S / dfres
    # linear benchmark
    bl = np.linalg.lstsq(X, Y, rcond=None)[0]
    el = Y - X @ bl
    Sl = float(el @ el)
    r2 = 1 - S / float(((Y - Y.mean()) ** 2).sum())
    aic_l = T * np.log(Sl / T) + 2 * q
    aic_s = T * np.log(S / T) + 2 * npar
    bic_l = T * np.log(Sl / T) + q * np.log(T)
    bic_s = T * np.log(S / T) + npar * np.log(T)
    edge_g = gam >= GAMMA_HI * 0.98
    edge_c = (c - crange[0]) < 1e-4 * sd or (crange[1] - c) < 1e-4 * sd
    weak = edge_g or not np.isfinite(se[-2:]).all() or se[-2] == 0
    tdist = stats.t(dfres)
    par_rows = [["gamma (smoothness, per s.d. of the transition variable)", f(gam, 4), f(se[-2], 4), f(gam / se[-2], 2) if se[-2] > 0 else None],
                [f"c (location, in units of {sname})", f(c, 4), f(se[-1], 4), f(((c) / se[-1]), 2) if se[-1] > 0 else None]]
    par_tab = table(f"Transition function ({kind})", ["Parameter", "Estimate", "Std. error", "t"], par_rows,
                    "Standard errors are robust (Newey-West) and rely on a Gauss-Newton approximation. A large gamma makes the transition abrupt and its standard error unreliable, because the likelihood is almost flat in gamma; "
                    "the location c is then the better-identified quantity." + (" Here gamma is at its upper limit, so the model is close to a threshold model." if edge_g else ""))
    coef_rows = []
    for k, nm in enumerate(names):
        tp = phi[k] / se[k] if se[k] > 0 else None
        tt = theta[k] / se[q + k] if se[q + k] > 0 else None
        coef_rows.append([nm, f"{f(phi[k], 4)} ({f(se[k], 4)})", f(2 * tdist.sf(abs(tp)), 4) if tp is not None else None,
                          f"{f(theta[k], 4)} ({f(se[q + k], 4)})", f(2 * tdist.sf(abs(tt)), 4) if tt is not None else None])
    coef_tab = table("Coefficients", ["Variable", "Linear part (G = 0)", "p-value", "Nonlinear part (G = 1 adds)", "p-value"], coef_rows,
                     "The coefficient at transition value G is the linear part plus G times the nonlinear part. Robust standard errors in parentheses.")
    # implied coefficients at G=0 and G=1
    imp_rows = []
    for k, nm in enumerate(names):
        v = V[np.ix_([k, q + k], [k, q + k])]
        s1 = np.sqrt(max(v[0, 0] + v[1, 1] + 2 * v[0, 1], 0))
        imp_rows.append([nm, f"{f(phi[k], 4)} ({f(se[k], 4)})", f"{f(phi[k] + theta[k], 4)} ({f(s1, 4)})"])
    if kind == "logistic":
        g0, g1 = f"low {sname}", f"high {sname}"
    else:
        g0, g1 = f"{sname} near c (middle)", f"{sname} far from c (outer)"
    imp_tab = table("Coefficients in the two extreme regimes", ["Variable", f"G = 0 ({g0})", f"G = 1 ({g1})"], imp_rows,
                    "G = 0 and G = 1 are the limits of the transition function; most observations lie between them.")
    # residual autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(e, lags=[4], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    lbl = float(lb["lb_stat"].iloc[0])
    fit_tab = table("Fit against the linear model", ["Model", "SSR", "Residual std. dev.", "AIC", "BIC"],
                    [["Linear", f(Sl, 4), f(np.sqrt(Sl / (T - q)), 4), f(aic_l, 3), f(bic_l, 3)],
                     [f"Smooth transition ({kind})", f(S, 4), f(np.sqrt(s2), 4), f(aic_s, 3), f(bic_s, 3)]],
                    "AIC and BIC count gamma and c as parameters. Lower is better.")
    diag_tab = table("Residual diagnostics", ["Test", "Statistic", "p-value", "Detail"],
                     [test_row("Ljung-Box serial correlation (4 lags)", lbl, lb_p, "H0: no serial correlation")], None)
    order = np.argsort(s)
    charts = [chart(f"Transition function against {sname}", [float(v) for v in s[order]], [("G", G[order])], sname, "G"),
              chart(f"Transition function over time ({kind})", lab, [("G", G)], "", "G")]
    summary = [stat("Observations", T), stat("Transition variable", sname), stat("Transition function", kind),
               stat("Linearity F (all terms)", lin["H0"][0], 3), stat("Linearity p", lin["H0"][1], 4),
               stat("gamma", gam, 3), stat("c", c, 4), stat("c at the edge of its range", "yes" if edge_c else "no"), stat("R-squared", r2, 4), stat("AIC gain over linear", aic_l - aic_s, 2)]
    notes = ["All coefficients, including the constant, are allowed to move with the transition function.",
             "The linearity tests are the F versions of Terasvirta's LM tests and assume homoskedastic errors. If they reject, the smooth-transition model is worth estimating; if they do not, the linear model is the safer choice.",
             "The location c is searched between the " + f"{int(trim * 100)}th and {int((1 - trim) * 100)}th percentiles of the transition variable, and gamma between {GAMMA_LO} and {GAMMA_HI:g}."]
    if edge_c:
        notes.append("The location c is at the edge of its search range, so the range may be excluding the best location. Widen the range and compare.")
    if weak:
        notes.append("The transition is nearly abrupt, so gamma is imprecisely estimated. Compare with the threshold regression before settling on a smooth form.")
    if lin["H0"][1] is not None and np.isfinite(lin["H0"][1]) and lin["H0"][1] > 0.10:
        notes.append("Linearity is not rejected at 10%, so the nonlinear estimates are not well determined.")
    tabs = [table("Linearity tests (Terasvirta)", ["Test", "F", "p-value", "Restrictions", "Detail"], lin_rows, lin_note), par_tab, coef_tab, imp_tab, fit_tab, diag_tab]
    return result("str", f"Smooth-transition regression: {y}", T, summary, tabs, charts, notes)
