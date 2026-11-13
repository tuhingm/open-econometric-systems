"""Bayesian VARs estimated with state-space Gibbs samplers: time-varying parameters (TVP-VAR), stochastic
volatility (SV-VAR) and mixed frequencies (MF-BVAR).

TVP-VAR: coefficients follow random walks (Cogley and Sargent 2002; Primiceri 2005 priors from a training sample),
constant residual covariance; Carter-Kohn simulation smoother.
SV-VAR: constant coefficients, Sigma_t = A^-1 H_t A^-1' with log-volatilities following random walks (Cogley and
Sargent 2005); Kim, Shephard and Chib (1998) mixture for the log-volatilities.
MF-BVAR: a monthly VAR in which quarterly variables are latent monthly series observed as quarterly averages (or as
Mariano-Murasawa triangle-weighted sums of monthly growth rates); Schorfheide and Song (2015) style Gibbs sampler.
"""
import numpy as np

from .bayes import (_draw_settings, _floats, _hyper, ar_sigma, companion, forecast_charts, irf, irf_charts, iw, lagmat,
                    minnesota, nw_draws, nw_posterior, nw_prior, post_table, reg_names, simulate_forecast)
from .core import OESError, as_list, chart, check_columns, distinct, f, float_option, int_option, need, numeric, result, stat, table, time_order
from .timeseries import _series_frame


def _mvn(m, V, rng):
    V = (V + V.T) / 2
    try:
        L = np.linalg.cholesky(V + 1e-10 * np.eye(len(m)))
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(V)
        L = U * np.sqrt(np.clip(w, 0, None))
    return m + L @ rng.standard_normal(len(m))


def _band_chart(title, x, draws, ylabel=""):
    med, lo, hi = (np.percentile(draws, q, axis=0) for q in (50, 16, 84))
    return chart(title, x, [("median", med)], "", ylabel, band=(lo, hi))


def _names(spec):
    names = as_list(need(spec, "vars", "variables (at least two)"))
    if len(names) < 2:
        raise OESError("A VAR needs at least two variables.")
    distinct(names)
    return names


# ================================================================================================ TVP-VAR
def tvpvar(df, spec):
    names = _names(spec)
    d, labels = _series_frame(df, spec, names)
    Yall = d[names].to_numpy(float)
    n = len(names)
    p = int_option(spec, "lags", 2, 1, 4, "Lags")
    tau = int_option(spec, "training", 40, 10, 200, "Training-sample length")
    kQ = float_option(spec, "kQ", 0.01, "kQ")
    ndraw, burn, rng = _draw_settings(spec, 1000, 1000)
    if len(Yall) < tau + p + 20:
        raise OESError("The sample is too short for this training sample: lower the training-sample length.")
    # training sample: OLS mean and covariance of the coefficients
    Yt0, X0 = lagmat(Yall[:tau + p], p)
    B_ols = np.linalg.lstsq(X0, Yt0, rcond=None)[0]
    E0 = Yt0 - X0 @ B_ols
    S_ols = E0.T @ E0 / max(len(E0) - X0.shape[1], 1)
    V_ols = np.kron(S_ols, np.linalg.pinv(X0.T @ X0))
    Yt, X = lagmat(Yall[tau:], p)
    lab = labels[tau + p:]
    T, k = X.shape
    m = n * k
    b0, P0 = B_ols.flatten("F"), 4 * V_ols + 1e-8 * np.eye(m)
    Q0, nuQ = kQ ** 2 * tau * V_ols + 1e-10 * np.eye(m), max(tau, m + 2)
    Z = np.array([np.kron(np.eye(n), x[None, :]) for x in X])                     # T x n x m
    Q, S = Q0 / nuQ, S_ols.copy()
    stab_tries = int_option(spec, "stability_tries", 0, 0, 50, "Stability retries") or 1
    check_stab = int_option(spec, "stability_tries", 0, 0, 50, "Stability retries") > 0
    accepted, rejected = 0, 0
    keep_b = np.empty((ndraw, T, m))
    keep_S = np.empty((ndraw, n, n))
    for it in range(burn + ndraw):
        # 1. coefficient paths (Carter-Kohn)
        bf, Pf = np.empty((T, m)), np.empty((T, m, m))
        b, P = b0, P0
        for t in range(T):
            P = P + Q
            Zt = Z[t]
            F = Zt @ P @ Zt.T + S
            K = np.linalg.solve(F, Zt @ P).T
            b = b + K @ (Yt[t] - Zt @ b)
            P = P - K @ Zt @ P
            bf[t], Pf[t] = b, P
        # Carter-Kohn backward sampling; paths with an explosive VAR at some date are rejected (Cogley and Sargent 2005)
        for attempt in range(stab_tries):
            Bn = np.empty((T, m))
            Bn[-1] = _mvn(bf[-1], Pf[-1], rng)
            for t in range(T - 2, -1, -1):
                G = np.linalg.solve(Pf[t] + Q, Pf[t]).T
                Bn[t] = _mvn(bf[t] + G @ (Bn[t + 1] - bf[t]), Pf[t] - G @ Pf[t], rng)
            if not check_stab or all(np.abs(np.linalg.eigvals(companion(Bn[t].reshape((k, n), order="F"), n, p))).max() < 1 for t in range(0, T, 4)):
                B = Bn
                accepted += 1
                break
        else:
            rejected += 1
            if it == 0:
                B = Bn
        # 2. state innovation covariance and 3. residual covariance
        dB = np.diff(np.vstack([b0, B]), axis=0)
        Q = np.atleast_2d(iw(Q0 + dB.T @ dB, nuQ + T, rng))
        E = Yt - np.einsum("tnm,tm->tn", Z, B)
        S = np.atleast_2d(iw(np.eye(n) + E.T @ E, n + 1 + T, rng))
        if it >= burn:
            keep_b[it - burn], keep_S[it - burn] = B, S
    rn = reg_names(names, p)
    h = int_option(spec, "horizon", 20, 1, 60, "IRF horizon")
    charts = []
    for i, v in enumerate(names):
        charts.append(_band_chart(f"Time-varying constant: {v} equation", lab, keep_b[:, :, i * k], "coefficient"))
        charts.append(_band_chart(f"Time-varying own first lag: {v} equation", lab, keep_b[:, :, i * k + 1 + i], "coefficient"))
    dates = sorted({0, T // 2, T - 1})
    step = max(1, ndraw // 200)
    for i in range(n):
        series = []
        for t in dates:
            I = np.array([irf(keep_b[q, t].reshape((k, n), order="F"), keep_S[q], n, p, h) for q in range(0, ndraw, step)])
            series.append((str(lab[t]), np.median(I[:, :, i, n - 1], 0)))
        charts.append(chart(f"Response of {names[i]} to a shock in {names[-1]}, at three dates", list(range(h + 1)), series, "horizon", names[i]))
    first, last = keep_b[:, 0], keep_b[:, -1]
    rows = []
    for i, v in enumerate(names):
        for r, nm in enumerate(rn):
            j = i * k + r
            ch = last[:, j] - first[:, j]
            rows.append([v, nm, f(first[:, j].mean()), f(last[:, j].mean()), f(ch.mean()), f((ch > 0).mean(), 3)])
    from . import diagnostics as D
    Bmed = keep_b.mean(0)
    E = Yt - np.einsum("tnm,tm->tn", Z, Bmed)
    sel = [i * k + r for i in range(n) for r in (0, 1 + i)]
    tabs = [D.unit_roots(Yall, names),
            table(f"Coefficients at the start ({lab[0]}) and the end ({lab[-1]}) of the sample", ["Equation", "Regressor", "Start (mean)", "End (mean)", "Change", "P(change > 0)"], rows,
                  "If P(change > 0) is near 0 or 1, the coefficient has moved over the sample: evidence of time variation."),
            post_table("Residual covariance (posterior)", [f"Sigma[{a},{b}]" for a in names for b in names], keep_S.reshape(ndraw, -1)),
            D.mcmc_table(np.column_stack([keep_b[:, -1, sel], keep_S.reshape(ndraw, -1)]),
                         [f"{names[j // k]}: {rn[j % k]} (last date)" for j in sel] + [f"Sigma[{a},{b}]" for a in names for b in names]),
            D.resid_tests(E, names, title="Residual diagnostics (posterior-mean coefficient paths)")]
    summary = [stat("Observations", T), stat("Training sample", tau), stat("Lags", p), stat("kQ", kQ), stat("Draws", ndraw),
               stat("Iterations with no stable path", rejected)]
    notes = [f"Random-walk coefficients with a training-sample prior (Primiceri 2005): b_0 ~ N(OLS, 4 V_OLS), Q ~ IW(kQ^2 tau V_OLS, tau); "
             f"the first {tau + p} observations are used only for the prior. "
             + ("Coefficient paths that imply an explosive VAR at any date are rejected (Cogley and Sargent 2005). " if check_stab else "")
             + f"Bands are 68% credible intervals; {ndraw} draws after {burn}.",
             "Impulse responses at a date use that date's coefficients (Cholesky ordering), as is usual for TVP-VARs."]
    return result("tvpvar", "Time-varying parameter VAR", T, summary, tabs, charts, notes)


# ================================================================================================ SV-VAR
KSC_Q = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC_M = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704
KSC_V = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])


def _ck_rw_univariate(y, R, s, h0, P0, rng):
    """Carter-Kohn for h_t = h_{t-1} + v_t (var s), y_t = h_t + mean_t + noise(var R_t), with y already demeaned."""
    T = len(y)
    hf, Pf = np.empty(T), np.empty(T)
    hh, P = h0, P0
    for t in range(T):
        P = P + s
        K = P / (P + R[t])
        hh = hh + K * (y[t] - hh)
        P = (1 - K) * P
        hf[t], Pf[t] = hh, P
    out = np.empty(T)
    out[-1] = hf[-1] + np.sqrt(Pf[-1]) * rng.standard_normal()
    for t in range(T - 2, -1, -1):
        G = Pf[t] / (Pf[t] + s)
        out[t] = hf[t] + G * (out[t + 1] - hf[t]) + np.sqrt(max(Pf[t] - G * Pf[t], 1e-12)) * rng.standard_normal()
    return out


def svvar(df, spec):
    names = _names(spec)
    d, labels = _series_frame(df, spec, names)
    Y = d[names].to_numpy(float)
    n = len(names)
    p = int_option(spec, "lags", 2, 1, 8, "Lags")
    hyper = _hyper(spec)
    delta = _floats(spec, "delta", n, "Prior mean of the own first lag")
    delta = 0.0 if delta is None else delta
    ndraw, burn, rng = _draw_settings(spec, 1000, 500)
    sig = ar_sigma(Y, p)
    Yt, X = lagmat(Y, p)
    T, k = X.shape
    lab = labels[p:]
    B0, V0 = minnesota(n, p, sig, delta=delta, **hyper)
    b0, Oi = B0.flatten("F"), np.diag(1 / V0.flatten("F"))
    B = np.linalg.lstsq(X, Yt, rcond=None)[0]
    A = np.eye(n)
    hmat = np.tile(np.log(sig ** 2), (T, 1))
    s = np.full(n, 0.01)
    keepB, keepH, keepA, keeps = np.empty((ndraw, k, n)), np.empty((ndraw, T, n)), np.empty((ndraw, n, n)), np.empty((ndraw, n))
    for it in range(burn + ndraw):
        # 1. coefficients given Sigma_t = A^-1 H_t A^-1'  (GLS posterior)
        Hinv = np.exp(-hmat)                                            # T x n
        Sinv = np.einsum("ji,tj,jk->tik", A, Hinv, A)                   # A' H_t^-1 A
        P = Oi.copy()
        rhs = Oi @ b0
        for i in range(n):
            for j in range(n):
                w = Sinv[:, i, j]
                P[i * k:(i + 1) * k, j * k:(j + 1) * k] += (X * w[:, None]).T @ X
            rhs[i * k:(i + 1) * k] += X.T @ np.einsum("tj,tj->t", Sinv[:, i, :], Yt)
        P = (P + P.T) / 2
        L = np.linalg.cholesky(P)
        bvec = np.linalg.solve(P, rhs) + np.linalg.solve(L.T, rng.standard_normal(k * n))
        B = bvec.reshape((k, n), order="F")
        U = Yt - X @ B
        # 2. contemporaneous relations A (lower triangular, unit diagonal)
        for i in range(1, n):
            w = np.exp(-hmat[:, i])
            Xi = -U[:, :i]
            Pi = np.eye(i) / 10 + (Xi * w[:, None]).T @ Xi
            mi = np.linalg.solve(Pi, (Xi * w[:, None]).T @ U[:, i])
            A[i, :i] = _mvn(mi, np.linalg.inv(Pi), rng)
        Eps = U @ A.T
        # 3. log-volatilities (KSC mixture) and 4. their innovation variances
        for i in range(n):
            ystar = np.log(Eps[:, i] ** 2 + 1e-6)
            dev = ystar[:, None] - hmat[:, [i]] - KSC_M[None, :]
            lp = np.log(KSC_Q) - 0.5 * np.log(KSC_V) - 0.5 * dev ** 2 / KSC_V
            pr = np.exp(lp - lp.max(1, keepdims=True))
            pr /= pr.sum(1, keepdims=True)
            comp = (pr.cumsum(1) > rng.random(T)[:, None]).argmax(1)
            hmat[:, i] = _ck_rw_univariate(ystar - KSC_M[comp], KSC_V[comp], s[i], np.log(sig[i] ** 2), 4.0, rng)
            dh = np.diff(np.r_[np.log(sig[i] ** 2), hmat[:, i]])
            s[i] = 1 / rng.gamma(1 + T / 2, 1 / (0.01 + dh @ dh / 2))
        if it >= burn:
            j = it - burn
            keepB[j], keepH[j], keepA[j], keeps[j] = B, hmat, A, s
    charts = []
    for i, v in enumerate(names):
        charts.append(_band_chart(f"Volatility of the {v} shock (standard deviation)", lab, np.exp(keepH[:, :, i] / 2), "s.d."))
    h = int_option(spec, "horizon", 20, 1, 60, "IRF horizon")
    step = max(1, ndraw // 300)

    def sig_t(q, t):
        Ai = np.linalg.inv(keepA[q])
        return Ai @ np.diag(np.exp(keepH[q, t])) @ Ai.T

    dates = sorted({int(np.argmin(np.median(keepH[:, :, -1], 0))), int(np.argmax(np.median(keepH[:, :, -1], 0))), T - 1})
    for i in range(n):
        series = []
        for t in dates:
            I = np.array([irf(keepB[q], sig_t(q, t), n, p, h) for q in range(0, ndraw, step)])
            series.append((str(lab[t]), np.median(I[:, :, i, n - 1], 0)))
        charts.append(chart(f"Response of {names[i]} to a shock in {names[-1]} (calm, volatile and last dates)", list(range(h + 1)), series, "horizon", names[i]))
    tabs = []
    rn = reg_names(names, p)
    for i, v in enumerate(names):
        tabs.append(post_table(f"Equation: {v}", rn, keepB[:, :, i]))
    tabs.append(post_table("Variance of the log-volatility innovations", [f"s[{v}]" for v in names], keeps))
    from . import diagnostics as D
    tabs.insert(0, D.unit_roots(Y, names))
    tabs.append(D.mcmc_table(np.column_stack([keepB.reshape(ndraw, -1), keeps]), [f"{names[i]}: {r}" for r in rn for i in range(n)] + [f"s[{v}]" for v in names]))
    Bm = keepB.mean(0)
    Am = keepA.mean(0)
    Hm = np.median(keepH, 0)
    std_shocks = ((Yt - X @ Bm) @ Am.T) / np.exp(Hm / 2)
    tabs.append(D.resid_tests(std_shocks, [f"shock {v}" for v in names], title="Diagnostics of the standardised structural shocks"))
    tabs[-1]["note"] = "Shocks divided by their estimated volatility should be white noise and close to normal if the volatility model fits; compare with a constant-volatility VAR, where ARCH tests usually reject."
    vol = np.exp(keepH / 2)
    tabs.append(table("Shock volatility over the sample (posterior median s.d.)", ["Variable", "Minimum", "Date", "Maximum", "Date", "Last"],
                      [[v, f(np.median(vol[:, :, i], 0).min()), str(lab[int(np.argmin(np.median(vol[:, :, i], 0)))]),
                        f(np.median(vol[:, :, i], 0).max()), str(lab[int(np.argmax(np.median(vol[:, :, i], 0)))]), f(np.median(vol[:, -1, i]))]
                       for i, v in enumerate(names)]))
    summary = [stat("Observations", T), stat("Lags", p), stat("Draws", ndraw)] + [stat(f"Last volatility: {v}", float(np.median(vol[:, -1, i]))) for i, v in enumerate(names)]
    notes = ["Stochastic-volatility VAR (Cogley and Sargent 2005): constant coefficients with a Minnesota prior; Sigma_t = A^-1 H_t A^-1', "
             "log-volatilities follow random walks, sampled with the Kim-Shephard-Chib mixture. Shocks are the recursive (Cholesky) shocks; "
             f"bands are 68% credible intervals; {ndraw} draws after {burn}."]
    return result("svvar", "Stochastic-volatility VAR", T, summary, tabs, charts, notes)


# ================================================================================================ mixed-frequency BVAR
def mfbvar(df, spec):
    mvars = as_list(need(spec, "monthly", "monthly variables"))
    qvars = as_list(need(spec, "quarterly", "quarterly variables"))
    distinct(mvars, qvars)
    tcol = spec.get("time")
    if tcol:
        check_columns(df, [tcol])
        df = time_order(df, tcol)
    check_columns(df, mvars + qvars)
    d = df.copy()
    for c in mvars + qvars:
        d[c] = __import__("pandas").to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=mvars, how="any").reset_index(drop=True)
    labels = d[tcol].astype(str).tolist() if tcol else [str(i + 1) for i in range(len(d))]
    names = mvars + qvars
    nm, nq = len(mvars), len(qvars)
    n = nm + nq
    Yobs = d[names].to_numpy(float)
    T = len(Yobs)
    if T < 60:
        raise OESError("Mixed-frequency estimation needs at least 60 months.")
    agg = spec.get("aggregation", "triangle")
    w = np.array([1, 2, 3, 2, 1]) / 9 if agg == "triangle" else np.array([1, 1, 1]) / 3
    L = len(w)
    p = int_option(spec, "lags", 3, 1, 12, "Lags (months)")
    r = max(p, L)
    lam1 = float_option(spec, "lambda1", 0.2, "Overall tightness (lambda1)")
    delta = _floats(spec, "delta", n, "Prior mean of the own first lag")
    delta = 0.0 if delta is None else delta
    ndraw, burn, rng = _draw_settings(spec, 300, 200)
    for j in range(nq):
        if np.isfinite(Yobs[:, nm + j]).sum() < 12:
            raise OESError(f"The quarterly variable {qvars[j]} has fewer than 12 observations.")
    # initial latent monthly series: quarterly values spread over the quarter, gaps interpolated
    Y = Yobs.copy()
    for j in range(nq):
        col = Y[:, nm + j]
        idx = np.where(np.isfinite(col))[0]
        Y[:, nm + j] = np.interp(np.arange(T), idx, col[idx])
    var = np.nanvar(Yobs, axis=0)
    Hn = 1e-4 * np.maximum(var, 1e-8)
    S = np.diag(np.nanvar(np.diff(Y, axis=0), axis=0))
    keepY, keepB, keepS = np.empty((ndraw, T, nq)), [], []
    ms = n * r
    for it in range(burn + ndraw):
        # VAR parameters given the monthly data (Normal-Wishart, Minnesota scale)
        sig = ar_sigma(Y, p)
        B0, Phi0, S0, a0 = nw_prior(n, p, sig, lam1, 1.0, 100.0, delta)
        Yt, X = lagmat(Y, p)
        post = nw_posterior(Yt, X, B0, Phi0, S0, a0)
        for _ in range(20):
            Bd, Sd = nw_draws(*post, 1, rng)
            if np.abs(np.linalg.eigvals(companion(Bd[0], n, p))).max() < 1:
                break
        B, S = Bd[0], Sd[0]
        # state space: s_t = [y_t, ..., y_{t-r+1}]
        F = np.zeros((ms, ms))
        F[:n, :n * p] = B[1:].T
        if r > 1:
            F[n:, :-n] = np.eye(n * (r - 1))
        c = np.zeros(ms)
        c[:n] = B[0]
        Qs = S
        s0 = np.concatenate([Y[r - 1 - l] for l in range(r)])
        P = 10 * np.eye(ms)
        sf, Pf = np.empty((T, ms)), np.empty((T, ms, ms))
        st = s0
        for t in range(T):
            if t >= r:
                st = c + F @ st
                P = F @ P @ F.T
                P[:n, :n] += Qs
            rows, obs, noise = [], [], []
            for i in range(nm):
                if np.isfinite(Yobs[t, i]):
                    z = np.zeros(ms); z[i] = 1
                    rows.append(z); obs.append(Yobs[t, i]); noise.append(Hn[i])
            for j in range(nq):
                if np.isfinite(Yobs[t, nm + j]) and t >= L - 1:
                    z = np.zeros(ms)
                    for l in range(L):
                        z[l * n + nm + j] = w[l]
                    rows.append(z); obs.append(Yobs[t, nm + j]); noise.append(Hn[nm + j])
            if rows and t >= r - 1:
                Zt = np.array(rows)
                Fv = Zt @ P @ Zt.T + np.diag(noise)
                K = np.linalg.solve(Fv, Zt @ P).T
                st = st + K @ (np.array(obs) - Zt @ st)
                P = P - K @ Zt @ P
            sf[t], Pf[t] = st, (P + P.T) / 2
        draw = np.empty((T, ms))
        draw[-1] = _mvn(sf[-1], Pf[-1], rng)
        # condition on the whole next state: its lag blocks pin down this state's first blocks, so the quarterly
        # aggregation constraints observed later carry back (a tiny variance stands in for the exact identities)
        Qf = np.full(ms, 1e-9 * float(np.mean(var)))
        Qf = np.diag(Qf)
        Qf[:n, :n] += Qs
        for t in range(T - 2, r - 2, -1):
            if t < r - 1:
                break
            G = np.linalg.solve(F @ Pf[t] @ F.T + Qf, F @ Pf[t]).T
            m_ = sf[t] + G @ (draw[t + 1] - c - F @ sf[t])
            draw[t] = _mvn(m_, Pf[t] - G @ F @ Pf[t], rng)
        for t in range(r - 1):
            draw[t] = sf[t]
        Ynew = draw[:, :n].copy()
        Ynew[:r - 1] = Y[:r - 1]
        Ynew[:, :nm] = np.where(np.isfinite(Yobs[:, :nm]), Yobs[:, :nm], Ynew[:, :nm])
        Y = Ynew
        if it >= burn:
            keepY[it - burn] = Y[:, nm:]
            keepB.append(B); keepS.append(S)
    keepB, keepS = np.array(keepB), np.array(keepS)
    charts, rows = [], []
    back = min(T, 120)
    for j, v in enumerate(qvars):
        D = keepY[:, -back:, j]
        med, lo, hi = (np.percentile(D, q, axis=0) for q in (50, 16, 84))
        obs = [x if np.isfinite(x) else None for x in Yobs[-back:, nm + j]]
        charts.append(chart(f"Monthly estimate of {v} (median, 68% band) and quarterly data", labels[-back:], [("monthly estimate", med), ("published quarterly", obs)], "", v, band=(lo, hi)))
        last = int(np.where(np.isfinite(Yobs[:, nm + j]))[0].max())
        for t in range(last + 1, T):
            agg_now = np.array([sum(w[l] * keepY[q, t - l, j] for l in range(L)) for q in range(ndraw)])
            rows.append([v, labels[t], f(np.median(keepY[:, t, j])), f(np.percentile(keepY[:, t, j], 16)), f(np.percentile(keepY[:, t, j], 84)),
                         f(np.median(agg_now))])
    tabs = []
    if rows:
        tabs.append(table("Nowcasts: months after the last quarterly release", ["Variable", "Month", "Monthly (median)", "16%", "84%", "Implied quarterly value"], rows,
                          "The implied quarterly value aggregates the latent monthly path with the chosen weights; at the quarter's last month it is the nowcast of the release."))
    rn = reg_names(names, p)
    from . import diagnostics as D
    ur = []
    for i, v in enumerate(names):
        x = Yobs[:, i][np.isfinite(Yobs[:, i])]
        ur.append(x)
    from statsmodels.tsa.stattools import adfuller
    tabs.insert(0, table("Unit-root tests (observed data)", ["Variable", "Frequency", "Observations", "ADF p-value (level)", "ADF p-value (difference)"],
                         [[v, "quarterly" if i >= nm else "monthly", len(ur[i]), f(adfuller(ur[i], autolag="AIC")[1], 4), f(adfuller(np.diff(ur[i]), autolag="AIC")[1], 4)]
                          for i, v in enumerate(names)],
                         "Use growth rates (triangle aggregation) for I(1) variables such as GDP, or levels with the 3-month average."))
    for i, v in enumerate(names):
        tabs.append(post_table(f"Monthly VAR equation: {v}", rn, keepB[:, :, i]))
    tabs.append(D.mcmc_table(np.column_stack([keepB.reshape(ndraw, -1), keepY[:, -1, :]]), [f"{names[i]}: {r}" for r in rn for i in range(n)] + [f"last month of {v}" for v in qvars]))
    Bm = keepB.mean(0)
    Ylat = Yobs.copy()
    Ylat[:, nm:] = np.median(keepY, 0)
    Yl, Xl = lagmat(Ylat, p)
    tabs.append(D.resid_tests(Yl - Xl @ Bm, names, title="Residual diagnostics of the monthly VAR (posterior means)"))
    h = int_option(spec, "horizon", 24, 1, 60, "IRF horizon")
    I = np.array([irf(keepB[q], keepS[q], n, p, h) for q in range(ndraw)])
    charts += irf_charts(I, names, h, shocks=[n - 1] if n > 2 else None, max_charts=6)
    summary = [stat("Months", T), stat("Monthly variables", nm), stat("Quarterly variables", nq), stat("Aggregation", "triangle (growth rates)" if agg == "triangle" else "average (levels)"),
               stat("Lags", p), stat("Draws", ndraw)]
    notes = ["Mixed-frequency BVAR: the VAR runs at the monthly frequency; each quarterly variable is a latent monthly series whose "
             + ("triangle-weighted sum (1,2,3,2,1)/9 of monthly growth rates" if agg == "triangle" else "three-month average")
             + " equals the quarterly figure in the quarter's last month. Put quarterly values in the row of the quarter's last month and "
             "leave the other months blank. Normal-Wishart Minnesota-type prior; Gibbs sampler with Carter-Kohn draws of the latent months; "
             f"{ndraw} draws after {burn}."]
    return result("mfbvar", "Mixed-frequency BVAR", T, summary, tabs, charts, notes)
