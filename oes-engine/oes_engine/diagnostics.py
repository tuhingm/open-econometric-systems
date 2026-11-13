"""Tests and tables a paper needs around a (Bayesian or regime-switching) VAR: unit roots, lag choice by marginal
likelihood, residual diagnostics, MCMC convergence, forecast-error variance and historical decompositions, prior
sensitivity, out-of-sample forecast comparison, regime classification and regime dating."""
import numpy as np
from scipy import stats

from .core import chart, f, table, test_row, tests_table


# ------------------------------------------------------------------------------------------------ before estimation
def unit_roots(Y, names):
    """ADF with constant and trend on levels, ADF with constant on first differences, KPSS on levels."""
    import warnings
    from statsmodels.tsa.stattools import adfuller, kpss
    rows = []
    for i, v in enumerate(names):
        x = Y[:, i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a0 = adfuller(x, autolag="AIC", regression="ct")
            ac = adfuller(x, autolag="AIC", regression="c")
            a1 = adfuller(np.diff(x), autolag="AIC", regression="c")
            try:
                k = kpss(x, regression="c", nlags="auto")
            except Exception:
                k = (np.nan, np.nan)
        lvl = min(a0[1], ac[1])
        order = "I(0)" if lvl < 0.05 else ("I(1)" if a1[1] < 0.05 else ("I(1) or I(2): borderline" if a1[1] < 0.10 else "I(2)?"))
        rows.append([v, f(ac[1], 4), f(a0[1], 4), f(a1[1], 4), f(k[1], 4), order])
    return table("Unit-root tests", ["Variable", "ADF level, constant: p", "ADF level, trend: p", "ADF first difference: p", "KPSS level: p", "Conclusion"], rows,
                 "ADF H0: unit root; KPSS H0: stationary (p-values are truncated at 0.01 and 0.10). For a Minnesota-type prior use a prior mean of 1 "
                 "on the own first lag for I(1) variables in levels (random walk) and 0 for stationary variables. A VAR in levels with a "
                 "random-walk prior remains valid when the variables are I(1) or cointegrated (Sims, Stock and Watson 1990).")


def lag_selection_ml(Y, pmax, lam1=0.2, delta=0.0):
    """Normal-Wishart marginal likelihood (and BIC of the OLS VAR) for lags 1..pmax on a common sample."""
    from .bayes import ar_sigma, lagmat, nw_logml, nw_prior
    n = Y.shape[1]
    rows, best = [], (None, -np.inf)
    for p in range(1, pmax + 1):
        Yp = Y[pmax - p:]
        Yt, X = lagmat(Yp, p)
        sig = ar_sigma(Y, 1)
        B0, Phi0, S0, a0 = nw_prior(n, p, sig, lam1, 1.0, 100.0, delta)
        ml = nw_logml(Yt, X, B0, Phi0, S0, a0)
        Bo = np.linalg.lstsq(X, Yt, rcond=None)[0]
        E = Yt - X @ Bo
        S = E.T @ E / len(E)
        bic = np.linalg.slogdet(S)[1] + np.log(len(E)) * Bo.size / len(E)
        rows.append([p, f(ml, 3), f(bic, 4)])
        if ml > best[1]:
            best = (p, ml)
    return table("Lag order: Bayesian marginal likelihood and BIC", ["Lags", "Log marginal likelihood (higher is better)", "BIC of the OLS VAR (lower is better)"], rows,
                 f"Common sample; Normal-Wishart prior with lambda1 = {lam1:g}. Highest marginal likelihood: {best[0]} lag(s)."), best[0]


def lambda_selection(Y, p, grid=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0), delta=0.0):
    """Hyperparameter choice by marginal likelihood (Giannone, Lenza and Primiceri 2015)."""
    from .bayes import ar_sigma, lagmat, nw_logml, nw_prior
    n = Y.shape[1]
    Yt, X = lagmat(Y, p)
    sig = ar_sigma(Y, p)
    rows, best = [], (None, -np.inf)
    for lam in grid:
        B0, Phi0, S0, a0 = nw_prior(n, p, sig, lam, 1.0, 100.0, delta)
        ml = nw_logml(Yt, X, B0, Phi0, S0, a0)
        rows.append([lam, f(ml, 3)])
        if ml > best[1]:
            best = (lam, ml)
    return table("Overall tightness lambda1 by marginal likelihood", ["lambda1", "Log marginal likelihood"], rows,
                 f"Normal-Wishart prior, {p} lag(s). The data prefer lambda1 = {best[0]:g} (Giannone, Lenza and Primiceri 2015)."), best[0]


# ------------------------------------------------------------------------------------------------ after estimation
def resid_tests(E, names, lags=None, title="Residual diagnostics"):
    """Multivariate portmanteau (Hosking) plus per-equation Ljung-Box, Jarque-Bera and ARCH-LM."""
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.stattools import jarque_bera
    E = np.asarray(E, float)
    T, n = E.shape
    h = lags or max(4, min(12, T // 10))
    C0 = E.T @ E / T
    C0i = np.linalg.inv(C0)
    Q = 0.0
    for j in range(1, h + 1):
        Cj = E[j:].T @ E[:-j] / T
        Q += np.trace(Cj.T @ C0i @ Cj @ C0i) / (T - j)
    Q *= T ** 2
    dfq = n * n * h
    rows = [test_row(f"Portmanteau (Hosking), {h} lags", Q, stats.chi2.sf(Q, dfq), f"df = {dfq}; H0: no residual autocorrelation (approximate after estimating lags)")]
    for i, v in enumerate(names):
        lb = acorr_ljungbox(E[:, i], lags=[h], return_df=True)
        rows.append(test_row(f"Ljung-Box, {v}", float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0]), "H0: no autocorrelation"))
        jb = jarque_bera(E[:, i])
        rows.append(test_row(f"Jarque-Bera, {v}", jb[0], jb[1], "H0: normal residuals"))
        try:
            ar = het_arch(E[:, i], nlags=min(4, T // 10))
            rows.append(test_row(f"ARCH LM, {v}", ar[0], ar[1], "H0: no conditional heteroskedasticity"))
        except Exception:
            pass
    return tests_table(rows, title)


def ess(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = x @ x / n
    if v <= 0:
        return float(n)
    s = 0.0
    for lag in range(1, min(n - 1, 1000)):
        r = (x[:-lag] @ x[lag:]) / n / v
        if r < 0.05:
            break
        s += r
    return n / (1 + 2 * s)


def geweke(x):
    x = np.asarray(x, float)
    a, b = x[: max(len(x) // 10, 2)], x[len(x) // 2:]
    va = np.var(a, ddof=1) / ess(a)
    vb = np.var(b, ddof=1) / ess(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb) if va + vb > 0 else 0.0


def mcmc_table(draws, names, title="MCMC convergence diagnostics", max_rows=12):
    D = np.asarray(draws, float)
    idx = list(range(D.shape[1]))
    if len(idx) > max_rows:
        sd = D.std(0)
        idx = sorted(np.argsort(-np.abs(D.mean(0)) / np.maximum(sd, 1e-12))[:max_rows].tolist())
    rows = []
    for j in idx:
        e = ess(D[:, j])
        z = geweke(D[:, j])
        rows.append([names[j], f(e, 0), f(e / len(D), 3), f(z, 3), f(2 * stats.norm.sf(abs(z)), 4)])
    worst = min(r[2] for r in rows) if rows else None
    return table(title, ["Parameter", "Effective sample size", "Efficiency (ESS / draws)", "Geweke z", "p-value"], rows,
                 "Geweke compares the first 10% with the last 50% of the chain: |z| > 1.96 suggests the chain has not settled; run more "
                 "draws or a longer burn-in. ESS above about 200 gives reliable posterior means and bands."
                 + (f" Lowest efficiency: {worst}." if worst is not None else ""))


def fevd_table(I, names, horizons=(1, 4, 8, 12, 20)):
    """I: draws x (h+1) x n x n Cholesky responses. Median shares of forecast-error variance (%)."""
    H = I.shape[1] - 1
    horizons = [h for h in horizons if h <= H + 1]
    cum = np.cumsum(I ** 2, axis=1)                                   # draws x h x resp x shock
    share = cum / cum.sum(axis=3, keepdims=True)
    med = np.median(share, axis=0) * 100
    rows = []
    for i, v in enumerate(names):
        for h in horizons:
            rows.append([v, h] + [f(med[h - 1, i, j], 1) for j in range(len(names))])
    return table("Forecast-error variance decomposition (posterior median, %)", ["Variable", "Horizon"] + [f"shock: {s}" for s in names], rows,
                 "Share of each variable's h-step forecast-error variance due to each Cholesky shock.")


def historical_decomposition(B, S, Y, p, names, labels, target=0):
    """Contribution of each structural shock (and the deterministic part) to one variable, at posterior means."""
    from .bayes import irf, lagmat
    n = Y.shape[1]
    Yt, X = lagmat(Y, p)
    E = Yt - X @ B
    P = np.linalg.cholesky(S + 1e-12 * np.eye(n))
    eps = np.linalg.solve(P, E.T).T                                    # T x n structural shocks
    T = len(Yt)
    Psi = irf(B, S, n, p, T - 1)                                      # T x n x n
    contrib = np.zeros((T, n))
    for t in range(T):
        for j in range(n):
            contrib[t, j] = (Psi[:t + 1, target, j][::-1] * eps[:t + 1, j]).sum()
    base = Yt[:, target] - contrib.sum(1)
    lab = labels[p:]
    return chart(f"Historical decomposition of {names[target]} (contributions of each shock)", lab,
                 [(f"shock: {s}", contrib[:, j]) for j, s in enumerate(names)] + [("initial conditions and constant", base)], "", names[target])


def forecast_comparison(Y, p, lam1=0.2, delta=0.0, share=0.2, hs=(1, 4), names=None):
    """Recursive pseudo out-of-sample RMSE: BVAR (Normal-Wishart posterior mean) vs OLS VAR vs random walk."""
    from .bayes import ar_sigma, lagmat, nw_posterior, nw_prior
    T, n = Y.shape
    start = max(int(T * (1 - share)), p + n * p + 10)
    errs = {m: {h: [] for h in hs} for m in ("BVAR", "OLS VAR", "Random walk")}
    for t0 in range(start, T - max(hs) + 1):
        Ytr = Y[:t0]
        Yt, X = lagmat(Ytr, p)
        sig = ar_sigma(Ytr, p)
        B0, Phi0, S0, a0 = nw_prior(n, p, sig, lam1, 1.0, 100.0, delta)
        Bb = nw_posterior(Yt, X, B0, Phi0, S0, a0)[0]
        Bo = np.linalg.lstsq(X, Yt, rcond=None)[0]
        for name, B in (("BVAR", Bb), ("OLS VAR", Bo)):
            hist = list(Ytr[-p:])
            for h in range(1, max(hs) + 1):
                x = np.r_[1.0, [v for l in range(1, p + 1) for v in hist[-l]]]
                yhat = x @ B
                hist.append(yhat)
                if h in hs:
                    errs[name][h].append(Y[t0 + h - 1] - yhat)
        for h in hs:
            errs["Random walk"][h].append(Y[t0 + h - 1] - Ytr[-1])
    rows = []
    for name, d in errs.items():
        for h in hs:
            e = np.array(d[h])
            rows.append([name, h] + [f(np.sqrt((e[:, i] ** 2).mean()), 4) for i in range(n)])
    return table(f"Out-of-sample forecast RMSE (recursive, last {int(round(share * 100))}% of the sample, {len(errs['BVAR'][hs[0]])} forecasts)",
                 ["Model", "Horizon"] + [f"RMSE {names[i]}" if names else f"RMSE {i + 1}" for i in range(n)], rows,
                 "The BVAR uses the Normal-Wishart posterior mean with the same lambda1; lower RMSE is better. Columns follow the variable order.")


def prior_sensitivity(Y, p, names, lams=(0.05, 0.1, 0.2, 0.5, 1.0), delta=0.0, shock=0, resp=-1, horizons=(0, 4, 8, 12)):
    """Response of one variable to one shock under different tightness values (Normal-Wishart posterior means)."""
    from .bayes import ar_sigma, irf, lagmat, nw_posterior, nw_prior
    n = Y.shape[1]
    Yt, X = lagmat(Y, p)
    sig = ar_sigma(Y, p)
    rows = []
    for lam in lams:
        B0, Phi0, S0, a0 = nw_prior(n, p, sig, lam, 1.0, 100.0, delta)
        Bb, Phib, Sb, ab = nw_posterior(Yt, X, B0, Phi0, S0, a0)
        I = irf(Bb, Sb / (ab - n - 1), n, p, max(horizons))
        rows.append([lam] + [f(I[h, resp, shock]) for h in horizons])
    Bo = np.linalg.lstsq(X, Yt, rcond=None)[0]
    E = Yt - X @ Bo
    I = irf(Bo, E.T @ E / (len(E) - X.shape[1]), n, p, max(horizons))
    rows.append(["OLS (no prior)"] + [f(I[h, resp, shock]) for h in horizons])
    return table(f"Prior sensitivity: response of {names[resp]} to a {names[shock]} shock", ["lambda1"] + [f"h = {h}" for h in horizons], rows,
                 "Posterior-mean responses under looser and tighter priors. Conclusions that hold across the rows are robust to the prior.")


# ------------------------------------------------------------------------------------------------ regimes
def rcm(prob):
    """Regime classification measure (Ang and Bekaert 2002): 0 = perfect classification, 100 = none."""
    P = np.asarray(prob, float)
    K = P.shape[1]
    return float(100 * K ** 2 * np.mean(np.prod(P, axis=1)))


def regime_dates(prob, labels, min_len=1):
    P = np.asarray(prob, float)
    reg = P.argmax(1)
    rows, start = [], 0
    for t in range(1, len(reg) + 1):
        if t == len(reg) or reg[t] != reg[start]:
            if t - start >= min_len:
                rows.append([f"Regime {reg[start]}", labels[start], labels[t - 1], t - start, f(P[start:t, reg[start]].mean(), 3)])
            start = t
    return table("Regime dating (most likely regime, consecutive periods)", ["Regime", "From", "To", "Periods", "Average probability"], rows)


def concordance(prob, ref, regime):
    """Share of periods where 'prob of regime > 0.5' agrees with a 0/1 reference series (e.g. NBER recessions)."""
    p = np.asarray(prob)[:, regime] > 0.5
    r = np.asarray(ref) > 0.5
    agree = float((p == r).mean())
    tp = float((p & r).sum() / max(r.sum(), 1))
    fp = float((p & ~r).sum() / max((~r).sum(), 1))
    return [["Concordance (share of periods classified alike)", f(agree, 3)], ["Reference periods detected", f(tp, 3)],
            ["False signals (share of non-reference periods)", f(fp, 3)]]


def numerical_hessian(fn, x, h=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    f0 = fn(x)
    step = h * np.maximum(1, np.abs(x))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = step[i]
            ej = np.zeros(k); ej[j] = step[j]
            H[i, j] = H[j, i] = (fn(x + ei + ej) - fn(x + ei - ej) - fn(x - ei + ej) + fn(x - ei - ej)) / (4 * step[i] * step[j])
    return H
