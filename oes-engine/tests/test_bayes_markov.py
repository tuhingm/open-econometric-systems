"""Tests of the Bayesian and Markov-switching models: limits that must reproduce OLS, published estimates
(Hamilton 1989; Stata's mswitch manual), cross-checks with statsmodels and recovery of known parameters."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import oes_engine as E
from oes_engine import bayes as B, markov as M

from test_models import frame, rows


def _sim_var(T=400, A=((0.5, 0.1), (0.0, 0.6)), c=(1.0, -0.5), seed=0, S=None):
    rng = np.random.default_rng(seed)
    A, c = np.array(A), np.array(c)
    n = len(c)
    L = np.linalg.cholesky(np.eye(n) if S is None else np.array(S))
    y = np.zeros((T + 100, n))
    for t in range(1, T + 100):
        y[t] = c + A @ y[t - 1] + L @ rng.standard_normal(n)
    return pd.DataFrame(y[100:], columns=[f"y{i}" for i in range(n)])


# ------------------------------------------------------------------------------------------------ Bayesian regression
@pytest.mark.parametrize("method", ["conjugate", "gibbs"])
def test_bayes_reg_with_a_diffuse_prior_is_ols(method):
    d, s = frame("bayes_reg")
    spec = dict(s, method=method, prior="normal", prior_sd=1e4, draws=20000, burn=500)
    r = E.run("bayes_reg", d, spec)
    ols = sm.OLS(d[s["y"]], sm.add_constant(d[s["x"]])).fit()
    t = rows(r["tables"][0])
    for v in s["x"]:
        assert t[v][1] == pytest.approx(ols.params[v], abs=3 * ols.bse[v] / np.sqrt(20000) + 1e-4 * abs(ols.params[v]) + 1e-6)
        assert t[v][2] == pytest.approx(ols.bse[v], rel=0.03)


def test_bayes_reg_g_prior_shrinks_by_g_over_one_plus_g():
    d, s = frame("bayes_reg")
    r = E.run("bayes_reg", d, dict(s, g=9))
    ols = sm.OLS(d[s["y"]], sm.add_constant(d[s["x"]])).fit()
    t = rows(r["tables"][0])
    for v in s["x"]:
        assert t[v][1] == pytest.approx(0.9 * ols.params[v], rel=1e-4)


# ------------------------------------------------------------------------------------------------ BVAR
def test_nw_bvar_with_a_loose_prior_is_the_ols_var():
    from statsmodels.tsa.api import VAR
    d, s = frame("bvar")
    r = E.run("bvar", d, dict(s, prior="nw", lambda1=1e4, lambda4=1e6, draws=200))
    ref = VAR(d[s["vars"]].reset_index(drop=True)).fit(4)
    eq = rows(r["tables"][0])
    for name, val in ref.params[s["vars"][0]].items():
        assert eq[name][1] == pytest.approx(val, abs=0.02 * (1 + abs(val)))


def test_minnesota_with_a_tight_prior_is_the_prior():
    d, s = frame("bvar")
    r = E.run("bvar", d, dict(s, prior="minnesota", lambda1=1e-6, draws=200))
    eq = rows(r["tables"][0])
    assert eq["L1.log_price"][1] == pytest.approx(1.0, abs=1e-4)
    assert eq["L1.log_m2"][1] == pytest.approx(0.0, abs=1e-4)


def test_minnesota_posterior_mean_is_the_analytic_formula():
    d, s = frame("bvar")
    Y = d[s["vars"]].to_numpy(float)
    sig = B.ar_sigma(Y, 2)
    Yt, X = B.lagmat(Y, 2)
    B0, V0 = B.minnesota(3, 2, sig)
    M_, _ = B.minnesota_posterior(Yt, X, B0, V0, sig)
    i = 1
    V = np.diag(V0[:, i])
    direct = np.linalg.solve(np.linalg.inv(V) + X.T @ X / sig[i] ** 2, np.linalg.inv(V) @ B0[:, i] + X.T @ Yt[:, i] / sig[i] ** 2)
    assert np.allclose(M_[:, i], direct)


@pytest.mark.parametrize("prior", ["minnesota", "inw", "nd"])
def test_block_exogeneity_switches_off_feedback(prior):
    d, s = frame("bvar")
    r = E.run("bvar", d, dict(s, prior=prior, block=["log_gdp"], draws=300, burn=200, lags=2))
    eq = rows(next(t for t in r["tables"] if t["title"] == "Equation: log_gdp"))
    for v in ("log_price", "log_m2"):
        assert abs(eq[f"L1.{v}"][1]) < 1e-3


def test_nw_marginal_likelihood_against_monte_carlo():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((30, 1)).cumsum(0) * 0.1 + 1
    Yt, X = B.lagmat(Y, 1)
    B0, Phi0, S0, a0 = B.nw_prior(1, 1, np.array([0.1]), 0.5, 1.0, 10.0, 1.0)
    ml = B.nw_logml(Yt, X, B0, Phi0, S0, a0)
    from scipy import stats
    draws = []
    for _ in range(40000):
        s2 = stats.invwishart.rvs(a0, S0, random_state=rng)
        b = B0[:, 0] + np.sqrt(s2) * rng.standard_normal(2) * np.sqrt(np.diag(Phi0))
        e = Yt[:, 0] - X @ b
        draws.append(-0.5 * len(e) * np.log(2 * np.pi * s2) - 0.5 * e @ e / s2)
    draws = np.array(draws)
    mc = draws.max() + np.log(np.mean(np.exp(draws - draws.max())))
    assert ml == pytest.approx(mc, abs=0.1)


def test_favar_factors_are_principal_components():
    d, s = frame("favar")
    r = E.run("favar", d, s)
    Xs = d[s["x"]].to_numpy(float)
    Xs = (Xs - Xs.mean(0)) / Xs.std(0, ddof=1)
    ev = np.linalg.eigvalsh(np.cov(Xs.T))[::-1]
    assert r["tables"][0]["rows"][0][1] == pytest.approx(ev[0] / ev.sum(), abs=1e-4)


def test_mean_adjusted_bvar_recovers_the_steady_state():
    df = _sim_var(c=(1.0, -0.5))
    mu = np.linalg.solve(np.eye(2) - np.array(((0.5, 0.1), (0.0, 0.6))), [1.0, -0.5])
    r = E.run("mabvar", df, dict(vars=["y0", "y1"], lags=1, ss_mean="0, 0", ss_sd="10, 10", draws=1500, burn=500))
    t = r["tables"][0]["rows"]
    for i in range(2):
        assert abs(t[i][1] - mu[i]) < 3 * t[i][2] + 0.05
    tight = E.run("mabvar", df, dict(vars=["y0", "y1"], lags=1, ss_mean="5, 5", ss_sd="0.001, 0.001", draws=500, burn=200))
    assert tight["tables"][0]["rows"][0][1] == pytest.approx(5, abs=0.01)


def test_threshold_var_finds_the_threshold():
    rng = np.random.default_rng(3)
    T = 600
    y = np.zeros((T, 2))
    for t in range(1, T):
        hi = y[t - 1, 0] > 0.0
        c = np.array([-0.4, 0.5]) if hi else np.array([0.4, -0.5])
        A = np.array([[0.2, 0.0], [0.1, 0.3]]) if hi else np.array([[0.6, 0.0], [-0.1, 0.5]])
        y[t] = c + A @ y[t - 1] + 0.5 * rng.standard_normal(2)
    df = pd.DataFrame(y, columns=["a", "b"])
    r = E.run("tvar", df, dict(vars=["a", "b"], threshold_var="a", lags=1, max_delay=2, draws=200, grid=60))
    s = {x["label"]: x["value"] for x in r["summary"]}
    assert s["Delay"] == 1 and abs(s["Threshold"]) < 0.3


def test_tvp_var_detects_a_shift_in_the_constant():
    df = pd.concat([_sim_var(T=150, c=(0.0, 0.0), seed=1), _sim_var(T=150, c=(2.0, 0.0), seed=2)], ignore_index=True)
    r = E.run("tvpvar", df, dict(vars=["y0", "y1"], lags=1, training=40, kQ=0.05, draws=300, burn=200))
    t = next(row for row in r["tables"][0]["rows"] if row[0] == "y0" and row[1] == "const")
    assert t[3] - t[2] > 1.0 and t[5] > 0.95


def test_sv_var_detects_higher_volatility():
    a = _sim_var(T=200, seed=4)
    b = _sim_var(T=200, seed=5, S=((9.0, 0.0), (0.0, 1.0)))
    df = pd.concat([a, b], ignore_index=True)
    r = E.run("svvar", df, dict(vars=["y0", "y1"], lags=1, draws=600, burn=400))
    vol = r["charts"][0]["series"][0]["y"]
    assert np.mean(vol[250:]) > 2 * np.mean(vol[:150])


def test_mixed_frequency_bvar_recovers_the_latent_months():
    rng = np.random.default_rng(6)
    T = 300
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.5, 0.1], [0.3, 0.2, 0.5]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.standard_normal(3)
    q = np.full(T, np.nan)
    for t in range(2, T, 3):
        q[t] = y[t - 2:t + 1, 2].mean()
    df = pd.DataFrame({"m1": y[:, 0], "m2": y[:, 1], "q": q})
    r = E.run("mfbvar", df, dict(monthly=["m1", "m2"], quarterly=["q"], aggregation="average", lags=3, draws=150, burn=100))
    est = np.array(r["charts"][0]["series"][0]["y"], float)
    truth = y[-len(est):, 2]
    assert np.corrcoef(est, truth)[0, 1] > 0.8


# ------------------------------------------------------------------------------------------------ panel VAR
def test_panel_var_mean_group_is_the_average_of_unit_vars():
    d, s = frame("pvar")
    r = E.run("pvar", d, s)
    Bs = []
    for _, g in d.sort_values(["country", "year"]).groupby("country"):
        Yt, X = B.lagmat(g[s["vars"]].to_numpy(float), 1)
        Bs.append(np.linalg.lstsq(X, Yt, rcond=None)[0])
    t = rows(r["tables"][0])
    assert t["L1.gdp_growth"][1] == pytest.approx(np.mean(Bs, 0)[1, 0], abs=1e-5)


def test_pooled_panel_var_with_a_loose_prior_is_pooled_ols():
    d, s = frame("pvar")
    r = E.run("pvar", d, dict(s, estimator="pooled", lambda1=1e4, draws=300))
    Ys, Xs = [], []
    for _, g in d.sort_values(["country", "year"]).groupby("country"):
        Yt, X = B.lagmat(g[s["vars"]].to_numpy(float), 1)
        Ys.append(Yt); Xs.append(X)
    ols = np.linalg.lstsq(np.vstack(Xs), np.vstack(Ys), rcond=None)[0]
    t = rows(r["tables"][0])
    assert t["L1.gdp_growth"][1] == pytest.approx(ols[1, 0], abs=0.01)


@pytest.mark.parametrize("est", ["zh", "hierarchical", "static", "dynamic"])
def test_panel_var_estimators_recover_persistence(est):
    rng = np.random.default_rng(7)
    rows_ = []
    for u in range(6):
        y = np.zeros((60, 2))
        for t in range(1, 60):
            y[t] = np.array([[0.6, 0.0], [0.0, 0.6]]) @ y[t - 1] + rng.standard_normal(2)
        rows_ += [dict(u=u, t=t, a=y[t, 0], b=y[t, 1]) for t in range(60)]
    df = pd.DataFrame(rows_)
    r = E.run("pvar", df, dict(vars=["a", "b"], unit="u", time="t", lags=1, estimator=est, draws=400, burn=300))
    assert "error" not in r, r.get("error")
    if est in ("zh", "hierarchical"):
        assert rows(r["tables"][0])["L1.a"][1] == pytest.approx(0.6, abs=0.12)
    else:
        f_ = rows(r["tables"][0])
        own = sum(f_[k][1] for k in f_ if k in ("common", "own lag", "same unit (domestic)", "lag 1"))
        unit_var = np.mean([f_[k][1] for k in f_ if k.startswith("unit:")]) + np.mean([f_[k][1] for k in f_ if k.startswith("variable:")])
        assert own + unit_var == pytest.approx(0.6, abs=0.15)


# ------------------------------------------------------------------------------------------------ Markov switching
def test_ms_regression_reproduces_stata_mswitch():
    r = E.BY_ID["ms_reg"]
    out = E.run("ms_reg", *_ex("ms_reg"))
    t = rows(out["tables"][0])
    # Stata 14 manual, mswitch dr: p11 = .9820939, p21 = .0503587, mu1 = 3.70877, mu2 = 9.556793, sigma = 2.107562
    assert t["p[0->0]"][1] == pytest.approx(0.9820939, abs=1e-4)
    assert t["const[0]"][1] == pytest.approx(3.70877, abs=1e-3)
    assert t["const[1]"][1] == pytest.approx(9.556793, abs=1e-3)
    assert t["sigma2"][1] == pytest.approx(2.107562 ** 2, abs=1e-3)


def test_ms_ar_reproduces_hamilton_1989():
    out = E.run("ms_ar", *_ex("ms_ar"))
    t = rows(out["tables"][0])
    assert t["const[0]"][1] == pytest.approx(-0.358811, abs=1e-3)
    assert t["const[1]"][1] == pytest.approx(1.163516, abs=1e-3)
    assert t["ar.L1"][1] == pytest.approx(0.013486, abs=1e-3)
    assert {x["label"]: x["value"] for x in out["summary"]}["Log-likelihood"] == pytest.approx(-181.26339, abs=1e-3)


def _ex(mid):
    from test_models import csv
    ex = E.BY_ID[mid]["example"]
    return csv(ex["file"]), dict(ex["spec"], ops=ex.get("ops", []))


@pytest.mark.parametrize("kind,sv", [("MSI", False), ("MSIH", True)])
def test_msvar_likelihood_matches_statsmodels(kind, sv):
    text, _ = _ex("ms_reg")
    ff = pd.read_csv(__import__("io").StringIO(text))
    fit = M._ms_em([(ff[["fedfunds"]].to_numpy(float), np.zeros((len(ff), 0)))], 2, kind)
    mod = sm.tsa.MarkovRegression(ff.fedfunds, k_regimes=2, switching_variance=sv)
    P = fit["P"]
    par = [P[0, 0], P[1, 0], fit["nu"][0][0], fit["nu"][1][0]] + ([fit["S"][0][0, 0], fit["S"][1][0, 0]] if sv else [fit["S"][0][0, 0]])
    assert fit["llf"] == pytest.approx(mod.loglike(np.array(par)), abs=1e-6)     # same likelihood function
    ref = mod.fit(disp=False)
    assert fit["llf"] >= ref.llf - 0.05                                          # and at least as good an optimum


def _sim_msvar(T=500, units=1, seed=8):
    rng = np.random.default_rng(seed)
    P = np.array([[0.95, 0.05], [0.1, 0.9]])
    s = 0
    states = []
    for t in range(T):
        s = rng.choice(2, p=P[s])
        states.append(s)
    out = []
    for u in range(units):
        y = np.zeros((T, 2))
        for t in range(1, T):
            nu = np.array([-1.0, 0.5]) if states[t] == 0 else np.array([2.0, -0.5])
            sd = 0.5 if states[t] == 0 else 1.0
            y[t] = nu + 0.3 * y[t - 1] + sd * rng.standard_normal(2)
        out.append(y)
    return out, np.array(states)


def test_msvar_recovers_simulated_regimes():
    (y,), st = _sim_msvar()
    df = pd.DataFrame(y, columns=["a", "b"])
    r = E.run("msvar", df, dict(vars=["a", "b"], lags=1, switching="MSIH"))
    t = rows(r["tables"][0])
    assert t["a"][1] == pytest.approx(-1.0, abs=0.25) and t["a"][2] == pytest.approx(2.0, abs=0.25)
    prob1 = np.array(r["charts"][0]["series"][1]["y"])
    assert np.mean((prob1 > 0.5) == (st[1:] == 1)) > 0.9


def test_ms_panel_var_recovers_common_regimes():
    ys, st = _sim_msvar(T=200, units=4, seed=9)
    df = pd.concat([pd.DataFrame(dict(u=u, t=np.arange(200), a=y[:, 0], b=y[:, 1])) for u, y in enumerate(ys)])
    r = E.run("mspvar", df, dict(vars=["a", "b"], unit="u", time="t", lags=1, switching="MSIH", demean="no"))
    prob1 = np.array(r["charts"][0]["series"][1]["y"])
    assert np.mean((prob1 > 0.5) == (st[1:] == 1)) > 0.95
    one = E.run("mspvar", df[df.u == 0], dict(vars=["a", "b"], unit="u", time="t", lags=1, demean="no")) if False else None
