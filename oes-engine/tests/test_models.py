"""Tests of every OES model.

1. Every model runs on its guide example and returns a well-formed result.
2. Estimates are checked against reference implementations (statsmodels, linearmodels, scikit-learn, xgboost),
   against the numbers published in the website guides, or against known parameters of simulated data.

Run from oes-engine/:  python -m pytest -q        (set OES_SITE to a checkout of the website for the example files)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import oes_engine as E

SITE = Path(os.environ.get("OES_SITE", Path(__file__).resolve().parents[3] / "almamun"))
FAST = os.environ.get("OES_FAST") == "1"
SLOW = {"dml_irm", "dml_iv", "causal_forest", "staggered", "did", "tvpvar", "mfbvar", "svvar"}


def site(path):
    p = SITE / path
    if not p.exists():
        pytest.skip(f"website example {path} not found (set OES_SITE)")
    return p


def csv(path):
    return site(path).read_text()


def example(model_id, **override):
    m = E.BY_ID[model_id]
    ex = m["example"]
    spec = dict(ex["spec"], ops=ex.get("ops", []), **override)
    return E.run(model_id, csv(ex["file"]), spec)


def rows(table):
    return {r[0]: r for r in table["rows"]}


def frame(model_id):
    ex = E.BY_ID[model_id]["example"]
    return E.prepare(pd.read_csv(site(ex["file"])), ex.get("ops", [])), ex["spec"]


# ------------------------------------------------------------------------------------------------ every model runs
@pytest.mark.parametrize("model_id", [m["id"] for m in E.MODELS])
def test_every_model_runs_on_its_guide_example(model_id):
    if FAST and model_id in SLOW:
        pytest.skip("slow model skipped with OES_FAST=1")
    r = example(model_id)
    assert "error" not in r, r.get("error")
    assert r["model"] == model_id and r["nobs"] > 0
    assert r["tables"] and all(t["rows"] for t in r["tables"][:1])
    for t in r["tables"]:
        assert all(len(row) == len(t["columns"]) for row in t["rows"]), t["title"]
    json.dumps(r, allow_nan=False)          # strictly valid JSON for the browser


def test_catalog_covers_every_guide_page():
    pages = {"time-series-guide": {"ols", "ardl", "nardl", "var", "vecm", "gmm", "lp", "svar", "bp"},
             "panel-data-guide": {"pols", "fe", "re", "diff", "fedk", "gmm", "mg", "pmg", "csardl", "plp", "pthresh"},
             "causal-inference-guide": {"did", "staggered", "its", "iv", "rdd", "scm", "gsc"},
             "ml-guide": {"lasso", "ridge", "enet", "rf", "xgb", "dml-plr", "dml-irm", "dml-iv", "causal-forest"},
             "bayesian-guide": {"bvar", "bayes-reg", "favar", "steady-state", "panel-var", "tvp-var", "sv-var", "mixed-frequency", "threshold-var"},
             "markov-guide": {"ms-ar", "ms-regression", "ms-var", "ms-panel-var"}}
    guides = {m["guide"] for m in E.MODELS if m["guide"]}
    for folder, slugs in pages.items():
        for s in slugs:
            assert f"{folder}/{s}.html" in guides, f"{folder}/{s}.html has no OES model"
    assert len(E.MODELS) == 49
    assert all(m["guide"] for m in E.MODELS)                       # every model links to a guide page
    json.dumps(E.catalog())


# ------------------------------------------------------------------------------------------------ time series
def test_ols_matches_statsmodels():
    import statsmodels.api as sm
    d, s = frame("ols")
    ref = sm.OLS(d[s["y"]], sm.add_constant(d[s["x"]])).fit(cov_type="HC1")
    r = example("ols")
    t = rows(r["tables"][0])
    for v in ["const"] + s["x"]:
        assert t[v][1] == pytest.approx(ref.params[v], abs=1e-5)
        assert t[v][2] == pytest.approx(ref.bse[v], abs=1e-5)


def test_ardl_reproduces_the_guide():
    r = example("ardl")
    s = {x["label"]: x["value"] for x in r["summary"]}
    assert s["Model"] == "ARDL(3, 2, 1, 3)"
    assert s["Bounds F"] == pytest.approx(8.18, abs=0.01)          # guide: F = 8.18
    assert s["Speed of adjustment"] == pytest.approx(-0.045, abs=0.001)
    lr = rows(next(t for t in r["tables"] if t["title"] == "Long-run coefficients"))
    assert lr["inflation"][1] == pytest.approx(1.98, abs=0.01)        # guide: 1.98 (s.e. 0.45)
    assert lr["inflation"][2] == pytest.approx(0.45, abs=0.01)


def test_nardl_partial_sums_and_symmetry_tests():
    r = example("nardl")
    names = [row[0] for row in r["tables"][3]["rows"]]
    assert any("Long-run symmetry" in n for n in names) and any("Short-run symmetry" in n for n in names)
    d = E.prepare(pd.DataFrame({"x": [1.0, 3.0, 2.0, 2.5]}), [])
    dx = d["x"].diff().fillna(0)
    assert list(dx.clip(lower=0).cumsum()) == [0, 2, 2, 2.5] and list(dx.clip(upper=0).cumsum()) == [0, 0, -1, -1]


def test_var_matches_statsmodels():
    from statsmodels.tsa.api import VAR
    d, s = frame("var")
    data = d[s["vars"]].astype(float).diff().dropna().reset_index(drop=True)
    r = example("var", lags=2)
    ref = VAR(data).fit(2)
    eq = rows(next(t for t in r["tables"] if t["title"] == f"Equation: {s['vars'][0]}"))
    for name, val in ref.params[s["vars"][0]].items():
        assert eq[name][1] == pytest.approx(val, abs=1e-6)


def test_vecm_matches_statsmodels():
    from statsmodels.tsa.vector_ar.vecm import VECM
    d, s = frame("vecm")
    r = example("vecm", lags=2, rank=1)
    ref = VECM(d[s["vars"]].astype(float).reset_index(drop=True), k_ar_diff=2, coint_rank=1, deterministic="ci").fit()
    alpha = rows(next(t for t in r["tables"] if t["title"].startswith("Adjustment")))
    for i, v in enumerate(s["vars"]):
        assert alpha[v][1] == pytest.approx(ref.alpha[i, 0], abs=1e-6)


@pytest.mark.parametrize("method,cov", [("2sls", "kernel"), ("2sls", "robust"), ("2sls", "unadjusted"), ("gmm", "kernel"), ("gmm", "robust")])
def test_iv_gmm_matches_linearmodels(method, cov):
    lm = pytest.importorskip("linearmodels.iv")
    d, s = frame("iv_gmm")
    d = d.dropna(subset=[s["y"]] + s["endog"] + s["instruments"] + s["exog"]).copy()
    d["const"] = 1.0
    args = (d[s["y"]], d[["const"] + s["exog"]], d[s["endog"]], d[s["instruments"]])
    kw = dict(kernel=dict(cov_type="kernel", kernel="bartlett", bandwidth=4), robust=dict(cov_type="robust"),
              unadjusted=dict(cov_type="unadjusted"))[cov]
    if method == "2sls":
        ref = lm.IV2SLS(*args).fit(**kw)
    else:
        wt = dict(weight_type="kernel", kernel="bartlett", bandwidth=4) if cov == "kernel" else dict(weight_type="robust")
        ref = lm.IVGMM(*args, **wt).fit(iter_limit=2, **kw)
    r = example("iv_gmm", method=method, cov=cov)
    t = r["tables"][0]["rows"]
    assert [x[1] for x in t] == pytest.approx(list(ref.params.values), abs=1e-4)
    assert [x[2] for x in t] == pytest.approx(list(ref.std_errors.values), abs=1e-4)
    if method == "gmm":
        j = next(x for x in r["tables"][3]["rows"] if x[0].startswith("Hansen"))
        assert j[1] == pytest.approx(ref.j_stat.stat, abs=1e-3)


# ------------------------------------------------------------------------------------------------ panel data
@pytest.mark.parametrize("cov", ["unadjusted", "robust", "clustered", "dk"])
@pytest.mark.parametrize("te", [False, True])
def test_fixed_effects_match_linearmodels(cov, te):
    lm = pytest.importorskip("linearmodels.panel")
    import statsmodels.api as sm
    df = pd.read_csv(site("panel-data-guide/examples/crime4.csv"))
    xs = ["lprbarr", "lprbconv", "lprbpris", "lavgsen", "lpolpc"]
    p = df.set_index(["county", "year"])
    kw = dict(unadjusted={}, robust=dict(cov_type="robust"), clustered=dict(cov_type="clustered", cluster_entity=True),
              dk=dict(cov_type="kernel", kernel="bartlett", bandwidth=3))[cov]
    ref = lm.PanelOLS(p["lcrmrte"], sm.add_constant(p[xs]), entity_effects=True, time_effects=te).fit(**kw)
    r = E.run("fe", df, dict(y="lcrmrte", x=xs, unit="county", time="year", cov=cov, bandwidth=3, time_effects="yes" if te else "no"))
    t = rows(r["tables"][0])
    for v in xs:
        assert t[v][1] == pytest.approx(ref.params[v], abs=1e-5)
        assert t[v][2] == pytest.approx(ref.std_errors[v], rel=1e-4)


@pytest.mark.parametrize("model,cls", [("pols", "PooledOLS"), ("re", "RandomEffects")])
@pytest.mark.parametrize("cov", ["unadjusted", "clustered"])
def test_pooled_and_random_effects_match_linearmodels(model, cls, cov):
    lm = pytest.importorskip("linearmodels.panel")
    import statsmodels.api as sm
    df = pd.read_csv(site("panel-data-guide/examples/crime4.csv"))
    xs = ["lprbarr", "lprbconv", "lprbpris", "lavgsen", "lpolpc"]
    p = df.set_index(["county", "year"])
    kw = {} if cov == "unadjusted" else dict(cov_type="clustered", cluster_entity=True)
    ref = getattr(lm, cls)(p["lcrmrte"], sm.add_constant(p[xs])).fit(**kw)
    r = E.run(model, df, dict(y="lcrmrte", x=xs, unit="county", time="year", cov=cov))
    t = rows(r["tables"][0])
    for v in ["const"] + xs:
        assert t[v][1] == pytest.approx(ref.params[v], abs=1e-5)
        assert t[v][2] == pytest.approx(ref.std_errors[v], rel=1e-4)


def test_fedk_matches_the_guide_setup():
    lm = pytest.importorskip("linearmodels.panel")
    import statsmodels.api as sm
    d, s = frame("fedk")
    x = d.dropna(subset=[s["y"]] + s["x"]).set_index(["country", "year"])
    ref = lm.PanelOLS(x[s["y"]], sm.add_constant(x[s["x"]]), entity_effects=True).fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
    t = rows(example("fedk")["tables"][0])
    for v in s["x"]:
        assert t[v][2] == pytest.approx(ref.std_errors[v], rel=1e-4)


def test_first_differences_match_linearmodels():
    lm = pytest.importorskip("linearmodels.panel")
    import statsmodels.api as sm
    df, s = frame("fd")
    df = df.sort_values(["country", "year"])
    d = df.groupby("country")[[s["y"]] + s["x"]].diff()
    d[["country", "year"]] = df[["country", "year"]]
    d = d.dropna().set_index(["country", "year"])
    ref = lm.PanelOLS(d[s["y"]], sm.add_constant(d[s["x"]])).fit(cov_type="kernel", kernel="bartlett", bandwidth=2)
    t = rows(example("fd")["tables"][0])
    for v in s["x"]:
        assert t["D." + v][1] == pytest.approx(ref.params[v], abs=1e-6)
        assert t["D." + v][2] == pytest.approx(ref.std_errors[v], rel=1e-4)


def test_difference_gmm_reproduces_arellano_bond():
    # Arellano and Bond (1991), Table 4 column (a1): one-step estimates n(-1) = 0.686, n(-2) = -0.085, w = -0.608
    t = rows(example("diff_gmm")["tables"][0])
    assert t["L1.log_emp"][1] == pytest.approx(0.686, abs=0.002)
    assert t["L2.log_emp"][1] == pytest.approx(-0.085, abs=0.002)
    assert t["log_wage"][1] == pytest.approx(-0.608, abs=0.002)


def test_mean_group_is_the_average_of_unit_regressions():
    d, s = frame("mg")
    d = d.dropna(subset=[s["y"]] + s["x"])
    b = []
    for _, g in d.groupby("country"):
        if len(g) >= len(s["x"]) + 4:
            X = np.column_stack([np.ones(len(g)), g[s["x"]]])
            b.append(np.linalg.lstsq(X, g[s["y"]], rcond=None)[0][1])
    t = rows(example("mg")["tables"][0])
    assert t["sr"][1] == pytest.approx(np.mean(b), abs=1e-5)
    assert t["sr"][2] == pytest.approx(np.std(b, ddof=1) / np.sqrt(len(b)), abs=1e-5)


def test_pmg_converges_and_reports_hausman():
    r = example("pmg")
    s = {x["label"]: x["value"] for x in r["summary"]}
    assert s["PMG iterations"] < 500 and s["PMG speed of adjustment"] < 0
    assert any(row[0].startswith("Hausman PMG vs MG") for row in r["tables"][2]["rows"])


def test_csardl_simulated_long_run():
    rng = np.random.default_rng(1)
    N, T, theta = 20, 40, 0.8
    f = np.cumsum(rng.normal(size=T)) * 0.1
    rows_ = []
    for i in range(N):
        x = np.zeros(T); y = np.zeros(T)
        for t in range(1, T):
            x[t] = 0.5 * x[t - 1] + rng.normal() + f[t]
            y[t] = y[t - 1] - 0.5 * (y[t - 1] - theta * x[t - 1]) + 0.2 * (x[t] - x[t - 1]) + rng.normal(scale=0.3) + f[t]
        rows_ += [dict(u=i, t=t, y=y[t], x=x[t]) for t in range(T)]
    r = E.run("csardl", pd.DataFrame(rows_), dict(y="y", x=["x"], unit="u", time="t", pt=1))
    lr = rows(r["tables"][0])["x"]
    assert abs(lr[1] - theta) < 3 * lr[2] + 0.1


# ------------------------------------------------------------------------------------------------ causal inference
def test_iv_card_matches_linearmodels():
    lm = pytest.importorskip("linearmodels.iv")
    d, s = frame("iv")
    d = d.copy(); d["const"] = 1.0
    ref = lm.IV2SLS(d[s["outcome"]], d[["const"] + s["controls"]], d[s["treat"]], d[s["instruments"]]).fit(cov_type="robust")
    t = example("iv")["tables"][0]["rows"][0]
    assert t[1] == pytest.approx(ref.params[s["treat"]], abs=1e-4)
    assert t[2] == pytest.approx(ref.std_errors[s["treat"]], rel=0.01)


def test_rdd_matches_the_guide_helper():
    from oes_engine import _causal_core as C
    d, s = frame("rdd")
    ref = C.rd_estimate(d[s["running"]] - 0.5, d[s["outcome"]], 0.0)
    r = example("rdd")
    est = r["tables"][0]["rows"][0]
    assert est[1] == pytest.approx(ref["est"], rel=1e-3)


def test_twfe_in_did_matches_linearmodels():
    lm = pytest.importorskip("linearmodels.panel")
    d, s = frame("did")
    p = d.set_index([s["unit"], s["time"]])
    ref = lm.PanelOLS(p[s["outcome"]], p[[s["treat"]]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    r = example("did", boot=49)
    tw = r["tables"][0]["rows"][1]
    assert tw[1] == pytest.approx(ref.params[s["treat"]], abs=1e-3)


def test_did_recovers_simulated_effect():
    rng = np.random.default_rng(0)
    rows_ = []
    for i in range(60):
        a = rng.normal()
        tr = i < 30
        for t in range(10):
            d = int(tr and t >= 5)
            rows_.append(dict(u=i, t=t, d=d, y=a + 0.1 * t + 2.0 * d + rng.normal(scale=0.5)))
    r = E.run("did", pd.DataFrame(rows_), dict(outcome="y", treat="d", unit="u", time="t", boot=99))
    att = r["tables"][0]["rows"][0]
    assert abs(att[1] - 2.0) < 4 * att[2]


def test_synthetic_control_weights_sum_to_one():
    r = example("scm")
    w = [row[1] for row in r["tables"][0]["rows"]]
    assert all(x >= 0 for x in w) and sum(w) <= 1.0001


# ------------------------------------------------------------------------------------------------ machine learning
def test_lasso_coefficients_match_sklearn():
    from sklearn.linear_model import LassoCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from oes_engine.ml import _split
    d, s = frame("lasso")
    d = d.dropna(subset=[s["y"]] + s["x"]).reset_index(drop=True)
    tr, te = _split(len(d), False, 0.2)
    m = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0, max_iter=20000)).fit(d.loc[tr, s["x"]], d.loc[tr, s["y"]])
    t = rows(example("lasso")["tables"][1])
    for v, c in zip(s["x"], m[-1].coef_):
        assert t[v][1] == pytest.approx(c, abs=1e-6)


def test_xgboost_wrapper_matches_native_xgboost():
    xgboost = pytest.importorskip("xgboost")
    from oes_engine.ml import XGBoostRegressor
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)); y = X[:, 0] ** 2 + X[:, 1] + rng.normal(scale=0.1, size=300)
    m = XGBoostRegressor(n_estimators=50).fit(X, y)
    params = {"objective": "reg:squarederror", "eta": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 1.0, "lambda": 1.0, "seed": 0, "nthread": 1, "tree_method": "hist"}
    ref = xgboost.train(params, xgboost.DMatrix(X, label=y), 50)
    assert np.allclose(m.predict(X), ref.predict(xgboost.DMatrix(X)))
    assert m.score(X, y) > 0.8


def _sim_causal(n=1500, binary=False, seed=0, theta=1.0, het=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    z = rng.binomial(1, 0.5, n).astype(float)
    if binary:
        p = 1 / (1 + np.exp(-(0.5 * X[:, 0])))
        d = rng.binomial(1, p).astype(float)
    else:
        d = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(size=n)
    eff = theta + (1.5 * X[:, 2] if het else 0)
    y = eff * d + np.cos(X[:, 0]) + X[:, 1] ** 2 + rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    df["d"], df["y"], df["z"] = d, y, z
    return df


def test_dml_plr_recovers_simulated_effect():
    df = _sim_causal()
    r = E.run("dml_plr", df, dict(y="y", treat="d", x=[f"x{i}" for i in range(5)], learner="rf", reps=1))
    est = r["tables"][0]["rows"][0]
    assert abs(est[1] - 1.0) < 4 * est[2]


def test_dml_irm_recovers_simulated_effect():
    df = _sim_causal(binary=True)
    r = E.run("dml_irm", df, dict(y="y", treat="d", x=[f"x{i}" for i in range(5)], learner="rf", reps=1))
    est = r["tables"][0]["rows"][0]
    assert abs(est[1] - 1.0) < 4 * est[2]


def test_dml_iv_recovers_simulated_effect():
    rng = np.random.default_rng(3)
    n = 2000
    X = rng.normal(size=(n, 4)); z = rng.normal(size=n); u = rng.normal(size=n)
    d = 0.8 * z + np.sin(X[:, 0]) + u + rng.normal(size=n)
    y = 1.0 * d + X[:, 1] ** 2 + 2 * u + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["a", "b", "c", "e"]); df["d"], df["y"], df["z"] = d, y, z
    r = E.run("dml_iv", df, dict(y="y", treat="d", instrument="z", x=["a", "b", "c", "e"], learner="rf", reps=1))
    est = r["tables"][0]["rows"][0]
    assert abs(est[1] - 1.0) < 4 * est[2]


def test_causal_forest_detects_heterogeneity():
    df = _sim_causal(n=3000, binary=True, het=True)
    r = E.run("causal_forest", df, dict(y="y", treat="d", x=[f"x{i}" for i in range(5)]))
    ate = r["tables"][0]["rows"][0]
    assert abs(ate[1] - 1.0) < 4 * ate[2]
    slope = r["tables"][1]["rows"][1]
    assert slope[1] > 0 and slope[4] < 0.05
    assert r["tables"][3]["rows"][0][0] == "x2"           # the effect modifier is the most important variable


# ------------------------------------------------------------------------------------------------ data operations and errors
def test_data_operations_by_unit():
    df = pd.DataFrame({"u": [1, 1, 1, 2, 2], "t": [2001, 2002, 2003, 2001, 2002], "x": [1.0, 2.0, 4.0, 10.0, 20.0]})
    out = E.prepare(df, [dict(op="lag", vars=["x"], lags=[1], unit="u", time="t"), dict(op="diff", vars=["x"], unit="u", time="t"),
                         dict(op="growth", vars=["x"], unit="u", time="t"), dict(op="log", vars=["x"]),
                         dict(op="demean", vars=["x"], time="t")])
    out = out.sort_values(["u", "t"]).reset_index(drop=True)
    assert np.isnan(out["x_l1"][0]) and out["x_l1"][1] == 1.0 and np.isnan(out["x_l1"][3])
    assert list(out["d_x"].fillna(-1)) == [-1, 1, 2, -1, 10]
    assert out["g_x"][1] == pytest.approx(100 * np.log(2))
    assert out["dm_x"][0] == pytest.approx(1 - 5.5)
    s = E.prepare(pd.DataFrame({"q": ["1979Q2", "1979Q3", "2007Q4", "2008Q1"], "v": [1, 2, 3, 4]}),
                  [dict(op="sample", time="q", start="1979Q3", end="2007Q4")])
    assert list(s["v"]) == [2, 3]


@pytest.mark.parametrize("model_id,spec", [("ols", dict(y="nope", x=["a"])), ("fe", dict(y="a", x=["b"])), ("var", dict(vars=["a"])),
                                           ("iv_gmm", dict(y="a", endog=["b", "c"], instruments=["d"])), ("unknown", {})])
def test_bad_specifications_return_errors(model_id, spec):
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 4)), columns=list("abcd"))
    r = E.run(model_id, df.to_csv(index=False), spec)
    assert "error" in r and isinstance(r["error"], str)
