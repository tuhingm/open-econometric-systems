"""SVAR, Bai-Perron breaks and Hansen panel threshold: checked against statsmodels, brute force and simulated truth."""
import itertools
import numpy as np
import pandas as pd
import pytest

import oes_engine as E
from oes_engine import _bp_cv
from oes_engine.breaks import ssr_table, partitions, sup_f
from test_models import SITE


def _read(rel):
    return pd.read_csv(SITE / rel)


# ------------------------------------------------------------------------------------------------ SVAR
def test_svar_cholesky_matches_statsmodels():
    from statsmodels.tsa.api import VAR
    d = _read("time-series-guide/examples/yield.csv")[["date", "gdp_growth", "unemployment"]].dropna()
    out = E.svar.svar(d, dict(vars=["gdp_growth", "unemployment"], time="date", lags=4, horizon=8, boot=50))
    ref = VAR(d[["gdp_growth", "unemployment"]].to_numpy()).fit(4).irf(8).orth_irfs
    rows = out["tables"][0]["rows"]
    for i, v in enumerate(["gdp_growth", "unemployment"]):
        got = [r[2] for r in rows if r[0] == v]
        assert got == pytest.approx(ref[:, i, 0], abs=1e-5)


def test_svar_long_run_restriction_holds():
    d = _read("time-series-guide/examples/yield.csv")[["date", "gdp_growth", "unemployment"]].dropna()
    out = E.svar.svar(d, dict(vars=["gdp_growth", "unemployment"], time="date", identification="longrun", lags=4, horizon=8, boot=50))
    lr = [t for t in out["tables"] if t["title"] == "Long-run impact matrix"][0]["rows"]
    assert float(lr[0][2]) == pytest.approx(0.0, abs=1e-8)        # shock 2 has no long-run effect on the first variable
    # the long-run response of the first variable to shock 1 equals the sum of its responses in a stationary VAR
    assert abs(float(lr[0][1])) > 0.1


def test_svar_proxy_recovers_the_true_impact():
    rng = np.random.default_rng(3)
    T, n = 600, 3
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    Bt = np.array([[1.0, 0.0, 0.0], [0.6, 1.0, 0.0], [-0.4, 0.3, 1.0]])       # shock 1 raises variable 1 by 1, variable 2 by 0.6, variable 3 by -0.4
    e = rng.standard_normal((T + 50, n))
    Y = np.zeros((T + 50, n))
    for t in range(1, T + 50):
        Y[t] = A @ Y[t - 1] + Bt @ e[t]
    Y, e = Y[50:], e[50:]
    z = e[:, 0] + 0.5 * rng.standard_normal(T)
    d = pd.DataFrame(Y, columns=["a", "b", "c"]); d["z"] = z
    out = E.svar.svar(d, dict(vars=["a", "b", "c"], shock="a", identification="proxy", instrument=["z"], lags=1, horizon=4, boot=60))
    imp = {r[0]: float(r[1]) for r in [t for t in out["tables"] if t["title"].startswith("Impact")][0]["rows"]}
    assert imp["a"] == pytest.approx(1.0) and imp["b"] == pytest.approx(0.6, abs=0.12) and imp["c"] == pytest.approx(-0.4, abs=0.12)
    F = [s["value"] for s in out["summary"] if s["label"] == "First-stage F"][0]
    assert F > 100


# ------------------------------------------------------------------------------------------------ Bai-Perron
def test_dynamic_programming_matches_brute_force():
    rng = np.random.default_rng(1)
    T, h = 40, 6
    y = np.concatenate([rng.normal(0, 1, 15), rng.normal(3, 1, 12), rng.normal(0, 1, 13)])
    X = np.ones((T, 1))
    S = ssr_table(y, X, h)
    ssr, brk = partitions(S, T, h, 2)
    best = (np.inf, None)
    for a, b in itertools.combinations(range(h, T - h + 1), 2):
        if b - a < h:
            continue
        s = sum(((y[i:j] - y[i:j].mean()) ** 2).sum() for i, j in [(0, a), (a, b), (b, T)])
        if s < best[0]:
            best = (s, [a, b])
    assert ssr[2] == pytest.approx(best[0]) and brk[2] == best[1]


def test_breaks_finds_two_mean_shifts():
    rng = np.random.default_rng(11)
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 120), rng.normal(-1, 1, 80)])
    d = pd.DataFrame({"y": y, "t": np.arange(300)})
    out = E.breaks.breaks(d, dict(y="y", time="t"))
    s = {x["label"]: x["value"] for x in out["summary"]}
    assert s["Breaks selected"] == 2
    dates = [int(r[1]) for r in [t for t in out["tables"] if t["title"] == "Estimated break dates"][0]["rows"]]
    assert abs(dates[0] - 100) <= 6 and abs(dates[1] - 220) <= 6


def test_breaks_critical_values_match_andrews():
    # sup-F(1) with one changing coefficient and 15% trimming: Andrews (1993) gives 8.85 at the 5% level
    assert _bp_cv.CV[(1, 0.15)]["supf"][0] == pytest.approx(8.85, abs=0.6)
    assert 2 * _bp_cv.CV[(2, 0.15)]["supf"][0] == pytest.approx(11.79, abs=1.0)      # Andrews (1993), q = 2
    assert _bp_cv.CV[(1, 0.15)]["seq"][1] == pytest.approx(10.13, abs=0.6)           # Bai and Perron (2003), sup-F(2|1)


def test_breaks_no_break_is_rarely_rejected():
    rng = np.random.default_rng(2)
    rej = 0
    for _ in range(30):
        d = pd.DataFrame({"y": rng.standard_normal(150), "t": np.arange(150)})
        out = E.breaks.breaks(d, dict(y="y", time="t", select="sequential"))
        rej += [x["value"] for x in out["summary"] if x["label"] == "Breaks selected"][0] > 0
    assert rej <= 5


# ------------------------------------------------------------------------------------------------ Hansen threshold
def _panel(rng, N=60, T=12, gamma=0.5, b=(0.3, 1.2)):
    q = rng.uniform(-1, 2, (N, T))
    x = rng.standard_normal((N, T))
    mu = rng.standard_normal(N)[:, None]
    y = mu + np.where(q <= gamma, b[0], b[1]) * x + 0.5 * rng.standard_normal((N, T))
    return pd.DataFrame({"i": np.repeat(np.arange(N), T), "t": np.tile(np.arange(T), N), "y": y.ravel(), "x": x.ravel(), "q": q.ravel()})


def test_pthresh_finds_the_threshold():
    d = _panel(np.random.default_rng(4))
    out = E.pthreshold.pthresh(d, dict(y="y", x=["x"], threshold="q", unit="i", time="t", boot=100, grid=60))
    s = {x["label"]: x["value"] for x in out["summary"]}
    assert s["Thresholds found"] >= 1 and abs(s["Threshold estimate"] - 0.5) < 0.15
    coefs = [r for r in [t for t in out["tables"] if t["title"].startswith("Coefficients")][0]["rows"]]
    assert coefs[0][1] == pytest.approx(0.3, abs=0.15) and coefs[1][1] == pytest.approx(1.2, abs=0.15)


def test_pthresh_does_not_reject_linear_panel():
    rng = np.random.default_rng(8)
    d = _panel(rng, b=(0.8, 0.8))
    out = E.pthreshold.pthresh(d, dict(y="y", x=["x"], threshold="q", unit="i", time="t", boot=100, grid=60))
    p1 = float([t for t in out["tables"] if t["title"].startswith("Tests")][0]["rows"][0][2])
    assert p1 > 0.05


def test_pthresh_single_regime_equals_fixed_effects():
    # with the threshold pushed to the sample edge the regime-1 slope is the within estimate on that part; here we check the linear fit
    d = _panel(np.random.default_rng(5), b=(0.8, 0.8))
    from oes_engine.panel import fe
    ref = fe(d, dict(y="y", x=["x"], unit="i", time="t", effects="unit"))
    b_ref = float(ref["tables"][0]["rows"][0][1])
    out = E.pthreshold.pthresh(d, dict(y="y", x=["x"], threshold="q", unit="i", time="t", boot=50, grid=40))
    coefs = [t for t in out["tables"] if t["title"].startswith("Coefficients")][0]["rows"]
    assert coefs[0][1] == pytest.approx(b_ref, abs=0.2) and coefs[1][1] == pytest.approx(b_ref, abs=0.2)


def test_catalog_runs_new_guide_examples():
    for mid in ("svar", "bp", "pthresh"):
        ex = E.BY_ID[mid]["example"]
        out = E.run(mid, (SITE / ex["file"]).read_text(), dict(ex["spec"], ops=ex.get("ops", [])))
        assert "error" not in out and out["charts"], (mid, out.get("error"))
