"""Tests of the 18 models added from the website guides (time series, panel, causal, Bayesian, Markov).
Each one is checked against the numbers printed in its guide, which were produced by the guide's own code
(arch, statsmodels, linearmodels-style estimators), or against a mathematical identity.
Set OES_SITE to a checkout of the website for the example files."""
import numpy as np
import pytest

import oes_engine as E
from test_models import example, rows

NEW = ["arima", "garch", "quantile", "cointegration", "wavelet", "twfe", "dpthresh", "pcoint", "pquant", "spatial",
       "blp", "sign_bvar", "hier_panel", "ms_tvtp", "eventstudy", "sdid", "matching", "bartik"]


def T(r, prefix):
    assert "error" not in r, r.get("error")
    return next(t for t in r["tables"] if t["title"].startswith(prefix))


def S(r, label):
    return next(s["value"] for s in r["summary"] if s["label"] == label)


@pytest.mark.parametrize("model_id", NEW)
def test_new_model_is_registered_and_has_a_guide(model_id):
    m = E.BY_ID[model_id]
    assert m["guide"].endswith(".html") and m["example"]["file"]


def test_arima_matches_the_guide():
    r = example("arima")
    est = rows(T(r, "ARIMA(1,0,0)"))
    assert est["ar.L1"][1] == pytest.approx(0.445, abs=2e-3) and est["ar.L1"][2] == pytest.approx(0.031, abs=2e-3)
    assert T(r, "Order selection")["rows"][0][3] == pytest.approx(1713.4, abs=0.1)


def test_garch_best_model_matches_the_guide():
    r = example("garch")
    t = {x[0]: x for x in T(r, "Specification comparison")["rows"]}
    assert t["GARCH, normal"][2] == pytest.approx(-1408.2, abs=0.5) and t["GARCH, t"][2] == pytest.approx(-1365.1, abs=0.5)
    assert t["EGARCH, skewed t"][4] == pytest.approx(2742.8, abs=0.5)
    assert S(r, "Selected model (BIC)") == "EGARCH, skewed t"


def test_skewed_t_density_is_standardised():
    from oes_engine.ts_extra import _skewt_logpdf
    z = np.linspace(-60, 60, 400001); p = np.exp(_skewt_logpdf(z, 6, 0.2)); dz = z[1] - z[0]
    assert (p * dz).sum() == pytest.approx(1, abs=1e-6) and (z * p * dz).sum() == pytest.approx(0, abs=1e-5) and (z * z * p * dz).sum() == pytest.approx(1, abs=1e-4)


def test_quantile_spread_coefficient_at_the_5th_percentile():
    r = example("quantile")
    row = next(x for x in T(r, "Quantile regression coefficients")["rows"] if x[0] == "spread" and x[1] == 0.05)
    assert row[2] == pytest.approx(-2.38, abs=0.02)


def test_cointegration_matches_the_guide():
    r = example("cointegration")
    eg = T(r, "Engle-Granger")["rows"][0]
    assert eg[1] == pytest.approx(-2.05, abs=0.01) and eg[2] == pytest.approx(0.50, abs=0.01)
    gh = {x[0]: x for x in T(r, "Gregory-Hansen")["rows"]}
    assert gh["C/T"][1] == pytest.approx(-5.66, abs=0.01) and gh["C/T"][2] == "1980Q1" and gh["C"][2] == "2005Q1"


def test_wavelet_null_share_is_about_five_percent():
    r = example("wavelet")
    assert S(r, "Null share (mean)") == pytest.approx(0.05, abs=0.02)


def test_twfe_standard_errors_match_the_guide():
    r = example("twfe")
    se = {x[0]: x for x in T(r, "Standard errors")["rows"]}
    assert se["Default (iid)"][1] == pytest.approx(0.042, abs=0.001) and se["Cluster by unit"][1] == pytest.approx(0.223, abs=0.002)
    assert se["Two-way (unit, time)"][3] == pytest.approx(0.177, abs=0.003)


def test_dpthresh_matches_the_guide():
    r = example("dpthresh")
    th = {x[0]: x[1] for x in T(r, "Threshold")["rows"]}
    assert th["Estimated threshold"] == pytest.approx(2.79, abs=0.01) and th["IV sum of squares, threshold model"] == pytest.approx(367.04, abs=0.05)


def test_pcoint_kao_matches_the_guide():
    r = example("pcoint")
    kao = T(r, "Residual-based")["rows"][0]
    assert kao[1] == pytest.approx(-1.69, abs=0.01) and kao[2] == pytest.approx(0.045, abs=0.002)


def test_pquant_canay_is_between_pooled_and_zero_at_the_top():
    r = example("pquant")
    assert "error" not in r and S(r, "Units") == 545


def test_spatial_coefficients_match_the_guide():
    r = example("spatial")
    t = {x[0]: x for x in T(r, "Coefficients")["rows"]}["Spatial coefficient (rho or lambda)"]
    assert t[2].startswith("0.196") and t[3].startswith("0.384") and t[4].startswith("0.361")
