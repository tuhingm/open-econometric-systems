"""Local projections: results match independent calculations (statsmodels for the time series, linearmodels for the panel)."""
import numpy as np
import pandas as pd
import pytest

import oes_engine as E
from test_models import SITE


def _read(rel):
    return pd.read_csv(SITE / rel)


def test_lp_matches_statsmodels_hac():
    import statsmodels.api as sm
    d = _read("oes/examples/us_macro.csv")
    d["d_tbill"] = d.tbilrate.diff()
    h = 2
    X = pd.DataFrame({"s": d.d_tbill, "inflation": d.inflation, "gdp_growth": d.gdp_growth})
    for k in range(1, 5):
        for c in ["unemp", "d_tbill", "inflation", "gdp_growth"]:
            X[f"{c}{k}"] = d[c].shift(k)
    D = pd.concat([d.unemp.shift(-h).rename("y"), X], axis=1).dropna()
    ref = sm.OLS(D.y, sm.add_constant(D.drop(columns="y"))).fit(cov_type="HAC", cov_kwds=dict(maxlags=h + 1, use_correction=True))
    out = E.lp.lp(d, dict(y="unemp", shock="d_tbill", contemp=["inflation", "gdp_growth"], lags=4, horizon=h))
    row = out["tables"][0]["rows"][h]
    assert row[1] == pytest.approx(ref.params["s"], abs=1e-5)
    assert row[2] == pytest.approx(ref.bse["s"], abs=1e-5)
    assert row[6] == len(D)


def test_lp_standardised_shock_scales_response():
    d = _read("oes/examples/us_macro.csv")
    d["d_tbill"] = d.tbilrate.diff()
    spec = dict(y="unemp", shock="d_tbill", lags=2, horizon=3)
    a = E.lp.lp(d, spec)["tables"][0]["rows"][1][1]
    b = E.lp.lp(d, dict(spec, scale="sd"))["tables"][0]["rows"][1][1]
    assert b == pytest.approx(a * d.d_tbill.std(), rel=1e-4)


def test_lp_iv_reports_first_stage_and_rejects_missing_roles():
    d = _read("oes/examples/us_macro.csv")
    d["d_tbill"] = d.tbilrate.diff()
    d["z"] = d.d_tbill.shift(1) + np.random.default_rng(0).normal(0, .3, len(d))
    out = E.lp.lp(d, dict(y="unemp", shock="d_tbill", instrument="z", lags=2, horizon=3))
    assert out["tables"][1]["title"].startswith("First stage")
    with pytest.raises(E.OESError):
        E.lp.lp(d, dict(y="unemp", shock="unemp"))
    with pytest.raises(E.OESError):
        E.lp.lp(d, dict(shock="d_tbill"))


@pytest.mark.parametrize("cov,bw,kw", [("clustered", None, "clustered"), ("dk", 3, "kernel")])
def test_panel_lp_matches_linearmodels(cov, bw, kw):
    pytest.importorskip("linearmodels")
    from linearmodels.panel import PanelOLS
    p = _read("panel-data-guide/examples/cigar.csv").sort_values(["state", "year"])
    p["d_price"] = p.groupby("state").log_price.diff()
    g, h = p.groupby("state"), 2
    p["y_h"] = g.log_sales.shift(-h)
    for k in (1, 2):
        p[f"y_l{k}"], p[f"d_l{k}"] = g.log_sales.shift(k), g.d_price.shift(k)
    d = p.dropna(subset=["y_h", "d_price", "y_l1", "y_l2", "d_l1", "d_l2"]).set_index(["state", "year"])
    opts = dict(cov_type="clustered", cluster_entity=True) if kw == "clustered" else dict(cov_type="kernel", kernel="bartlett", bandwidth=bw)
    ref = PanelOLS(d.y_h, d[["d_price", "y_l1", "y_l2", "d_l1", "d_l2"]], entity_effects=True, drop_absorbed=False).fit(**opts)
    out = E.lp.plp(p, dict(y="log_sales", shock="d_price", unit="state", time="year", lags=2, horizon=h, cov=cov, bandwidth=bw))
    row = out["tables"][0]["rows"][h]
    assert row[1] == pytest.approx(ref.params["d_price"], abs=1e-5)
    assert row[2] == pytest.approx(ref.std_errors["d_price"], abs=1e-5)


def test_panel_lp_handles_gaps_and_options():
    p = _read("panel-data-guide/examples/cigar.csv")
    p["d_price"] = p.sort_values(["state", "year"]).groupby("state").log_price.diff()
    gappy = p.drop(p.index[::17])                      # unbalanced panel with missing years
    for eff in ("unit", "twoway", "none"):
        out = E.lp.plp(gappy, dict(y="log_sales", shock="d_price", unit="state", time="year", lags=1, horizon=3, effects=eff))
        assert len(out["tables"][0]["rows"]) == 4 and out["charts"][0]["band"]
    with pytest.raises(E.OESError):
        E.lp.plp(p.iloc[:30], dict(y="log_sales", shock="d_price", unit="state", time="year"))
    with pytest.raises(E.OESError):
        E.lp.plp(p, dict(y="log_sales", shock="d_price", unit="state", time="year", cov="robust"))


def test_catalog_runs_both_guide_examples():
    for mid in ("lp", "plp"):
        m = E.BY_ID[mid]
        ex = m["example"]
        out = E.run(mid, (SITE / ex["file"]).read_text(), dict(ex["spec"], ops=ex.get("ops", [])))
        assert "error" not in out and out["charts"], (mid, out.get("error"))
