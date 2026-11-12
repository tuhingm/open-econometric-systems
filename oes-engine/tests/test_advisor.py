import pandas as pd
import pytest

from oes_engine.advisor import profile_data, recommend_models
from oes_engine.core import OESError


def test_profile_cross_section():
    df = pd.DataFrame({"y": [1, 2, 3, 4], "x": [2, 3, 4, 5], "sector": ["a", "a", "b", "b"]})
    p = profile_data(df)
    assert p["data_structure"] == "cross_section"
    assert p["nobs"] == 4
    assert p["numeric_columns"] == ["y", "x"]


def test_profile_panel_balance_and_duplicates():
    df = pd.DataFrame({"id": [1, 1, 2, 2], "year": [1, 2, 1, 2], "y": [1, 2, 3, 4]})
    p = profile_data(df, time="year", unit="id")
    assert p["data_structure"] == "panel"
    assert p["panel"]["n_units"] == 2
    assert p["panel"]["balanced"] is True
    assert p["panel"]["duplicate_unit_time_pairs"] == 0


def test_profile_rejects_missing_identifiers():
    df = pd.DataFrame({"y": [1, 2], "x": [2, 3]})
    with pytest.raises(OESError):
        profile_data(df, time="year")


def test_time_series_advisor_recommends_dynamic_models():
    df = pd.DataFrame({"date": range(1, 81), "y": range(80), "x": range(2, 82)})
    out = recommend_models(df, objective="explanatory", y="y", x=["x"], time="date")
    ids = {m["model_id"] for m in out["recommendations"]}
    assert "ardl" in ids
    assert "vecm" in ids
    assert "var" in ids
    assert out["preliminary_tests"]


def test_panel_advisor_recommends_fe():
    df = pd.DataFrame({"id": [1, 1, 2, 2, 3, 3], "year": [1, 2, 1, 2, 1, 2],
                       "y": [1, 2, 2, 3, 3, 4], "x": [2, 3, 3, 4, 4, 5]})
    out = recommend_models(df, y="y", x=["x"], unit="id", time="year")
    ids = {m["model_id"] for m in out["recommendations"]}
    assert "fe" in ids
    assert "re" in ids


def test_causal_advisor_requires_and_uses_design_roles():
    df = pd.DataFrame({"y": [1, 2, 2, 3, 4, 5], "d": [0, 0, 1, 1, 0, 1],
                       "z": [1, 2, 3, 4, 5, 6], "x": [2, 3, 2, 4, 3, 5]})
    out = recommend_models(df, objective="causal", y="y", treatment="d", instrument="z")
    ids = {m["model_id"] for m in out["recommendations"]}
    assert "iv" in ids
    assert "dml_iv" in ids


def test_unknown_objective_is_rejected():
    with pytest.raises(OESError):
        recommend_models(pd.DataFrame({"y": [1, 2], "x": [2, 3]}), objective="magic")
