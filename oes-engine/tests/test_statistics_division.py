"""Regression tests for the OES Statistics Division engine and OMVP suite."""
import json
import pandas as pd
import pytest
from oes_engine import statistics_division as sd


def test_registry_has_31_unique_methods_across_seven_categories():
    ids = [item["id"] for item in sd.STAT_CATALOG]
    categories = {item["cat"] for item in sd.STAT_CATALOG}
    assert len(ids) == 31
    assert len(set(ids)) == 31
    assert len(categories) == 7
    assert set(ids) == set(sd.DISPATCH)


def test_all_31_estimators_execute_on_reproducible_dgp():
    for item in sd.STAT_CATALOG:
        model_id = item["id"]
        df, spec = sd._generate_omvp_dataset(model_id, seed=101)
        result = sd.DISPATCH[model_id](df, spec)
        assert result["method"]
        assert isinstance(result["summary"], dict)
        assert isinstance(result["table"], list)
        assert isinstance(result["diagnostics"], dict)
        assert isinstance(result["plot"], dict)
        json.dumps(sd._clean(result), allow_nan=False)


def test_all_31_omvp_guards_and_reproducibility():
    for item in sd.STAT_CATALOG:
        report = json.loads(sd.run_omvp_json(item["id"], n_reps=10))
        stages = report["stages"]
        assert stages["Stage_1_Numerical_Finiteness"] == "PASS", item["id"]
        assert stages["Stage_3_4_Monte_Carlo_Replications_Completed"] == 10, item["id"]
        assert stages["Stage_3_4_Monte_Carlo_Replications_Failed"] == 0, item["id"]
        assert stages["Monte_Carlo_Results_Finite"] == "PASS", item["id"]
        assert stages["Stage_7_Seeded_Reproducibility"] == "PASS", item["id"]
        assert stages["Stage_8_Edge_Case_Guard"] == "PASS", item["id"]


def test_power_calculator_accepts_its_one_row_reference_csv():
    csv_text = "note\nAnalytical non-central t power calculation"
    result = json.loads(sd.run_json("stat_power", csv_text, json.dumps({
        "effect_size": 0.5, "alpha": 0.05, "target_power": 0.8
    })))
    assert result["summary"]["Required_n_per_Group"] > 0


def test_power_calculator_rejects_invalid_inference_parameters():
    with pytest.raises(ValueError):
        sd.est_power(pd.DataFrame({"dummy": [1, 2, 3]}), {
            "effect_size": 0.5, "alpha": 1.0, "target_power": 0.8
        })


def test_kaplan_meier_handles_zero_terminal_survival_without_nan():
    df, spec = sd._generate_omvp_dataset("stat_km", seed=101)
    result = sd.est_km(df, spec)
    json.dumps(sd._clean(result), allow_nan=False)


def test_canonical_correlation_uses_supported_statsmodels_result_table():
    df, spec = sd._generate_omvp_dataset("stat_cca", seed=101)
    result = sd.est_cca(df, spec)
    assert result["table"]
    assert "Wilks_Lambda_1" in result["diagnostics"]


def test_cox_ph_summary_uses_supported_phreg_result_attributes():
    df, spec = sd._generate_omvp_dataset("stat_cox_ph", seed=101)
    result = sd.est_cox_ph(df, spec)
    assert result["summary"]["N"] == len(df)
    assert "Partial_Likelihood_AIC" in result["summary"]
    assert "Partial_Likelihood_BIC" in result["summary"]
