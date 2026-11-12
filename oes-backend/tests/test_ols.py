import numpy as np
import pytest
from oes_api.estimators.ols import fit_ols

def test_ols_recovers_linear_coefficients():
    x = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
    y = 2.0 + 3.0 * x[:, 1]
    result = fit_ols(y, x, variable_names=["const", "trend"])
    assert result["coefficients"][0]["estimate"] == pytest.approx(2.0)
    assert result["coefficients"][1]["estimate"] == pytest.approx(3.0)
    assert result["r_squared"] == pytest.approx(1.0)

def test_ols_rejects_rank_deficiency():
    x = np.column_stack([np.ones(5), np.ones(5)])
    with pytest.raises(ValueError, match="rank deficient"):
        fit_ols(np.arange(5, dtype=float), x)

def test_ols_rejects_nonfinite_values():
    x = np.column_stack([np.ones(4), [1.0, 2.0, np.nan, 4.0]])
    with pytest.raises(ValueError, match="finite"):
        fit_ols(np.arange(4, dtype=float), x)
