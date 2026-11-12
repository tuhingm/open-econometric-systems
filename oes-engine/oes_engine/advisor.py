"""Transparent, rule-based model selection for the OES estimation engine.

The advisor ranks only estimators that are actually registered in OES. It does not
claim that a data profile proves identification, stationarity, exogeneity, or
cointegration; those remain research-design and testing decisions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import OESError, read_csv


def _frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, str):
        return read_csv(data)
    raise OESError("Pass a pandas DataFrame or CSV text to the model advisor.")


def _normalise_objective(value: str) -> str:
    value = (value or "explanatory").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "explain": "explanatory", "explanation": "explanatory", "estimate": "explanatory",
        "predict": "prediction", "forecast": "forecasting", "forecasting": "forecasting",
        "causal_inference": "causal", "treatment_effect": "causal", "effect": "causal",
        "summary": "descriptive", "describe": "descriptive",
    }
    value = aliases.get(value, value)
    if value not in {"explanatory", "prediction", "forecasting", "causal", "descriptive"}:
        raise OESError("Objective must be explanatory, prediction, forecasting, causal, or descriptive.")
    return value


def profile_data(data: Any, time: str | None = None, unit: str | None = None) -> dict:
    """Return a compact, non-destructive profile of the supplied dataset."""
    df = _frame(data)
    if df.empty:
        raise OESError("The dataset has no observations.")
    if df.columns.duplicated().any():
        raise OESError("The dataset contains duplicate column names.")
    missing = {str(c): int(v) for c, v in df.isna().sum().items() if int(v) > 0}
    numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [str(c) for c in df.columns if c not in numeric]
    profile = {
        "nobs": int(len(df)),
        "nvars": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "missing_by_column": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "data_structure": "cross_section",
        "time_column": time if time in df.columns else None,
        "unit_column": unit if unit in df.columns else None,
        "panel": None,
    }
    if unit and unit not in df.columns:
        raise OESError(f"Unit identifier column '{unit}' was not found.")
    if time and time not in df.columns:
        raise OESError(f"Time column '{time}' was not found.")
    if unit and time:
        pairs = df[[unit, time]].dropna()
        counts = pairs.groupby(unit, dropna=False)[time].nunique()
        profile["data_structure"] = "panel"
        profile["panel"] = {
            "n_units": int(pairs[unit].nunique()),
            "n_periods": int(pairs[time].nunique()),
            "duplicate_unit_time_pairs": int(pairs.duplicated().sum()),
            "balanced": bool(len(counts) > 0 and counts.nunique() == 1),
            "periods_per_unit_min": int(counts.min()) if len(counts) else 0,
            "periods_per_unit_max": int(counts.max()) if len(counts) else 0,
        }
    elif time:
        profile["data_structure"] = "time_series"
        profile["time_is_monotonic"] = bool(df[time].is_monotonic_increasing)
        profile["time_unique"] = int(df[time].nunique(dropna=True))
    return profile


def recommend_models(
    data: Any,
    objective: str = "explanatory",
    y: str | None = None,
    x: list[str] | None = None,
    time: str | None = None,
    unit: str | None = None,
    treatment: str | None = None,
    instrument: str | None = None,
    running: str | None = None,
    integration: str = "unknown",
    cointegration: str = "unknown",
    max_recommendations: int = 8,
) -> dict:
    """Rank currently implemented OES estimators using explicit design rules.

    Parameters integration and cointegration are researcher-supplied conclusions,
    not inferred from the dataset. Use preliminary tests before setting them.
    """
    df = _frame(data)
    objective = _normalise_objective(objective)
    profile = profile_data(df, time=time, unit=unit)
    for key, value in (("outcome", y), ("treatment", treatment), ("instrument", instrument), ("running variable", running)):
        if value and value not in df.columns:
            raise OESError(f"The selected {key} column '{value}' was not found.")
    for value in (x or []):
        if value not in df.columns:
            raise OESError(f"Explanatory variable column '{value}' was not found.")
    if max_recommendations < 1:
        raise OESError("max_recommendations must be at least 1.")

    structure = profile["data_structure"]
    # Import lazily so this module remains independent of __init__ construction.
    from . import catalog
    available = {item["id"]: item for item in catalog()}

    ranked: list[dict] = []

    def add(model_id: str, score: int, reasons: list[str], conditions: list[str] | None = None):
        if model_id not in available:
            return
        if any(item["model_id"] == model_id for item in ranked):
            return
        item = available[model_id]
        ranked.append({
            "model_id": model_id,
            "name": item["name"],
            "group": item["group"],
            "score": int(score),
            "reasons": reasons,
            "conditions_to_check": conditions or [],
            "guide": item.get("guide"),
            "example": item.get("example"),
            "fields": item.get("fields", []),
        })

    tests: list[str] = []
    cautions: list[str] = []
    integration = (integration or "unknown").lower()
    cointegration = (cointegration or "unknown").lower()

    if profile["nobs"] < 30:
        cautions.append("The sample contains fewer than 30 observations; asymptotic inference and complex models may be unreliable.")
    if profile["missing_by_column"]:
        cautions.append("Missing values are present. Check whether deletion or imputation changes the estimation sample.")
    if y and y not in profile["numeric_columns"]:
        cautions.append("The selected outcome is not numeric; many registered estimators require a numeric outcome.")
    if structure == "panel" and profile["panel"]["duplicate_unit_time_pairs"]:
        cautions.append("Duplicate unit-time pairs exist. Resolve or explicitly aggregate them before panel estimation.")

    if structure == "time_series":
        tests += ["Plot each series and inspect frequency, gaps, outliers and deterministic components.",
                  "Assess integration order with ADF and PP tests; use KPSS as a complementary null-of-stationarity test.",
                  "Check structural breaks before interpreting conventional unit-root and cointegration tests."]
        if objective in {"prediction", "forecasting"}:
            add("var", 96, ["The data are time ordered and the objective is forecasting.", "A VAR models joint dynamics when multiple endogenous series are selected."],
                ["Choose variables and lag order using the research design and information criteria.", "Use rolling-origin or expanding-window validation; do not shuffle time observations."])
            add("bvar", 91, ["A Bayesian VAR can regularize a multivariate forecasting system, especially when parameters are numerous relative to observations."],
                ["Specify and report the prior; assess prior sensitivity and forecast performance out of sample."])
            add("ms_ar", 72, ["Regime-switching autoregression is a candidate when there is a substantive hypothesis of discrete changes in dynamics."],
                ["Compare against non-switching benchmarks; check regime identification and transition probabilities."])
            add("lasso", 66, ["Regularized regression can be useful for a larger predictor set."],
                ["Use time-ordered validation and construct lags using only information available at each forecast origin."])
        elif objective == "causal":
            add("iv_gmm", 84, ["The current time-series catalogue includes IV/GMM estimation."],
                ["State the identifying restrictions and assess instrument relevance and exclusion; time-series dependence requires appropriate covariance treatment."])
            add("ols", 60, ["OLS is available as a baseline conditional association model."],
                ["A causal interpretation requires a defensible identification strategy and appropriate treatment of serial correlation."])
            cautions.append("Time-series causal identification is design-specific; the advisor will not infer exogeneity or a valid intervention from column patterns.")
        else:
            add("ardl", 94, ["ARDL is a candidate for a single-equation dynamic relationship when regressors are a mixture of I(0) and I(1), but none is I(2)."],
                ["Run unit-root tests first.", "The bounds-test case and deterministic terms must match the specification.", "Do not use the standard bounds procedure if any variable is I(2)."])
            add("vecm", 91 if cointegration in {"yes", "true", "cointegrated"} else 76,
                ["VECM is appropriate for a system of I(1) variables with cointegration."],
                ["Establish integration order and select cointegration rank using Johansen procedures.", "Specify deterministic terms and lag order."])
            add("var", 88, ["VAR is a system model for joint dynamics; it is a candidate for stationary variables or appropriately transformed series."],
                ["Check stability, lag order, residual serial correlation and variable transformations."])
            add("bvar", 82, ["BVAR provides prior-based shrinkage for multivariate dynamic systems."],
                ["Report prior hyperparameters and sensitivity; compare with an unrestricted VAR where feasible."])
            add("ols", 70, ["OLS is useful as a baseline conditional relationship."],
                ["Address serial correlation, nonstationarity and spurious regression risk before interpretation."])
            add("ms_ar", 62, ["Markov-switching AR is a candidate if a regime-change mechanism is theoretically motivated."],
                ["Test whether regime complexity is supported; inspect convergence and regime probabilities."])
        if integration in {"i1", "i(1)", "1"} and cointegration in {"no", "false", "not_cointegrated"}:
            cautions.append("The supplied integration assessment indicates I(1) variables without cointegration; estimate in differences or reconsider the variable system.")
        if integration in {"i2", "i(2)", "2"}:
            cautions.append("At least one variable is reported as I(2); standard ARDL bounds testing and conventional I(1) VECM workflows are not appropriate.")

    elif structure == "panel":
        tests += ["Check the unit-time key for uniqueness and inspect panel balance and gaps.",
                  "Compare pooled, fixed-effects and random-effects specifications using the assumptions and relevant specification tests.",
                  "Assess cross-sectional dependence and serial correlation; choose covariance estimators accordingly."]
        if objective in {"prediction", "forecasting"}:
            add("fe", 82, ["Fixed effects can provide a panel baseline while controlling for time-invariant unit heterogeneity."],
                ["For prediction, use unit-aware or time-aware validation that prevents leakage across the intended deployment setting."])
            add("rf", 78, ["Random forests can capture nonlinear predictor relationships."],
                ["Define grouped or time-based validation; do not randomly split dependent panel observations."])
            add("xgb", 76, ["Gradient-boosted trees are a flexible nonlinear prediction benchmark."],
                ["Confirm XGBoost is available in the browser runtime and use leakage-safe validation."])
        elif objective == "causal":
            if treatment and time and unit:
                add("did", 98, ["Outcome, treatment, unit and time roles are specified for a panel intervention design."],
                    ["Assess parallel trends, anticipation, treatment timing and spillovers.", "For staggered adoption, inspect treatment-effect heterogeneity and estimator assumptions."])
                add("staggered", 96, ["The treatment and panel identifiers support a staggered-adoption DiD candidate."],
                    ["Check group-time support, not-yet-treated comparison validity and parallel trends."])
                add("scm", 79, ["Synthetic control is a candidate for an aggregate treated unit with a suitable donor pool."],
                    ["Check pre-treatment fit, donor-pool contamination and placebo/permutation inference."])
                add("gsc", 77, ["Generalized synthetic control can accommodate latent interactive factors in panel settings."],
                    ["Assess factor selection, treatment timing and pre-treatment fit."])
            if instrument:
                add("iv", 89, ["An instrument column is specified for an instrumental-variable design."],
                    ["Justify relevance and exclusion restrictions; report first-stage strength and weak-instrument robust inference where available."])
            add("dml_plr", 70, ["Double machine learning is available for partially linear treatment-effect settings."],
                ["Specify treatment and controls; justify conditional exchangeability and cross-fitting assumptions."])
        else:
            add("fe", 96, ["Panel identifiers are specified; fixed effects control for time-invariant unit heterogeneity."],
                ["Use time effects where warranted; select clustered or dependence-robust covariance based on the error structure."])
            add("re", 84, ["Random effects is a candidate when unit effects are uncorrelated with included regressors under the model assumptions."],
                ["State the orthogonality assumption and compare with fixed effects."])
            add("pols", 78, ["Pooled OLS is a useful baseline."],
                ["Account for within-unit dependence and omitted unit heterogeneity."])
            add("fedk", 80, ["Fixed effects with Driscoll-Kraay errors is available for settings with cross-sectional and temporal dependence."],
                ["Check the time dimension and bandwidth sensitivity."])
            add("diff_gmm", 74, ["Difference GMM is a candidate for dynamic panels with endogenous lagged outcomes and suitable instruments."],
                ["Inspect AR(1)/AR(2), instrument count and Hansen/Sargan diagnostics; instrument proliferation can invalidate inference."])
            add("pmg", 72, ["Panel ARDL estimators are available when a long-run relationship with heterogeneous short-run dynamics is motivated."],
                ["Check integration order, cross-sectional dependence and long-run homogeneity assumptions."])
            add("csardl", 70, ["CS-ARDL is a candidate when common correlated effects are needed to address unobserved common factors."],
                ["Assess cross-sectional dependence and ensure adequate N and T dimensions."])
            add("mg", 68, ["Mean-group/CCE mean-group estimators allow heterogeneous slope coefficients."],
                ["Check cross-sectional dependence and the adequacy of time observations per unit."])
    else:
        tests += ["Inspect outcome scale, predictor coding, missingness and influential observations.",
                  "Define the estimand and identification assumptions before choosing a causal estimator.",
                  "Use held-out or cross-validated evaluation for prediction; avoid evaluating on the training sample."]
        if objective in {"prediction", "forecasting"}:
            add("lasso", 91, ["LASSO performs variable shrinkage and selection in high-dimensional prediction settings."],
                ["Standardize predictors within the training folds and tune regularization using cross-validation."])
            add("ridge", 88, ["Ridge regression stabilizes estimates under multicollinearity."],
                ["Tune the penalty using cross-validation; retain all predictors."])
            add("enet", 86, ["Elastic net combines L1 and L2 regularization."],
                ["Tune alpha and the mixing parameter using cross-validation."])
            add("rf", 84, ["Random forest captures nonlinearities and interactions with limited functional-form assumptions."],
                ["Tune tree depth and leaf size; evaluate on a held-out sample."])
            add("xgb", 82, ["Gradient boosting is a flexible nonlinear prediction benchmark."],
                ["Tune hyperparameters and assess calibration as well as predictive error."])
        elif objective == "causal":
            if instrument:
                add("iv", 98, ["An instrument is specified, making IV/2SLS a candidate."],
                    ["The instrument must be relevant and satisfy the exclusion restriction; report first-stage diagnostics."])
                add("dml_iv", 90, ["Double ML with IV is available for high-dimensional controls."],
                    ["Requires valid instruments, cross-fitting and adequate overlap/support."])
            elif running:
                add("rdd", 98, ["A running variable is specified for a regression-discontinuity design."],
                    ["Specify the cutoff; assess manipulation, continuity, bandwidth sensitivity and local polynomial order."])
            elif treatment:
                add("dml_plr", 88, ["A treatment variable is specified; DML can adjust flexibly for observed controls."],
                    ["Conditional exchangeability, overlap and cross-fitting assumptions must be defended."])
                add("dml_irm", 85, ["DML interactive regression is a candidate for binary treatment."],
                    ["Treatment must be coded 0/1; assess overlap and conditional exchangeability."])
            else:
                cautions.append("Specify the treatment, instrument or running variable to identify a suitable causal design.")
            add("ols", 55, ["OLS can serve as a descriptive baseline, not automatic causal identification."],
                ["Do not interpret the coefficient causally without an identification argument."])
        else:
            add("ols", 94, ["OLS is an interpretable baseline for a continuous outcome and numeric regressors."],
                ["Check functional form, heteroskedasticity, influential observations and dependence."])
            add("bayes_reg", 76, ["Bayesian linear regression is available when prior information is part of the analysis."],
                ["Report prior choices and posterior sensitivity."])
            add("iv", 70, ["IV/2SLS is available if some regressors are endogenous and valid excluded instruments exist."],
                ["Do not use IV without a substantive exclusion restriction and relevance argument."])

    # Enforce the current executable catalogue, then sort deterministically.
    ranked.sort(key=lambda item: (-item["score"], item["model_id"]))
    selected = ranked[: int(max_recommendations)]
    return {
        "advisor_version": "0.1.0",
        "objective": objective,
        "profile": profile,
        "selected_roles": {"outcome": y, "predictors": x or [], "time": time, "unit": unit,
                           "treatment": treatment, "instrument": instrument, "running": running},
        "recommendations": selected,
        "preliminary_tests": list(dict.fromkeys(tests)),
        "cautions": list(dict.fromkeys(cautions)),
        "methodology_note": (
            "Recommendations are transparent screening suggestions, not proof of identification or model validity. "
            "Review the stated conditions, run the required tests, and document the final specification."
        ),
        "available_model_count": len(available),
    }
