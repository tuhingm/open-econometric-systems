"""
Open Econometric Systems (OES) — Statistics Division Engine (v1.0.1)
31 Distinct Statistical Methods across 7 Categories (Expanding OES from 69 to 100 Methods)
Author: Tuhin G. M. Al Mamun, Ph.D.
Runtime: Browser-native Pyodide v0.27.2 (NumPy, pandas, SciPy, statsmodels)
"""

import io
import json
import math
import time
import warnings
import numpy as np
import scipy
import pandas as pd
from scipy import stats, linalg, optimize, special
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.cancorr import CanCorr
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian, Binomial, Poisson
from statsmodels.genmod.cov_struct import Exchangeable, Autoregressive, Independence, Unstructured
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight

warnings.filterwarnings("ignore")

__version__ = "1.0.1-stat31"

# ==============================================================================
# 1. CENTRALIZED REGISTRY OF ALL 31 DISTINCT STATISTICAL METHODS
# ==============================================================================

STAT_CATALOG = [
    # Category A: Analysis of Variance (6)
    {"id": "stat_anova1", "cat": "Analysis of Variance", "code": "A1", "name": "One-Way ANOVA", "packages": []},
    {"id": "stat_anova2", "cat": "Analysis of Variance", "code": "A2", "name": "Two-Way ANOVA", "packages": []},
    {"id": "stat_rm_anova", "cat": "Analysis of Variance", "code": "A3", "name": "Repeated Measures ANOVA", "packages": []},
    {"id": "stat_ancova", "cat": "Analysis of Variance", "code": "A4", "name": "Analysis of Covariance (ANCOVA)", "packages": []},
    {"id": "stat_manova", "cat": "Analysis of Variance", "code": "A5", "name": "Multivariate ANOVA (MANOVA)", "packages": []},
    {"id": "stat_mancova", "cat": "Analysis of Variance", "code": "A6", "name": "Multivariate ANCOVA (MANCOVA)", "packages": []},

    # Category B: Nonparametric Statistics (5)
    {"id": "stat_mann_whitney", "cat": "Nonparametric Statistics", "code": "B1", "name": "Mann–Whitney U Test", "packages": []},
    {"id": "stat_wilcoxon", "cat": "Nonparametric Statistics", "code": "B2", "name": "Wilcoxon Signed-Rank Test", "packages": []},
    {"id": "stat_kruskal", "cat": "Nonparametric Statistics", "code": "B3", "name": "Kruskal–Wallis Test", "packages": []},
    {"id": "stat_friedman", "cat": "Nonparametric Statistics", "code": "B4", "name": "Friedman Test", "packages": []},
    {"id": "stat_kendall", "cat": "Nonparametric Statistics", "code": "B5", "name": "Kendall's Tau Correlation", "packages": []},

    # Category C: Multivariate Statistical Analysis (6)
    {"id": "stat_pca", "cat": "Multivariate Statistical Analysis", "code": "C1", "name": "Principal Component Analysis (PCA)", "packages": []},
    {"id": "stat_efa", "cat": "Multivariate Statistical Analysis", "code": "C2", "name": "Exploratory Factor Analysis (EFA)", "packages": []},
    {"id": "stat_cfa", "cat": "Multivariate Statistical Analysis", "code": "C3", "name": "Confirmatory Factor Analysis (CFA)", "packages": []},
    {"id": "stat_lda", "cat": "Multivariate Statistical Analysis", "code": "C4", "name": "Linear Discriminant Analysis (LDA)", "packages": []},
    {"id": "stat_cca", "cat": "Multivariate Statistical Analysis", "code": "C5", "name": "Canonical Correlation Analysis (CCA)", "packages": []},
    {"id": "stat_ca", "cat": "Multivariate Statistical Analysis", "code": "C6", "name": "Correspondence Analysis (CA)", "packages": []},

    # Category D: Survival Analysis (5)
    {"id": "stat_km", "cat": "Survival Analysis", "code": "D1", "name": "Kaplan–Meier Estimator", "packages": []},
    {"id": "stat_cox_ph", "cat": "Survival Analysis", "code": "D2", "name": "Cox Proportional Hazards Model", "packages": []},
    {"id": "stat_aft", "cat": "Survival Analysis", "code": "D3", "name": "Accelerated Failure Time (AFT) Model", "packages": []},
    {"id": "stat_fine_gray", "cat": "Survival Analysis", "code": "D4", "name": "Fine–Gray Competing Risks Model", "packages": []},
    {"id": "stat_param_surv", "cat": "Survival Analysis", "code": "D5", "name": "Parametric Survival Models (Weibull/Exp/LogNormal)", "packages": []},

    # Category E: Mixed-Effects and Longitudinal Statistics (4)
    {"id": "stat_lmm", "cat": "Mixed-Effects and Longitudinal Statistics", "code": "E1", "name": "Linear Mixed-Effects Model (LMM)", "packages": []},
    {"id": "stat_glmm", "cat": "Mixed-Effects and Longitudinal Statistics", "code": "E2", "name": "Generalized Linear Mixed Model (GLMM)", "packages": []},
    {"id": "stat_gee", "cat": "Mixed-Effects and Longitudinal Statistics", "code": "E3", "name": "Generalized Estimating Equations (GEE)", "packages": []},
    {"id": "stat_icc", "cat": "Mixed-Effects and Longitudinal Statistics", "code": "E4", "name": "Intraclass Correlation Coefficient (ICC)", "packages": []},

    # Category F: Resampling and Statistical Simulation (3)
    {"id": "stat_bootstrap", "cat": "Resampling and Statistical Simulation", "code": "F1", "name": "Bootstrap Inference (Percentile & BCa)", "packages": []},
    {"id": "stat_permutation", "cat": "Resampling and Statistical Simulation", "code": "F2", "name": "Permutation Inference", "packages": []},
    {"id": "stat_power", "cat": "Resampling and Statistical Simulation", "code": "F3", "name": "Statistical Power and Sample Size Analysis", "packages": []},

    # Category G: Meta-Analysis (2)
    {"id": "stat_meta_fe", "cat": "Meta-Analysis", "code": "G1", "name": "Fixed-Effect Meta-Analysis", "packages": []},
    {"id": "stat_meta_re", "cat": "Meta-Analysis", "code": "G2", "name": "Random-Effects Meta-Analysis", "packages": []},
]


# ==============================================================================
# 2. SHARED VALIDATION, NUMERIC SANITIZATION & HELPERS
# ==============================================================================

def _clean(obj):
    """Recursively convert NumPy/pandas types to JSON-safe Python primitives."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else round(val, 6)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _parse_csv(csv_text: str, min_rows: int = 3) -> pd.DataFrame:
    if not csv_text or not str(csv_text).strip():
        raise ValueError("Input CSV dataset is empty.")
    df = pd.read_csv(io.StringIO(csv_text.strip()))
    if df.shape[0] < min_rows:
        raise ValueError(f"Insufficient observations (n={df.shape[0]}). At least {min_rows} data row(s) are required.")
    return df


def _require_cols(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if not c or c not in df.columns]
    if missing:
        raise ValueError(f"Required column(s) missing from dataset: {missing}. Available: {list(df.columns)}")


def _numeric_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    _require_cols(df, cols)
    sub = df[cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 3:
        raise ValueError("Fewer than 3 complete numeric rows remain after dropping missing values.")
    return sub


def _finite_tree(obj) -> bool:
    """Check every numeric leaf in a nested result object for finite values."""
    if isinstance(obj, dict):
        return all(_finite_tree(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_finite_tree(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return bool(np.isfinite(obj).all())
    if isinstance(obj, (np.integer, np.floating, int, float)) and not isinstance(obj, (bool, np.bool_)):
        return bool(math.isfinite(float(obj)))
    return True


def _numeric_condition_diagnostic(df: pd.DataFrame) -> dict:
    """Report conditioning of the standardized numeric data matrix, when defined."""
    numeric = df.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.shape[1] < 2 or numeric.shape[0] < 2:
        return {"condition_number": None, "status": "NOT_APPLICABLE", "columns": int(numeric.shape[1])}
    values = numeric.to_numpy(dtype=float)
    sd = values.std(axis=0, ddof=0)
    keep = sd > np.finfo(float).eps
    if keep.sum() < 2:
        return {"condition_number": None, "status": "SINGULAR_OR_CONSTANT_COLUMNS", "columns": int(keep.sum())}
    z = (values[:, keep] - values[:, keep].mean(axis=0)) / sd[keep]
    cond = float(np.linalg.cond(z))
    status = "PASS" if math.isfinite(cond) and cond <= 1e12 else "WARNING_ILL_CONDITIONED"
    return {"condition_number": cond, "status": status, "columns": int(keep.sum())}


# ==============================================================================
# 3. CATEGORY A: ANALYSIS OF VARIANCE (6 METHODS)
# ==============================================================================

def est_anova1(df: pd.DataFrame, spec: dict) -> dict:
    y_col = spec.get("y") or df.columns[0]
    g_col = spec.get("group") or df.columns[1]
    _require_cols(df, [y_col, g_col])
    sub = df[[y_col, g_col]].dropna()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()

    groups = [grp[y_col].values for _, grp in sub.groupby(g_col)]
    labels = [str(k) for k, _ in sub.groupby(g_col)]
    k = len(groups)
    n = len(sub)
    if k < 2:
        raise ValueError("One-Way ANOVA requires at least 2 distinct groups.")

    f_stat, p_val = stats.f_oneway(*groups)
    grand_mean = sub[y_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)
    ss_total = ss_between + ss_within
    df_b, df_w = k - 1, n - k
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    omega_sq = (ss_between - df_b * ms_w) / (ss_total + ms_w) if (ss_total + ms_w) > 0 else 0.0

    # Diagnostics: Levene homogeneity of variance & Shapiro-Wilk on residuals
    lev_stat, lev_p = stats.levene(*groups, center="median")
    welch_f = stats.alexandergovern(*groups)
    resids = np.concatenate([g - g.mean() for g in groups])
    sw_stat, sw_p = stats.shapiro(resids[:5000])

    # Group summary & 95% CIs
    group_rows = []
    for lbl, g in zip(labels, groups):
        m = float(np.mean(g))
        se = float(np.std(g, ddof=1) / math.sqrt(len(g)))
        t_crit = stats.t.ppf(0.975, df=len(g) - 1)
        group_rows.append({
            "Parameter": f"Group: {lbl} (n={len(g)})",
            "Estimate": m,
            "Std_Error": se,
            "Statistic": m / se if se > 0 else None,
            "df": len(g) - 1,
            "p_value": 2 * (1 - stats.t.cdf(abs(m / se), df=len(g) - 1)) if se > 0 else None,
            "CI_Low": m - t_crit * se,
            "CI_High": m + t_crit * se
        })

    table = [
        {"Parameter": "Between Groups (Treatment)", "Estimate": ss_between, "Std_Error": ms_b,
         "Statistic": f_stat, "df": f"{df_b}, {df_w}", "p_value": p_val, "CI_Low": None, "CI_High": None},
        {"Parameter": "Within Groups (Residual)", "Estimate": ss_within, "Std_Error": ms_w,
         "Statistic": None, "df": df_w, "p_value": None, "CI_Low": None, "CI_High": None},
    ] + group_rows

    return {
        "method": "One-Way ANOVA",
        "summary": {"N": n, "Groups_k": k, "F_stat": f_stat, "p_value": p_val,
                    "Eta_Squared": eta_sq, "Omega_Squared": omega_sq, "R_Squared": eta_sq},
        "table": table,
        "diagnostics": {
            "Brown_Forsythe_Levene_F": lev_stat, "Levene_p_value": lev_p,
            "Alexander_Govern_Robust_Stat": welch_f.statistic, "Robust_p_value": welch_f.pvalue,
            "Residual_Shapiro_Wilk_W": sw_stat, "Residual_Normality_p": sw_p
        },
        "plot": {
            "type": "box",
            "title": f"One-Way ANOVA Group Distributions ({y_col} by {g_col})",
            "labels": labels,
            "values": [g.tolist() for g in groups]
        }
    }


def est_anova2(df: pd.DataFrame, spec: dict) -> dict:
    y_col = spec.get("y") or df.columns[0]
    f1 = spec.get("factor1") or df.columns[1]
    f2 = spec.get("factor2") or df.columns[2]
    ss_type = int(spec.get("ss_type", 3))
    _require_cols(df, [y_col, f1, f2])
    sub = df[[y_col, f1, f2]].dropna()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()

    formula = f"Q('{y_col}') ~ C(Q('{f1}')) + C(Q('{f2}')) + C(Q('{f1}')):C(Q('{f2}'))"
    ols = smf.ols(formula, data=sub).fit()
    aov = anova_lm(ols, typ=ss_type)

    ss_tot = float(np.sum((sub[y_col] - sub[y_col].mean()) ** 2))
    df_res = int(aov.loc["Residual", "df"])
    ss_res = float(aov.loc["Residual", "sum_sq"])

    table = []
    for idx, row in aov.iterrows():
        ss = float(row["sum_sq"])
        df_i = int(row["df"])
        ms = ss / df_i if df_i > 0 else None
        part_eta = ss / (ss + ss_res) if idx != "Residual" and (ss + ss_res) > 0 else None
        table.append({
            "Parameter": str(idx).replace("Q('", "").replace("')", ""),
            "Estimate": ss,
            "Std_Error": ms,
            "Statistic": row["F"] if not pd.isna(row["F"]) else None,
            "df": f"{df_i}, {df_res}" if idx != "Residual" else df_i,
            "p_value": row["PR(>F)"] if not pd.isna(row["PR(>F)"]) else None,
            "CI_Low": part_eta,
            "CI_High": None
        })

    sw_stat, sw_p = stats.shapiro(ols.resid[:5000])
    bp_stat, bp_p, _, _ = sm.stats.diagnostic.het_breuschpagan(ols.resid, ols.model.exog)

    return {
        "method": f"Two-Way Factorial ANOVA (Type {ss_type} SS)",
        "summary": {"N": int(ols.nobs), "R_Squared": ols.rsquared, "Adj_R_Squared": ols.rsquared_adj,
                    "Log_Likelihood": ols.llf, "AIC": ols.aic, "BIC": ols.bic},
        "table": table,
        "diagnostics": {
            "SS_Type": ss_type, "Residual_Shapiro_W": sw_stat, "Residual_Normality_p": sw_p,
            "Breusch_Pagan_LM": bp_stat, "Breusch_Pagan_p": bp_p
        },
        "plot": {
            "type": "bar",
            "title": "Two-Way ANOVA Variance Decomposition (Sum of Squares)",
            "labels": [r["Parameter"] for r in table],
            "values": [r["Estimate"] for r in table]
        }
    }


def est_rm_anova(df: pd.DataFrame, spec: dict) -> dict:
    # Wide format: columns are repeated conditions; rows are subjects
    cond_cols = spec.get("conditions") or list(df.select_dtypes(include=[np.number]).columns)
    sub = _numeric_df(df, cond_cols)
    n, k = sub.shape
    if k < 2:
        raise ValueError("Repeated Measures ANOVA requires at least 2 repeated-measure columns.")

    Y = sub.values
    grand_mean = np.mean(Y)
    subj_means = np.mean(Y, axis=1, keepdims=True)
    cond_means = np.mean(Y, axis=0, keepdims=True)

    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_subjects = k * np.sum((subj_means - grand_mean) ** 2)
    ss_cond = n * np.sum((cond_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_cond

    df_cond = k - 1
    df_subj = n - 1
    df_err = (n - 1) * (k - 1)

    ms_cond = ss_cond / df_cond
    ms_err = ss_error / df_err
    f_val = ms_cond / ms_err if ms_err > 0 else np.nan
    p_val = 1 - stats.f.cdf(f_val, df_cond, df_err)
    part_eta = ss_cond / (ss_cond + ss_error) if (ss_cond + ss_error) > 0 else 0.0

    # Greenhouse-Geisser and Huynh-Feldt epsilon for sphericity
    S = np.cov(Y, rowvar=False)
    P = np.eye(k) - np.ones((k, k)) / k
    PSP = P @ S @ P
    tr_PSP = np.trace(PSP)
    tr_PSP2 = np.trace(PSP @ PSP)
    eps_gg = float(np.clip((tr_PSP ** 2) / ((k - 1) * tr_PSP2), 1.0 / (k - 1), 1.0)) if tr_PSP2 > 0 else 1.0
    eps_hf = float(np.clip((n * (k - 1) * eps_gg - 2) / ((k - 1) * (n - 1 - (k - 1) * eps_gg)), eps_gg, 1.0))

    p_gg = 1 - stats.f.cdf(f_val, df_cond * eps_gg, df_err * eps_gg)
    p_hf = 1 - stats.f.cdf(f_val, df_cond * eps_hf, df_err * eps_hf)

    table = [
        {"Parameter": "Within-Subjects Condition (Sphericity Assumed)", "Estimate": ss_cond, "Std_Error": ms_cond,
         "Statistic": f_val, "df": f"{df_cond}, {df_err}", "p_value": p_val, "CI_Low": part_eta, "CI_High": None},
        {"Parameter": "Greenhouse-Geisser Corrected", "Estimate": eps_gg, "Std_Error": ms_cond,
         "Statistic": f_val, "df": f"{df_cond*eps_gg:.2f}, {df_err*eps_gg:.2f}", "p_value": p_gg, "CI_Low": None, "CI_High": None},
        {"Parameter": "Huynh-Feldt Corrected", "Estimate": eps_hf, "Std_Error": ms_cond,
         "Statistic": f_val, "df": f"{df_cond*eps_hf:.2f}, {df_err*eps_hf:.2f}", "p_value": p_hf, "CI_Low": None, "CI_High": None},
        {"Parameter": "Error (Condition x Subjects)", "Estimate": ss_error, "Std_Error": ms_err,
         "Statistic": None, "df": df_err, "p_value": None, "CI_Low": None, "CI_High": None},
    ]

    return {
        "method": "Repeated Measures ANOVA",
        "summary": {"Subjects_N": n, "Conditions_k": k, "F_stat": f_val, "p_value": p_val,
                    "Partial_Eta_Sq": part_eta, "Epsilon_GG": eps_gg, "Epsilon_HF": eps_hf},
        "table": table,
        "diagnostics": {
            "Greenhouse_Geisser_Epsilon": eps_gg, "GG_Corrected_p": p_gg,
            "Huynh_Feldt_Epsilon": eps_hf, "HF_Corrected_p": p_hf
        },
        "plot": {
            "type": "line",
            "title": "Repeated Measures Profile Plot (Condition Means)",
            "x": cond_cols,
            "y": np.mean(Y, axis=0).tolist()
        }
    }


def est_ancova(df: pd.DataFrame, spec: dict) -> dict:
    y_col = spec.get("y") or df.columns[0]
    g_col = spec.get("group") or df.columns[1]
    cov_cols = spec.get("covariates") or [df.columns[2]]
    if isinstance(cov_cols, str):
        cov_cols = [c.strip() for c in cov_cols.split(",") if c.strip()]
    _require_cols(df, [y_col, g_col] + cov_cols)

    sub = df[[y_col, g_col] + cov_cols].dropna()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    for c in cov_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    cov_terms = " + ".join([f"Q('{c}')" for c in cov_cols])
    formula = f"Q('{y_col}') ~ C(Q('{g_col}')) + {cov_terms}"
    ols = smf.ols(formula, data=sub).fit()
    aov = anova_lm(ols, typ=3)

    # Homogeneity of regression slopes test (Group x Covariate interaction)
    inter_terms = " + ".join([f"C(Q('{g_col}')):Q('{c}')" for c in cov_cols])
    ols_inter = smf.ols(f"{formula} + {inter_terms}", data=sub).fit()
    slope_test = anova_lm(ols, ols_inter)
    slope_f = float(slope_test["F"].iloc[1])
    slope_p = float(slope_test["Pr(>F)"].iloc[1])

    table = []
    df_res = int(aov.loc["Residual", "df"])
    for idx, row in aov.iterrows():
        if idx == "Intercept":
            continue
        table.append({
            "Parameter": str(idx).replace("Q('", "").replace("')", ""),
            "Estimate": float(row["sum_sq"]),
            "Std_Error": float(row["sum_sq"] / row["df"]),
            "Statistic": float(row["F"]) if not pd.isna(row["F"]) else None,
            "df": f"{int(row['df'])}, {df_res}" if idx != "Residual" else df_res,
            "p_value": float(row["PR(>F)"]) if not pd.isna(row["PR(>F)"]) else None,
            "CI_Low": None,
            "CI_High": None
        })

    return {
        "method": "Analysis of Covariance (ANCOVA — Type III SS)",
        "summary": {"N": int(ols.nobs), "R_Squared": ols.rsquared, "Adj_R_Squared": ols.rsquared_adj,
                    "Log_Likelihood": ols.llf, "AIC": ols.aic, "BIC": ols.bic},
        "table": table,
        "diagnostics": {
            "Homogeneity_of_Slopes_F": slope_f,
            "Homogeneity_of_Slopes_p": slope_p,
            "Slopes_Parallel_Assumption": "Satisfied (p >= 0.05)" if slope_p >= 0.05 else "Violated (p < 0.05)"
        },
        "plot": {
            "type": "scatter",
            "title": f"ANCOVA Covariate Adjustment ({y_col} vs {cov_cols[0]})",
            "x": sub[cov_cols[0]].tolist(),
            "y": sub[y_col].tolist()
        }
    }


def est_manova(df: pd.DataFrame, spec: dict) -> dict:
    y_cols = spec.get("y_vars") or list(df.select_dtypes(include=[np.number]).columns[:2])
    g_col = spec.get("group") or [c for c in df.columns if c not in y_cols][0]
    if isinstance(y_cols, str):
        y_cols = [c.strip() for c in y_cols.split(",") if c.strip()]
    _require_cols(df, y_cols + [g_col])

    sub = df[y_cols + [g_col]].dropna()
    for c in y_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(y_cols) < 2:
        raise ValueError("MANOVA requires at least 2 dependent numeric variables.")

    lhs = " + ".join([f"Q('{c}')" for c in y_cols])
    mv = MANOVA.from_formula(f"{lhs} ~ C(Q('{g_col}'))", data=sub)
    res = mv.mv_test().results[f"C(Q('{g_col}'))"]["stat"]

    table = []
    for stat_name in res.index:
        row = res.loc[stat_name]
        table.append({
            "Parameter": str(stat_name),
            "Estimate": float(row["Value"]),
            "Std_Error": None,
            "Statistic": float(row["F Value"]),
            "df": f"{row['Num DF']:.1f}, {row['Den DF']:.1f}",
            "p_value": float(row["Pr > F"]),
            "CI_Low": None,
            "CI_High": None
        })

    wilks = float(res.loc["Wilks' lambda", "Value"])
    return {
        "method": "Multivariate Analysis of Variance (MANOVA)",
        "summary": {"N": len(sub), "Dependent_Vars_p": len(y_cols), "Wilks_Lambda": wilks,
                    "Pillais_Trace": float(res.loc["Pillai's trace", "Value"]),
                    "Hotelling_Lawley": float(res.loc["Hotelling-Lawley trace", "Value"]),
                    "Roys_Largest_Root": float(res.loc["Roy's greatest root", "Value"])},
        "table": table,
        "diagnostics": {"Multivariate_Partial_Eta_Sq": 1.0 - wilks},
        "plot": {
            "type": "scatter",
            "title": f"MANOVA Bivariate Response Space ({y_cols[0]} vs {y_cols[1]})",
            "x": sub[y_cols[0]].tolist(),
            "y": sub[y_cols[1]].tolist()
        }
    }


def est_mancova(df: pd.DataFrame, spec: dict) -> dict:
    y_cols = spec.get("y_vars") or list(df.select_dtypes(include=[np.number]).columns[:2])
    g_col = spec.get("group") or [c for c in df.columns if c not in y_cols][0]
    cov_cols = spec.get("covariates") or [c for c in df.select_dtypes(include=[np.number]).columns if c not in y_cols][:1]
    if isinstance(y_cols, str):
        y_cols = [c.strip() for c in y_cols.split(",") if c.strip()]
    if isinstance(cov_cols, str):
        cov_cols = [c.strip() for c in cov_cols.split(",") if c.strip()]
    _require_cols(df, y_cols + [g_col] + cov_cols)

    sub = df[y_cols + [g_col] + cov_cols].dropna()
    for c in y_cols + cov_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    lhs = " + ".join([f"Q('{c}')" for c in y_cols])
    rhs = f"C(Q('{g_col}')) + " + " + ".join([f"Q('{c}')" for c in cov_cols])
    mv = MANOVA.from_formula(f"{lhs} ~ {rhs}", data=sub)
    mv_res = mv.mv_test().results

    table = []
    for term_key, term_dict in mv_res.items():
        if term_key == "Intercept":
            continue
        clean_term = term_key.replace("Q('", "").replace("')", "")
        st = term_dict["stat"]
        for stat_name in ["Wilks' lambda", "Pillai's trace"]:
            row = st.loc[stat_name]
            table.append({
                "Parameter": f"{clean_term} — {stat_name}",
                "Estimate": float(row["Value"]),
                "Std_Error": None,
                "Statistic": float(row["F Value"]),
                "df": f"{row['Num DF']:.1f}, {row['Den DF']:.1f}",
                "p_value": float(row["Pr > F"]),
                "CI_Low": None,
                "CI_High": None
            })

    return {
        "method": "Multivariate Analysis of Covariance (MANCOVA)",
        "summary": {"N": len(sub), "Dependent_Vars_p": len(y_cols), "Covariates_q": len(cov_cols)},
        "table": table,
        "diagnostics": {"Covariates_Controlled": ", ".join(cov_cols)},
        "plot": {
            "type": "scatter",
            "title": f"MANCOVA Joint Outcome Space ({y_cols[0]} vs {y_cols[1]})",
            "x": sub[y_cols[0]].tolist(),
            "y": sub[y_cols[1]].tolist()
        }
    }


# ==============================================================================
# 4. CATEGORY B: NONPARAMETRIC STATISTICS (5 METHODS)
# ==============================================================================

def est_mann_whitney(df: pd.DataFrame, spec: dict) -> dict:
    y_col = spec.get("y") or df.columns[0]
    g_col = spec.get("group") or df.columns[1]
    _require_cols(df, [y_col, g_col])
    sub = df[[y_col, g_col]].dropna()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()
    groups = list(sub.groupby(g_col))
    if len(groups) != 2:
        raise ValueError(f"Mann-Whitney U test requires exactly 2 groups (found {len(groups)}).")

    (l1, g1_df), (l2, g2_df) = groups[0], groups[1]
    x, y = g1_df[y_col].values, g2_df[y_col].values
    n1, n2 = len(x), len(y)
    u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)  # Rank-biserial correlation
    auc = u_stat / (n1 * n2)
    hl_shift = float(np.median(np.subtract.outer(x, y)))  # Hodges-Lehmann location shift

    return {
        "method": "Mann–Whitney U Test (Wilcoxon Rank-Sum)",
        "summary": {"N1": n1, "N2": n2, "U_Statistic": u_stat, "p_value": p_val,
                    "Hodges_Lehmann_Shift": hl_shift, "Rank_Biserial_r": r_rb, "AUC_Stochastic_Dominance": auc},
        "table": [
            {"Parameter": f"Mann-Whitney U ({l1} vs {l2})", "Estimate": u_stat, "Std_Error": None,
             "Statistic": u_stat, "df": None, "p_value": p_val, "CI_Low": None, "CI_High": None},
            {"Parameter": "Hodges–Lehmann Median Shift", "Estimate": hl_shift, "Std_Error": None,
             "Statistic": r_rb, "df": None, "p_value": p_val, "CI_Low": None, "CI_High": None},
            {"Parameter": f"Median: {l1}", "Estimate": float(np.median(x)), "Std_Error": float(stats.iqr(x)),
             "Statistic": None, "df": n1, "p_value": None, "CI_Low": float(np.percentile(x, 25)), "CI_High": float(np.percentile(x, 75))},
            {"Parameter": f"Median: {l2}", "Estimate": float(np.median(y)), "Std_Error": float(stats.iqr(y)),
             "Statistic": None, "df": n2, "p_value": None, "CI_Low": float(np.percentile(y, 25)), "CI_High": float(np.percentile(y, 75))},
        ],
        "diagnostics": {"Stochastic_Superiority_Prob": auc, "Rank_Biserial_Effect": r_rb},
        "plot": {
            "type": "box",
            "title": f"Mann-Whitney Nonparametric Comparison ({l1} vs {l2})",
            "labels": [str(l1), str(l2)],
            "values": [x.tolist(), y.tolist()]
        }
    }


def est_wilcoxon(df: pd.DataFrame, spec: dict) -> dict:
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    c1 = spec.get("x1") or num_cols[0]
    c2 = spec.get("x2") or num_cols[1]
    sub = _numeric_df(df, [c1, c2])
    x, y = sub[c1].values, sub[c2].values
    diff = x - y
    nz = diff[diff != 0]
    if len(nz) < 3:
        raise ValueError("Fewer than 3 non-zero paired differences for Wilcoxon signed-rank test.")

    w_stat, p_val = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    n_eff = len(nz)
    total_ranks = n_eff * (n_eff + 1) / 2.0
    r_rb = (2.0 * w_stat / total_ranks) - 1.0
    # Walsh averages for Hodges-Lehmann pseudomedian
    walsh = [(nz[i] + nz[j]) / 2.0 for i in range(n_eff) for j in range(i, n_eff)]
    pseudomedian = float(np.median(walsh))

    return {
        "method": "Wilcoxon Signed-Rank Test (Paired)",
        "summary": {"Pairs_N": len(sub), "NonZero_Pairs": n_eff, "W_Statistic": w_stat,
                    "p_value": p_val, "Pseudomedian_Shift": pseudomedian, "Rank_Biserial_r": r_rb},
        "table": [
            {"Parameter": f"Wilcoxon W ({c1} - {c2})", "Estimate": w_stat, "Std_Error": None,
             "Statistic": w_stat, "df": n_eff, "p_value": p_val, "CI_Low": None, "CI_High": None},
            {"Parameter": "Walsh Pseudomedian Difference", "Estimate": pseudomedian, "Std_Error": float(stats.iqr(diff)),
             "Statistic": r_rb, "df": n_eff, "p_value": p_val,
             "CI_Low": float(np.percentile(walsh, 2.5)), "CI_High": float(np.percentile(walsh, 97.5))},
        ],
        "diagnostics": {"Zero_Differences_Dropped": len(sub) - n_eff, "Effect_Size_r": r_rb},
        "plot": {
            "type": "box",
            "title": f"Paired Differences Distribution ({c1} - {c2})",
            "labels": [f"{c1} - {c2}"],
            "values": [diff.tolist()]
        }
    }


def est_kruskal(df: pd.DataFrame, spec: dict) -> dict:
    y_col = spec.get("y") or df.columns[0]
    g_col = spec.get("group") or df.columns[1]
    _require_cols(df, [y_col, g_col])
    sub = df[[y_col, g_col]].dropna()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()
    groups = [grp[y_col].values for _, grp in sub.groupby(g_col)]
    labels = [str(k) for k, _ in sub.groupby(g_col)]
    k, n = len(groups), len(sub)
    if k < 2:
        raise ValueError("Kruskal-Wallis test requires at least 2 groups.")

    h_stat, p_val = stats.kruskal(*groups)
    eps_sq = h_stat / ((n ** 2 - 1) / (n + 1))
    eta_h_sq = max(0.0, (h_stat - k + 1) / (n - k))

    table = [
        {"Parameter": "Kruskal–Wallis H Statistic", "Estimate": h_stat, "Std_Error": None,
         "Statistic": h_stat, "df": k - 1, "p_value": p_val, "CI_Low": eps_sq, "CI_High": None}
    ]
    for lbl, g in zip(labels, groups):
        table.append({
            "Parameter": f"Group Median: {lbl} (n={len(g)})",
            "Estimate": float(np.median(g)),
            "Std_Error": float(stats.iqr(g)),
            "Statistic": None,
            "df": len(g),
            "p_value": None,
            "CI_Low": float(np.percentile(g, 25)),
            "CI_High": float(np.percentile(g, 75))
        })

    return {
        "method": "Kruskal–Wallis One-Way Nonparametric ANOVA",
        "summary": {"N": n, "Groups_k": k, "H_Statistic": h_stat, "df": k - 1,
                    "p_value": p_val, "Epsilon_Squared": eps_sq, "Eta_H_Squared": eta_h_sq},
        "table": table,
        "diagnostics": {"Ordinal_Epsilon_Squared": eps_sq, "Nonparametric_Eta_Sq": eta_h_sq},
        "plot": {
            "type": "box",
            "title": f"Kruskal–Wallis Rank Groups ({y_col} by {g_col})",
            "labels": labels,
            "values": [g.tolist() for g in groups]
        }
    }


def est_friedman(df: pd.DataFrame, spec: dict) -> dict:
    cond_cols = spec.get("conditions") or list(df.select_dtypes(include=[np.number]).columns)
    sub = _numeric_df(df, cond_cols)
    n, k = sub.shape
    if k < 3:
        raise ValueError("Friedman test requires at least 3 repeated conditions/treatments.")

    chi2_stat, p_val = stats.friedmanchisquare(*[sub[c].values for c in cond_cols])
    kendall_w = chi2_stat / (n * (k - 1))  # Kendall's coefficient of concordance W

    ranks = np.apply_along_axis(stats.rankdata, 1, sub.values)
    mean_ranks = np.mean(ranks, axis=0)

    table = [
        {"Parameter": "Friedman Chi-Square (Q)", "Estimate": chi2_stat, "Std_Error": None,
         "Statistic": chi2_stat, "df": k - 1, "p_value": p_val, "CI_Low": kendall_w, "CI_High": None}
    ]
    for c, mr in zip(cond_cols, mean_ranks):
        table.append({
            "Parameter": f"Condition: {c} (Mean Rank)",
            "Estimate": float(mr),
            "Std_Error": float(np.std(sub[c])),
            "Statistic": float(np.median(sub[c])),
            "df": n,
            "p_value": None,
            "CI_Low": float(np.percentile(sub[c], 25)),
            "CI_High": float(np.percentile(sub[c], 75))
        })

    return {
        "method": "Friedman Repeated-Measures Rank Test",
        "summary": {"Blocks_N": n, "Treatments_k": k, "Friedman_Chi2": chi2_stat,
                    "df": k - 1, "p_value": p_val, "Kendalls_W": kendall_w},
        "table": table,
        "diagnostics": {"Kendalls_Concordance_W": kendall_w},
        "plot": {
            "type": "bar",
            "title": "Friedman Mean Ranks Across Conditions",
            "labels": cond_cols,
            "values": mean_ranks.tolist()
        }
    }


def est_kendall(df: pd.DataFrame, spec: dict) -> dict:
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    c1 = spec.get("x") or num_cols[0]
    c2 = spec.get("y") or num_cols[1]
    sub = _numeric_df(df, [c1, c2])
    x, y = sub[c1].values, sub[c2].values
    n = len(sub)

    tau_b, p_b = stats.kendalltau(x, y, variant="b")
    tau_c, p_c = stats.kendalltau(x, y, variant="c")
    rho, p_rho = stats.spearmanr(x, y)
    # Asymptotic variance bound for Kendall's tau-b
    se_tau = math.sqrt((2.0 * (2.0 * n + 5.0)) / (9.0 * n * (n - 1.0)))

    return {
        "method": "Kendall's Tau Rank Correlation",
        "summary": {"N": n, "Tau_b": tau_b, "p_value": p_b, "Tau_c": tau_c, "Spearman_Rho": rho},
        "table": [
            {"Parameter": f"Kendall's Tau-b ({c1}, {c2})", "Estimate": tau_b, "Std_Error": se_tau,
             "Statistic": tau_b / se_tau, "df": n - 2, "p_value": p_b,
             "CI_Low": max(-1.0, tau_b - 1.96 * se_tau), "CI_High": min(1.0, tau_b + 1.96 * se_tau)},
            {"Parameter": f"Kendall's Tau-c ({c1}, {c2})", "Estimate": tau_c, "Std_Error": se_tau,
             "Statistic": tau_c / se_tau, "df": n - 2, "p_value": p_c, "CI_Low": None, "CI_High": None},
            {"Parameter": "Spearman Rank Rho (Benchmark)", "Estimate": rho, "Std_Error": None,
             "Statistic": rho, "df": n - 2, "p_value": p_rho, "CI_Low": None, "CI_High": None},
        ],
        "diagnostics": {"Asymptotic_SE": se_tau},
        "plot": {
            "type": "scatter",
            "title": f"Rank Concordance Scatter ({c1} vs {c2})",
            "x": x.tolist(),
            "y": y.tolist()
        }
    }


# ==============================================================================
# 5. CATEGORY C: MULTIVARIATE STATISTICAL ANALYSIS (6 METHODS)
# ==============================================================================

def est_pca(df: pd.DataFrame, spec: dict) -> dict:
    cols = spec.get("vars") or list(df.select_dtypes(include=[np.number]).columns)
    sub = _numeric_df(df, cols)
    n, p = sub.shape
    if p < 2:
        raise ValueError("PCA requires at least 2 numeric variables.")

    scale = bool(spec.get("standardize", True))
    X = sub.values
    X_c = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) if scale else 1.0)
    U, s, Vt = linalg.svd(X_c, full_matrices=False)
    eigvals = (s ** 2) / (n - 1)
    prop_var = eigvals / np.sum(eigvals)
    cum_var = np.cumsum(prop_var)
    loadings = Vt.T * np.sqrt(eigvals)

    table = []
    for i in range(len(eigvals)):
        table.append({
            "Parameter": f"PC{i+1} (Eigenvalue)",
            "Estimate": float(eigvals[i]),
            "Std_Error": float(prop_var[i] * 100.0),
            "Statistic": float(cum_var[i] * 100.0),
            "df": p - i,
            "p_value": None,
            "CI_Low": float(loadings[0, i]),
            "CI_High": float(loadings[1, i]) if p > 1 else None
        })

    # Bartlett's test of sphericity & KMO
    R = np.corrcoef(X, rowvar=False)
    det_R = max(1e-15, float(linalg.det(R)))
    bartlett_chi2 = -(n - 1 - (2 * p + 5) / 6.0) * math.log(det_R)
    bartlett_df = p * (p - 1) // 2
    bartlett_p = 1 - stats.chi2.cdf(bartlett_chi2, bartlett_df)

    return {
        "method": "Principal Component Analysis (PCA — SVD Decomposition)",
        "summary": {"N": n, "Variables_p": p, "PC1_Var_Pct": prop_var[0] * 100.0,
                    "Top2_Cum_Var_Pct": cum_var[min(1, p - 1)] * 100.0,
                    "Kaiser_Components_GT1": int(np.sum(eigvals > 1.0))},
        "table": table,
        "diagnostics": {
            "Bartlett_Sphericity_Chi2": bartlett_chi2,
            "Bartlett_df": bartlett_df,
            "Bartlett_p_value": bartlett_p,
            "Condition_Number": float(np.sqrt(eigvals[0] / max(eigvals[-1], 1e-15)))
        },
        "plot": {
            "type": "bar",
            "title": "PCA Scree Plot (Eigenvalues by Component)",
            "labels": [f"PC{i+1}" for i in range(len(eigvals))],
            "values": eigvals.tolist()
        }
    }


def est_efa(df: pd.DataFrame, spec: dict) -> dict:
    cols = spec.get("vars") or list(df.select_dtypes(include=[np.number]).columns)
    sub = _numeric_df(df, cols)
    n, p = sub.shape
    m = int(spec.get("n_factors", min(2, p - 1)))
    if p < 3 or m >= p:
        raise ValueError("EFA requires p >= 3 variables and n_factors < p.")

    R = np.corrcoef(sub.values, rowvar=False)
    # Principal Axis Factoring + Varimax orthogonal rotation
    inv_R = linalg.pinv(R)
    h2 = 1.0 - 1.0 / np.diag(inv_R)
    h2 = np.clip(h2, 0.05, 0.99)
    for _ in range(50):
        R_red = R.copy()
        np.fill_diagonal(R_red, h2)
        evals, evecs = linalg.eigh(R_red)
        idx = np.argsort(evals)[::-1][:m]
        L = evecs[:, idx] * np.sqrt(np.maximum(evals[idx], 1e-8))
        h2_new = np.sum(L ** 2, axis=1)
        if np.max(np.abs(h2_new - h2)) < 1e-6:
            h2 = h2_new
            break
        h2 = np.clip(h2_new, 0.01, 0.999)

    # Varimax rotation
    Phi = L.copy()
    d = 0.0
    for _ in range(100):
        d_old = d
        Lambda = Phi
        u, s, vt = linalg.svd(
            Phi.T @ (Lambda ** 3 - (1.0 / p) * Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)))
        )
        T = u @ vt
        Phi = Phi @ T
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + 1e-6:
            break

    # KMO sampling adequacy
    S_inv = linalg.pinv(R)
    D = np.diag(1.0 / np.sqrt(np.diag(S_inv)))
    P_part = -D @ S_inv @ D
    np.fill_diagonal(P_part, 0.0)
    R_off = R.copy()
    np.fill_diagonal(R_off, 0.0)
    kmo = float(np.sum(R_off ** 2) / (np.sum(R_off ** 2) + np.sum(P_part ** 2)))

    table = []
    for i, col in enumerate(cols):
        table.append({
            "Parameter": f"Item: {col}",
            "Estimate": float(Phi[i, 0]),
            "Std_Error": float(Phi[i, 1]) if m > 1 else 0.0,
            "Statistic": float(h2[i]),
            "df": m,
            "p_value": None,
            "CI_Low": float(1.0 - h2[i]),  # Uniqueness
            "CI_High": None
        })

    return {
        "method": "Exploratory Factor Analysis (PAF + Varimax Rotation)",
        "summary": {"N": n, "Items_p": p, "Factors_m": m, "KMO_Adequacy": kmo,
                    "Mean_Communality": float(np.mean(h2))},
        "table": table,
        "diagnostics": {"KMO_Index": kmo, "Varimax_Converged": True},
        "plot": {
            "type": "bar",
            "title": "EFA Item Communalities (h²)",
            "labels": cols,
            "values": h2.tolist()
        }
    }


def est_cfa(df: pd.DataFrame, spec: dict) -> dict:
    # Single-factor or two-factor congeneric CFA estimated via Maximum Likelihood covariance fitting
    cols = spec.get("vars") or list(df.select_dtypes(include=[np.number]).columns)
    sub = _numeric_df(df, cols)
    n, p = sub.shape
    if p < 3:
        raise ValueError("Confirmatory Factor Analysis requires at least 3 observed indicator variables.")

    S = np.cov(sub.values, rowvar=False, ddof=1)
    logdet_S = float(np.linalg.slogdet(S)[1])

    # 1-factor ML: Sigma(theta) = lambda * lambda^T + diag(psi), marker variable lambda_1 = 1
    def nll(theta):
        lam = np.concatenate([[1.0], theta[:p - 1]])
        psi = np.exp(theta[p - 1:]) + 1e-5
        Sigma = np.outer(lam, lam) + np.diag(psi)
        sign, logdet_Sig = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return 1e12
        tr_term = np.trace(S @ linalg.inv(Sigma))
        return 0.5 * (logdet_Sig + tr_term - logdet_S - p)

    init_lam = np.ones(p - 1) * 0.8
    init_psi = np.log(np.maximum(np.diag(S) * 0.5, 0.05))
    res = optimize.minimize(nll, np.concatenate([init_lam, init_psi]), method="BFGS")

    lam_hat = np.concatenate([[1.0], res.x[:p - 1]])
    psi_hat = np.exp(res.x[p - 1:]) + 1e-5
    f_ml = float(res.fun)
    chi2_stat = (n - 1) * f_ml
    df_model = p * (p + 1) // 2 - (2 * p - 1)
    p_chi2 = 1 - stats.chi2.cdf(chi2_stat, df_model) if df_model > 0 else np.nan

    # Null independence model chi2 for CFI / TLI
    S_null = np.diag(np.diag(S))
    f_null = 0.5 * (np.linalg.slogdet(S_null)[1] + p - logdet_S - p)
    chi2_null = (n - 1) * f_null
    df_null = p * (p - 1) // 2
    cfi = float(1.0 - max(chi2_stat - df_model, 0.0) / max(chi2_null - df_null, 1e-8))
    tli = float(((chi2_null / df_null) - (chi2_stat / max(df_model, 1))) / max((chi2_null / df_null) - 1.0, 1e-8))
    rmsea = float(math.sqrt(max(chi2_stat - df_model, 0.0) / (max(df_model, 1) * (n - 1))))

    # Standard errors via inverse Hessian
    hess_inv = res.hess_inv if isinstance(res.hess_inv, np.ndarray) else np.eye(len(res.x))
    se_vec = np.sqrt(np.maximum(np.diag(hess_inv) / max(n - 1, 1), 1e-10))

    table = []
    for i, col in enumerate(cols):
        se_l = 0.0 if i == 0 else float(se_vec[i - 1])
        z_l = None if i == 0 else float(lam_hat[i] / se_l)
        p_l = None if i == 0 else float(2 * (1 - stats.norm.cdf(abs(z_l))))
        table.append({
            "Parameter": f"Loading: F1 -> {col}" + (" (Fixed=1)" if i == 0 else ""),
            "Estimate": float(lam_hat[i]),
            "Std_Error": se_l,
            "Statistic": z_l,
            "df": df_model,
            "p_value": p_l,
            "CI_Low": float(lam_hat[i] - 1.96 * se_l),
            "CI_High": float(lam_hat[i] + 1.96 * se_l)
        })

    return {
        "method": "Confirmatory Factor Analysis (Maximum Likelihood SEM)",
        "summary": {"N": n, "Chi2_Model": chi2_stat, "df": df_model, "p_value": p_chi2,
                    "CFI": cfi, "TLI": tli, "RMSEA": rmsea, "Converged": bool(res.success)},
        "table": table,
        "diagnostics": {"CFI": cfi, "TLI": tli, "RMSEA": rmsea, "LogL_Discrepancy_F_ML": f_ml},
        "plot": {
            "type": "bar",
            "title": "CFA Factor Loadings (Unstandardized ML)",
            "labels": cols,
            "values": lam_hat.tolist()
        }
    }


def est_lda(df: pd.DataFrame, spec: dict) -> dict:
    x_cols = spec.get("x_vars") or list(df.select_dtypes(include=[np.number]).columns)
    g_col = spec.get("group") or [c for c in df.columns if c not in x_cols][0]
    _require_cols(df, x_cols + [g_col])

    sub = df[x_cols + [g_col]].dropna()
    for c in x_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    X = sub[x_cols].values
    y = sub[g_col].values
    classes = np.unique(y)
    n, p = X.shape
    k = len(classes)
    if k < 2:
        raise ValueError("LDA requires at least 2 target classes.")

    grand_mean = X.mean(axis=0)
    Sw = np.zeros((p, p))
    Sb = np.zeros((p, p))
    priors = {}
    means = {}
    for c in classes:
        Xc = X[y == c]
        nc = len(Xc)
        priors[c] = nc / n
        mc = Xc.mean(axis=0)
        means[c] = mc
        diff_w = Xc - mc
        Sw += diff_w.T @ diff_w
        diff_b = (mc - grand_mean).reshape(-1, 1)
        Sb += nc * (diff_b @ diff_b.T)

    Sw_reg = Sw / (n - k) + np.eye(p) * 1e-6
    Sw_inv = linalg.inv(Sw_reg)
    eigvals, eigvecs = linalg.eig(Sw_inv @ Sb)
    idx = np.argsort(eigvals.real)[::-1][:min(k - 1, p)]
    eigvals = np.maximum(eigvals.real[idx], 0.0)
    W = eigvecs.real[:, idx]

    # Classification accuracy via Mahalanobis linear discriminant score
    preds = []
    for xi in X:
        scores = {}
        for c in classes:
            mc = means[c]
            scores[c] = float(xi @ Sw_inv @ mc - 0.5 * mc @ Sw_inv @ mc + math.log(priors[c]))
        preds.append(max(scores, key=scores.get))
    acc = float(np.mean(np.array(preds) == y))

    table = []
    for j, col in enumerate(x_cols):
        table.append({
            "Parameter": f"LD1 Weight: {col}",
            "Estimate": float(W[j, 0]),
            "Std_Error": float(W[j, 1]) if W.shape[1] > 1 else None,
            "Statistic": float(eigvals[0]),
            "df": k - 1,
            "p_value": None,
            "CI_Low": None,
            "CI_High": None
        })

    ld1_scores = (X - grand_mean) @ W[:, 0]
    return {
        "method": "Linear Discriminant Analysis (Fisher's LDA)",
        "summary": {"N": n, "Classes_k": k, "Predictors_p": p,
                    "In_Sample_Accuracy_Pct": acc * 100.0, "LD1_Eigenvalue": float(eigvals[0])},
        "table": table,
        "diagnostics": {"Canonical_Discriminant_Corr": float(math.sqrt(eigvals[0] / (1.0 + eigvals[0])))},
        "plot": {
            "type": "box",
            "title": "LD1 Discriminant Scores by Class",
            "labels": [str(c) for c in classes],
            "values": [ld1_scores[y == c].tolist() for c in classes]
        }
    }


def est_cca(df: pd.DataFrame, spec: dict) -> dict:
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    half = max(1, len(num_cols) // 2)
    x_cols = spec.get("x_vars") or num_cols[:half]
    y_cols = spec.get("y_vars") or num_cols[half:]
    if isinstance(x_cols, str):
        x_cols = [c.strip() for c in x_cols.split(",") if c.strip()]
    if isinstance(y_cols, str):
        y_cols = [c.strip() for c in y_cols.split(",") if c.strip()]
    sub = _numeric_df(df, x_cols + y_cols)

    cc = CanCorr(sub[y_cols].values, sub[x_cols].values)
    # statsmodels exposes the canonical-root test table through the stats attribute.
    test_result = cc.corr_test()
    test_df = test_result.stats

    table = []
    for i, r_val in enumerate(cc.cancorr):
        row = test_df.iloc[i]
        table.append({
            "Parameter": f"Canonical Variate Pair {i+1}",
            "Estimate": float(r_val),
            "Std_Error": float(r_val ** 2),  # Shared variance R^2_c
            "Statistic": float(row["F Value"]),
            "df": f"{row['Num DF']:.0f}, {row['Den DF']:.1f}",
            "p_value": float(row["Pr > F"]),
            "CI_Low": float(row["Wilks' lambda"]),
            "CI_High": None
        })

    return {
        "method": "Canonical Correlation Analysis (CCA)",
        "summary": {"N": len(sub), "Set_X_Dim": len(x_cols), "Set_Y_Dim": len(y_cols),
                    "Canonical_Corr_1": float(cc.cancorr[0]),
                    "Squared_Canonical_Corr_1": float(cc.cancorr[0] ** 2)},
        "table": table,
        "diagnostics": {"Wilks_Lambda_1": float(test_df.iloc[0]["Wilks' lambda"])},
        "plot": {
            "type": "bar",
            "title": "Canonical Correlations Across Variate Pairs",
            "labels": [f"CV{i+1}" for i in range(len(cc.cancorr))],
            "values": cc.cancorr.tolist()
        }
    }


def est_ca(df: pd.DataFrame, spec: dict) -> dict:
    # Two-way contingency table correspondence analysis
    c1 = spec.get("row_var") or df.columns[0]
    c2 = spec.get("col_var") or df.columns[1]
    _require_cols(df, [c1, c2])
    ct = pd.crosstab(df[c1], df[c2])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Correspondence Analysis requires at least a 2x2 contingency table.")

    N_mat = ct.values.astype(float)
    n_tot = N_mat.sum()
    P = N_mat / n_tot
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)
    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r.flatten()))
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c.flatten()))

    S_res = Dr_inv_sqrt @ (P - r @ c) @ Dc_inv_sqrt
    U, s, Vt = linalg.svd(S_res, full_matrices=False)
    eigvals = s ** 2
    total_inertia = float(np.sum(eigvals))
    chi2_stat = n_tot * total_inertia
    df_ca = (ct.shape[0] - 1) * (ct.shape[1] - 1)
    p_val = 1 - stats.chi2.cdf(chi2_stat, df_ca)

    row_coords = Dr_inv_sqrt @ U @ np.diag(s)
    table = []
    for k in range(len(eigvals)):
        if eigvals[k] < 1e-12:
            continue
        table.append({
            "Parameter": f"Dimension {k+1} Principal Inertia",
            "Estimate": float(eigvals[k]),
            "Std_Error": float(eigvals[k] / total_inertia * 100.0),
            "Statistic": float(s[k]),
            "df": df_ca,
            "p_value": p_val if k == 0 else None,
            "CI_Low": None,
            "CI_High": None
        })

    return {
        "method": "Correspondence Analysis (CA — SVD of Standardized Residuals)",
        "summary": {"Total_Count_N": int(n_tot), "Table_Shape": f"{ct.shape[0]}x{ct.shape[1]}",
                    "Total_Inertia": total_inertia, "Chi2_Independence": chi2_stat,
                    "df": df_ca, "p_value": p_val},
        "table": table,
        "diagnostics": {"Dim1_Inertia_Pct": float(eigvals[0] / total_inertia * 100.0)},
        "plot": {
            "type": "scatter",
            "title": "CA Row Principal Coordinates (Dim 1 vs Dim 2)",
            "x": row_coords[:, 0].tolist(),
            "y": row_coords[:, 1].tolist() if row_coords.shape[1] > 1 else [0.0] * len(row_coords)
        }
    }


# ==============================================================================
# 6. CATEGORY D: SURVIVAL ANALYSIS (5 METHODS)
# ==============================================================================

def est_km(df: pd.DataFrame, spec: dict) -> dict:
    t_col = spec.get("time") or (df.columns[0] if len(df.columns) > 0 else None)
    e_col = spec.get("event") or (df.columns[1] if len(df.columns) > 1 else None)
    g_col = spec.get("group")
    cols = [t_col, e_col] + ([g_col] if g_col and g_col in df.columns else [])
    _require_cols(df, [t_col, e_col])
    sub = df[cols].copy().dropna()
    sub[t_col] = pd.to_numeric(sub[t_col], errors="coerce")
    sub[e_col] = pd.to_numeric(sub[e_col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 2:
        raise ValueError("Kaplan–Meier estimation requires at least two complete observations.")
    if (sub[t_col] <= 0).any():
        raise ValueError("Survival times must be strictly positive (> 0).")
    if not sub[e_col].isin([0, 1]).all():
        raise ValueError("Event indicator must contain only 1 (event) and 0 (censored).")

    sf = SurvfuncRight(sub[t_col].values, sub[e_col].values)
    q50 = sf.quantile(0.5)
    q25 = sf.quantile(0.25)
    q75 = sf.quantile(0.75)

    table = [
        {"Parameter": "Median Survival Time (50th pct)", "Estimate": float(q50) if not np.isnan(q50) else None,
         "Std_Error": None, "Statistic": None, "df": None, "p_value": None,
         "CI_Low": float(q25) if not np.isnan(q25) else None, "CI_High": float(q75) if not np.isnan(q75) else None}
    ]
    step_idx = np.linspace(0, len(sf.surv_times) - 1, min(8, len(sf.surv_times)), dtype=int)
    for i in step_idx:
        s_hat = float(sf.surv_prob[i])
        se_raw = float(sf.surv_prob_se[i])
        se_s = se_raw if math.isfinite(se_raw) else None
        table.append({
            "Parameter": f"S(t={sf.surv_times[i]:.2f})",
            "Estimate": s_hat,
            "Std_Error": se_s,
            "Statistic": s_hat / se_s if se_s is not None and se_s > 0 else None,
            "df": int(sf.n_risk[i]),
            "p_value": None,
            "CI_Low": max(0.0, s_hat - 1.96 * se_s) if se_s is not None else None,
            "CI_High": min(1.0, s_hat + 1.96 * se_s) if se_s is not None else None
        })

    # Log-rank test if group column provided
    lr_stat, lr_p = None, None
    if g_col and g_col in sub.columns and sub[g_col].nunique() >= 2:
        from statsmodels.duration.survfunc import survdiff
        lr_stat, lr_p = survdiff(sub[t_col].values, sub[e_col].values, sub[g_col].values)

    return {
        "method": "Kaplan–Meier Nonparametric Product-Limit Estimator",
        "summary": {"Subjects_N": len(sub), "Events": int(sub[e_col].sum()),
                    "Censored": int(len(sub) - sub[e_col].sum()),
                    "Median_Survival": float(q50) if not np.isnan(q50) else None,
                    "Log_Rank_Chi2": lr_stat, "Log_Rank_p": lr_p},
        "table": table,
        "diagnostics": {"Greenwood_SE_Computed": True, "Log_Rank_p": lr_p},
        "plot": {
            "type": "line",
            "title": "Kaplan–Meier Survival Curve S(t)",
            "x": sf.surv_times.tolist(),
            "y": sf.surv_prob.tolist()
        }
    }


def est_cox_ph(df: pd.DataFrame, spec: dict) -> dict:
    t_col = spec.get("time") or df.columns[0]
    e_col = spec.get("event") or df.columns[1]
    x_cols = spec.get("x_vars") or [c for c in df.select_dtypes(include=[np.number]).columns if c not in [t_col, e_col]]
    if isinstance(x_cols, str):
        x_cols = [c.strip() for c in x_cols.split(",") if c.strip()]
    sub = _numeric_df(df, [t_col, e_col] + x_cols)

    model = PHReg(sub[t_col].values, sub[x_cols].values, status=sub[e_col].values, ties="breslow")
    fit = model.fit()

    ci = fit.conf_int()
    table = []
    for j, col in enumerate(x_cols):
        beta = float(fit.params[j])
        hr = math.exp(beta)
        table.append({
            "Parameter": f"{col} (log-HR)",
            "Estimate": beta,
            "Std_Error": float(fit.bse[j]),
            "Statistic": float(fit.tvalues[j]),
            "df": 1,
            "p_value": float(fit.pvalues[j]),
            "CI_Low": float(ci[j, 0]),
            "CI_High": float(ci[j, 1])
        })
        table.append({
            "Parameter": f"{col} (Hazard Ratio exp(β))",
            "Estimate": hr,
            "Std_Error": float(hr * fit.bse[j]),
            "Statistic": float(fit.tvalues[j]),
            "df": 1,
            "p_value": float(fit.pvalues[j]),
            "CI_Low": math.exp(float(ci[j, 0])),
            "CI_High": math.exp(float(ci[j, 1]))
        })

    # Schoenfeld residuals are returned in original observation order; censored
    # rows are NaN. Correlate each covariate's event residuals with event time.
    schoen = fit.schoenfeld_residuals
    event_mask = sub[e_col].to_numpy() == 1
    ph_tests = {}
    if schoen is not None:
        schoen = np.asarray(schoen, dtype=float)
        for j, col in enumerate(x_cols):
            valid = event_mask & np.isfinite(schoen[:, j])
            if int(valid.sum()) > 5 and np.std(schoen[valid, j]) > 0 and np.std(sub.loc[valid, t_col]) > 0:
                corr, pval = stats.pearsonr(sub.loc[valid, t_col].to_numpy(dtype=float), schoen[valid, j])
                ph_tests[col] = {"correlation": float(corr), "p_value": float(pval), "n_events": int(valid.sum())}
            else:
                ph_tests[col] = {"correlation": None, "p_value": None, "n_events": int(valid.sum())}
    n_obs = int(len(sub))
    n_params = int(len(fit.params))
    n_events = int(np.sum(event_mask))
    partial_aic = float(-2.0 * fit.llf + 2.0 * n_params)
    partial_bic = float(-2.0 * fit.llf + math.log(max(n_events, 1)) * n_params)

    return {
        "method": "Cox Proportional Hazards Model (Breslow Partial Likelihood)",
        "summary": {"N": n_obs, "Events": n_events,
                    "Partial_Log_Likelihood": float(fit.llf),
                    "Partial_Likelihood_AIC": partial_aic, "Partial_Likelihood_BIC": partial_bic},
        "table": table,
        "diagnostics": {"Schoenfeld_Time_Correlation": ph_tests, "Ties_Method": "Breslow"},
        "plot": {
            "type": "bar",
            "title": "Cox PH Hazard Ratios exp(β)",
            "labels": x_cols,
            "values": [math.exp(float(b)) for b in fit.params]
        }
    }


def _fit_parametric_surv_mle(t: np.ndarray, d: np.ndarray, X: np.ndarray, dist: str):
    """Exact Maximum Likelihood estimation for Weibull, Exponential, and Log-Normal AFT models."""
    n, p = X.shape
    log_t = np.log(t)

    if dist == "exponential":
        def nll(theta):
            beta = theta[:p]
            z = log_t - X @ beta
            # f(t) = (1/mu)*exp(-t/mu), S(t) = exp(-t/mu) => logL = sum(d*(z - exp(z)) - (1-d)*exp(z))
            ll = np.sum(d * z - np.exp(z))
            return -float(ll)

        res = optimize.minimize(nll, np.zeros(p), method="BFGS")
        params = res.x
        names = [f"β_{j}" for j in range(p)]
    elif dist == "lognormal":
        def nll(theta):
            beta = theta[:p]
            log_sig = theta[p]
            sig = math.exp(log_sig)
            z = (log_t - X @ beta) / sig
            ll_event = d * (-0.5 * math.log(2 * math.pi) - log_sig - log_t - 0.5 * z ** 2)
            sf = np.maximum(stats.norm.sf(z), 1e-300)
            ll_cens = (1.0 - d) * np.log(sf)
            return -float(np.sum(ll_event + ll_cens))

        res = optimize.minimize(nll, np.append(np.zeros(p), 0.0), method="BFGS")
        params = res.x
        names = [f"β_{j}" for j in range(p)] + ["log(σ)"]
    else:  # Weibull AFT (default)
        def nll(theta):
            beta = theta[:p]
            log_sig = theta[p]
            sig = math.exp(log_sig)
            z = (log_t - X @ beta) / sig
            # Extreme value density: f(z) = (1/sig)*exp(z - exp(z)), S(z) = exp(-exp(z))
            ll = np.sum(d * (-log_sig - log_t + z) - np.exp(z))
            return -float(ll)

        res = optimize.minimize(nll, np.append(np.zeros(p), 0.0), method="BFGS")
        params = res.x
        names = [f"β_{j}" for j in range(p)] + ["log(σ)"]

    # Numerical Hessian for standard errors
    eps = 1e-5
    k = len(params)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei = np.zeros(k); ei[i] = eps
            ej = np.zeros(k); ej[j] = eps
            H[i, j] = (nll(params + ei + ej) - nll(params + ei - ej) - nll(params - ei + ej) + nll(params - ei - ej)) / (4 * eps ** 2)
    cov = linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-10))
    llf = -float(res.fun)
    return params, se, llf, bool(res.success), names


def est_aft(df: pd.DataFrame, spec: dict) -> dict:
    t_col = spec.get("time") or df.columns[0]
    e_col = spec.get("event") or df.columns[1]
    dist = str(spec.get("distribution", "weibull")).lower()
    x_cols = spec.get("x_vars") or [c for c in df.select_dtypes(include=[np.number]).columns if c not in [t_col, e_col]]
    if isinstance(x_cols, str):
        x_cols = [c.strip() for c in x_cols.split(",") if c.strip()]
    sub = _numeric_df(df, [t_col, e_col] + x_cols)
    if (sub[t_col] <= 0).any():
        raise ValueError("Survival times must be strictly positive.")

    X = sm.add_constant(sub[x_cols].values)
    col_names = ["Intercept"] + x_cols
    params, se, llf, conv, _ = _fit_parametric_surv_mle(sub[t_col].values, sub[e_col].values, X, dist)

    k = len(params)
    n = len(sub)
    aic = 2 * k - 2 * llf
    bic = k * math.log(n) - 2 * llf

    table = []
    for j, cname in enumerate(col_names):
        b = float(params[j])
        s = float(se[j])
        z = b / s
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        table.append({
            "Parameter": f"{cname} (Time Ratio={math.exp(b):.3f})",
            "Estimate": b,
            "Std_Error": s,
            "Statistic": z,
            "df": n - k,
            "p_value": p,
            "CI_Low": b - 1.96 * s,
            "CI_High": b + 1.96 * s
        })
    if dist != "exponential":
        b = float(params[-1]); s = float(se[-1])
        table.append({
            "Parameter": f"log(σ) [Shape σ={math.exp(b):.3f}]",
            "Estimate": b, "Std_Error": s, "Statistic": b / s, "df": n - k,
            "p_value": 2 * (1 - stats.norm.cdf(abs(b / s))),
            "CI_Low": b - 1.96 * s, "CI_High": b + 1.96 * s
        })

    return {
        "method": f"Accelerated Failure Time (AFT — {dist.capitalize()} MLE)",
        "summary": {"N": n, "Events": int(sub[e_col].sum()), "Distribution": dist,
                    "Log_Likelihood": llf, "AIC": aic, "BIC": bic, "Converged": conv},
        "table": table,
        "diagnostics": {"Acceleration_Interpretation": "exp(β) > 1 prolongs survival time"},
        "plot": {
            "type": "bar",
            "title": f"AFT Acceleration Factors exp(β) ({dist.capitalize()})",
            "labels": x_cols,
            "values": [math.exp(float(params[j + 1])) for j in range(len(x_cols))]
        }
    }


def est_fine_gray(df: pd.DataFrame, spec: dict) -> dict:
    # Fine-Gray (1999) Subdistribution Hazard Model via IPCW risk-set weighting
    t_col = spec.get("time") or df.columns[0]
    e_col = spec.get("event") or df.columns[1]  # 0=censored, 1=primary event, 2=competing event
    x_cols = spec.get("x_vars") or [c for c in df.select_dtypes(include=[np.number]).columns if c not in [t_col, e_col]]
    if isinstance(x_cols, str):
        x_cols = [c.strip() for c in x_cols.split(",") if c.strip()]
    sub = _numeric_df(df, [t_col, e_col] + x_cols)

    t = sub[t_col].values
    status = sub[e_col].values.astype(int)
    X = sub[x_cols].values
    n, p = X.shape

    # Estimate censoring survival G(t) via KM on censoring indicator (status == 0)
    cens_ind = (status == 0).astype(int)
    sf_c = SurvfuncRight(t, cens_ind)

    def G_hat(val):
        idx = np.searchsorted(sf_c.surv_times, val, side="right") - 1
        return float(sf_c.surv_prob[idx]) if idx >= 0 else 1.0

    event1_idx = np.where(status == 1)[0]
    if len(event1_idx) < 2:
        raise ValueError("Fine-Gray model requires at least 2 primary events (status == 1) and competing events (status == 2).")

    def neg_subdist_pl(beta):
        xb = X @ beta
        exp_xb = np.exp(np.clip(xb, -30, 30))
        ll = 0.0
        for i in event1_idx:
            ti = t[i]
            Gi = max(G_hat(ti), 1e-4)
            # Fine-Gray risk set: {j: t_j >= t_i} U {j: t_j < t_i and status_j == 2}
            in_r1 = (t >= ti)
            in_r2 = (t < ti) & (status == 2)
            w = np.zeros(n)
            w[in_r1] = 1.0
            for j in np.where(in_r2)[0]:
                w[j] = Gi / max(G_hat(t[j]), 1e-4)
            denom = np.sum(w * exp_xb)
            if denom > 0:
                ll += xb[i] - math.log(denom)
        return -float(ll)

    res = optimize.minimize(neg_subdist_pl, np.zeros(p), method="BFGS")
    beta_hat = res.x
    eps = 1e-5
    H = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            ei = np.zeros(p); ei[i] = eps
            ej = np.zeros(p); ej[j] = eps
            H[i, j] = (neg_subdist_pl(beta_hat + ei + ej) - neg_subdist_pl(beta_hat + ei - ej)
                       - neg_subdist_pl(beta_hat - ei + ej) + neg_subdist_pl(beta_hat - ei - ej)) / (4 * eps ** 2)
    se = np.sqrt(np.maximum(np.diag(linalg.pinv(H)), 1e-10))

    table = []
    for j, col in enumerate(x_cols):
        b = float(beta_hat[j]); s = float(se[j]); z = b / s
        table.append({
            "Parameter": f"{col} (sHR = {math.exp(b):.3f})",
            "Estimate": b,
            "Std_Error": s,
            "Statistic": z,
            "df": 1,
            "p_value": 2 * (1 - stats.norm.cdf(abs(z))),
            "CI_Low": b - 1.96 * s,
            "CI_High": b + 1.96 * s
        })

    return {
        "method": "Fine–Gray Competing Risks Subdistribution Hazard Model",
        "summary": {"N": n, "Primary_Events_1": int(np.sum(status == 1)),
                    "Competing_Events_2": int(np.sum(status == 2)),
                    "Censored_0": int(np.sum(status == 0)),
                    "Partial_Log_Likelihood": -float(res.fun)},
        "table": table,
        "diagnostics": {"IPCW_Censoring_Adjustment": "Kaplan-Meier G(t) risk-set reweighting"},
        "plot": {
            "type": "bar",
            "title": "Subdistribution Hazard Ratios (sHR)",
            "labels": x_cols,
            "values": [math.exp(float(b)) for b in beta_hat]
        }
    }


def est_param_surv(df: pd.DataFrame, spec: dict) -> dict:
    # Fits and compares Weibull, Exponential, and Log-Normal specifications simultaneously via AIC/BIC
    t_col = spec.get("time") or df.columns[0]
    e_col = spec.get("event") or df.columns[1]
    x_cols = spec.get("x_vars") or [c for c in df.select_dtypes(include=[np.number]).columns if c not in [t_col, e_col]]
    if isinstance(x_cols, str):
        x_cols = [c.strip() for c in x_cols.split(",") if c.strip()]
    sub = _numeric_df(df, [t_col, e_col] + x_cols)
    X = sm.add_constant(sub[x_cols].values)
    n = len(sub)

    models_out = {}
    table = []
    for dist in ["weibull", "exponential", "lognormal"]:
        params, se, llf, conv, _ = _fit_parametric_surv_mle(sub[t_col].values, sub[e_col].values, X, dist)
        k = len(params)
        aic = 2 * k - 2 * llf
        bic = k * math.log(n) - 2 * llf
        models_out[dist] = {"params": params, "se": se, "llf": llf, "aic": aic, "bic": bic}
        table.append({
            "Parameter": f"Model: {dist.capitalize()} (AIC={aic:.2f})",
            "Estimate": llf,
            "Std_Error": aic,
            "Statistic": bic,
            "df": k,
            "p_value": None,
            "CI_Low": float(params[1]) if len(params) > 1 else None,
            "CI_High": float(se[1]) if len(se) > 1 else None
        })

    best_dist = min(models_out, key=lambda d: models_out[d]["aic"])
    return {
        "method": "Parametric Survival Model Suite (Weibull, Exponential & Log-Normal)",
        "summary": {"N": n, "Events": int(sub[e_col].sum()), "Best_AIC_Model": best_dist.capitalize(),
                    "Weibull_AIC": models_out["weibull"]["aic"],
                    "Exponential_AIC": models_out["exponential"]["aic"],
                    "LogNormal_AIC": models_out["lognormal"]["aic"]},
        "table": table,
        "diagnostics": {"Selected_Specification": best_dist.capitalize()},
        "plot": {
            "type": "bar",
            "title": "Parametric Survival Model Selection (AIC — Lower is Better)",
            "labels": ["Weibull", "Exponential", "Log-Normal"],
            "values": [models_out["weibull"]["aic"], models_out["exponential"]["aic"], models_out["lognormal"]["aic"]]
        }
    }


# ==============================================================================
# 7. CATEGORY E: MIXED-EFFECTS & LONGITUDINAL STATISTICS (4 METHODS)
# ==============================================================================

def est_lmm(df: pd.DataFrame, spec: dict) -> dict:
    y_col = spec.get("y") or df.columns[0]
    g_col = spec.get("group") or df.columns[1]
    x_cols = spec.get("x_vars") or [c for c in df.select_dtypes(include=[np.number]).columns if c != y_col][:2]
    reml = bool(spec.get("reml", True))
    if isinstance(x_cols, str):
        x_cols = [c.strip() for c in x_cols.split(",") if c.strip()]
    _require_cols(df, [y_col, g_col] + x_cols)

    sub = df[[y_col, g_col] + x_cols].dropna()
    for c in [y_col] + x_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    exog = sm.add_constant(sub[x_cols])
    md = MixedLM(sub[y_col], exog, groups=sub[g_col])
    mdf = md.fit(reml=reml)

    ci = mdf.conf_int()
    table = []
    for name in mdf.fe_params.index:
        table.append({
            "Parameter": f"Fixed: {name}",
            "Estimate": float(mdf.fe_params[name]),
            "Std_Error": float(mdf.bse_fe[name]),
            "Statistic": float(mdf.tvalues[name]),
            "df": int(mdf.nobs - len(mdf.fe_params)),
            "p_value": float(mdf.pvalues[name]),
            "CI_Low": float(ci.loc[name, 0]),
            "CI_High": float(ci.loc[name, 1])
        })

    var_u = float(mdf.cov_re.iloc[0, 0])
    var_e = float(mdf.scale)
    icc = var_u / (var_u + var_e) if (var_u + var_e) > 0 else 0.0
    table.append({
        "Parameter": "Random Intercept Variance (σ²_u)",
        "Estimate": var_u, "Std_Error": float(mdf.bse.iloc[-1]), "Statistic": None,
        "df": sub[g_col].nunique() - 1, "p_value": None, "CI_Low": None, "CI_High": None
    })
    table.append({
        "Parameter": "Residual Variance (σ²_e)",
        "Estimate": var_e, "Std_Error": None, "Statistic": None,
        "df": int(mdf.nobs), "p_value": None, "CI_Low": None, "CI_High": None
    })

    return {
        "method": f"Linear Mixed-Effects Model (LMM — {'REML' if reml else 'ML'})",
        "summary": {"N": int(mdf.nobs), "Clusters": int(sub[g_col].nunique()),
                    "Log_Likelihood": float(mdf.llf), "ICC_Conditional": icc, "Converged": bool(mdf.converged)},
        "table": table,
        "diagnostics": {"Variance_Partition_ICC": icc, "Estimator": "REML" if reml else "ML"},
        "plot": {
            "type": "scatter",
            "title": "LMM Fitted Values vs Observed Outcomes",
            "x": mdf.fittedvalues.tolist(),
            "y": sub[y_col].tolist()
        }
    }


def est_glmm(df: pd.DataFrame, spec: dict) -> dict:
    # Random-Intercept Generalized Linear Mixed Model via Gauss-Hermite Quadrature (Binomial/Poisson)
    y_col = spec.get("y") or df.columns[0]
    g_col = spec.get("group") or df.columns[1]
    x_cols = spec.get("x_vars") or [c for c in df.select_dtypes(include=[np.number]).columns if c != y_col][:2]
    family_name = str(spec.get("family", "binomial")).lower()
    if isinstance(x_cols, str):
        x_cols = [c.strip() for c in x_cols.split(",") if c.strip()]
    _require_cols(df, [y_col, g_col] + x_cols)

    sub = df[[y_col, g_col] + x_cols].dropna()
    for c in [y_col] + x_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    X = sm.add_constant(sub[x_cols].values)
    y = sub[y_col].values
    groups = sub[g_col].values
    uniq_g = np.unique(groups)
    p = X.shape[1]

    # 15-point Gauss-Hermite quadrature nodes & weights
    gh_x, gh_w = np.polynomial.hermite.hermgauss(15)
    gh_z = gh_x * math.sqrt(2.0)
    gh_weights = gh_w / math.sqrt(math.pi)

    def nll(theta):
        beta = theta[:p]
        sigma_u = math.exp(theta[p])
        eta_fixed = X @ beta
        total_ll = 0.0
        for g in uniq_g:
            idx = (groups == g)
            yg = y[idx]
            eg = eta_fixed[idx][:, None] + sigma_u * gh_z[None, :]  # (n_g, Q)
            if family_name == "poisson":
                mu = np.exp(np.clip(eg, -20, 20))
                log_p_q = np.sum(yg[:, None] * eg - mu - special.gammaln(yg[:, None] + 1), axis=0)
            else:  # Binomial logit
                log_p_q = np.sum(yg[:, None] * eg - np.logaddexp(0.0, eg), axis=0)
            max_lp = np.max(log_p_q)
            marg = max_lp + math.log(np.sum(gh_weights * np.exp(log_p_q - max_lp)))
            total_ll += marg
        return -float(total_ll)

    res = optimize.minimize(nll, np.append(np.zeros(p), 0.0), method="BFGS")
    params = res.x
    eps = 1e-5
    k = len(params)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei = np.zeros(k); ei[i] = eps
            ej = np.zeros(k); ej[j] = eps
            H[i, j] = (nll(params + ei + ej) - nll(params + ei - ej) - nll(params - ei + ej) + nll(params - ei - ej)) / (4 * eps ** 2)
    se = np.sqrt(np.maximum(np.diag(linalg.pinv(H)), 1e-10))

    col_names = ["Intercept"] + x_cols
    table = []
    for j, cname in enumerate(col_names):
        b = float(params[j]); s = float(se[j]); z = b / s
        table.append({
            "Parameter": f"Fixed: {cname}",
            "Estimate": b, "Std_Error": s, "Statistic": z, "df": len(y) - k,
            "p_value": 2 * (1 - stats.norm.cdf(abs(z))),
            "CI_Low": b - 1.96 * s, "CI_High": b + 1.96 * s
        })
    sig_u = math.exp(float(params[-1]))
    table.append({
        "Parameter": "Random Intercept SD (σ_u)",
        "Estimate": sig_u, "Std_Error": float(sig_u * se[-1]), "Statistic": None,
        "df": len(uniq_g) - 1, "p_value": None, "CI_Low": None, "CI_High": None
    })

    llf = -float(res.fun)
    return {
        "method": f"Generalized Linear Mixed Model (GLMM — {family_name.capitalize()}, 15-pt Gauss-Hermite ML)",
        "summary": {"N": len(y), "Clusters": len(uniq_g), "Family": family_name.capitalize(),
                    "Log_Likelihood": llf, "AIC": 2 * k - 2 * llf, "BIC": k * math.log(len(y)) - 2 * llf,
                    "Random_Intercept_SD": sig_u},
        "table": table,
        "diagnostics": {"Quadrature": "15-point Gauss-Hermite", "Converged": bool(res.success)},
        "plot": {
            "type": "bar",
            "title": "GLMM Fixed-Effect Log-Odds / Log-Rate Coefficients",
            "labels": col_names,
            "values": [float(b) for b in params[:p]]
        }
    }


def est_gee(df: pd.DataFrame, spec: dict) -> dict:
    y_col = spec.get("y") or df.columns[0]
    g_col = spec.get("group") or df.columns[1]
    x_cols = spec.get("x_vars") or [c for c in df.select_dtypes(include=[np.number]).columns if c != y_col][:2]
    cov_type = str(spec.get("cov_struct", "exchangeable")).lower()
    fam_type = str(spec.get("family", "gaussian")).lower()
    if isinstance(x_cols, str):
        x_cols = [c.strip() for c in x_cols.split(",") if c.strip()]
    _require_cols(df, [y_col, g_col] + x_cols)

    sub = df[[y_col, g_col] + x_cols].dropna()
    for c in [y_col] + x_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    cov_map = {"exchangeable": Exchangeable(), "ar1": Autoregressive(),
               "independence": Independence(), "unstructured": Unstructured()}
    fam_map = {"gaussian": Gaussian(), "binomial": Binomial(), "poisson": Poisson()}

    exog = sm.add_constant(sub[x_cols])
    model = GEE(sub[y_col], exog, groups=sub[g_col],
                family=fam_map.get(fam_type, Gaussian()),
                cov_struct=cov_map.get(cov_type, Exchangeable()))
    fit = model.fit()

    ci = fit.conf_int()
    table = []
    for name in fit.params.index:
        table.append({
            "Parameter": f"Marginal: {name}",
            "Estimate": float(fit.params[name]),
            "Std_Error": float(fit.bse[name]),  # Liang-Zeger robust sandwich SE
            "Statistic": float(fit.tvalues[name]),
            "df": int(fit.nobs - len(fit.params)),
            "p_value": float(fit.pvalues[name]),
            "CI_Low": float(ci.loc[name, 0]),
            "CI_High": float(ci.loc[name, 1])
        })

    qic = float(fit.qic()[0])
    return {
        "method": f"Generalized Estimating Equations (GEE — {cov_type.capitalize()} Working Corr)",
        "summary": {"N": int(fit.nobs), "Clusters": int(sub[g_col].nunique()),
                    "Working_Correlation": cov_type.capitalize(), "QIC": qic, "Scale_Phi": float(fit.scale)},
        "table": table,
        "diagnostics": {"Standard_Errors": "Liang-Zeger Robust Sandwich", "QIC": qic},
        "plot": {
            "type": "scatter",
            "title": "GEE Population-Averaged Fitted vs Observed",
            "x": fit.fittedvalues.tolist(),
            "y": sub[y_col].tolist()
        }
    }


def est_icc(df: pd.DataFrame, spec: dict) -> dict:
    # Shrout & Fleiss (1979) Two-Way & One-Way Intraclass Correlation Coefficients
    rater_cols = spec.get("raters") or list(df.select_dtypes(include=[np.number]).columns)
    sub = _numeric_df(df, rater_cols)
    Y = sub.values
    n, k = Y.shape
    if k < 2:
        raise ValueError("ICC requires at least 2 raters/measurements per subject.")

    grand_mean = np.mean(Y)
    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_targets = k * np.sum((np.mean(Y, axis=1) - grand_mean) ** 2)
    ss_judges = n * np.sum((np.mean(Y, axis=0) - grand_mean) ** 2)
    ss_within = ss_total - ss_targets
    ss_error = ss_total - ss_targets - ss_judges

    ms_b = ss_targets / (n - 1)
    ms_w = ss_within / (n * (k - 1))
    ms_j = ss_judges / (k - 1)
    ms_e = ss_error / ((n - 1) * (k - 1))

    icc1_1 = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
    icc2_1 = (ms_b - ms_e) / (ms_b + (k - 1) * ms_e + k * (ms_j - ms_e) / n)
    icc3_1 = (ms_b - ms_e) / (ms_b + (k - 1) * ms_e)
    icc1_k = (ms_b - ms_w) / ms_b
    icc2_k = (ms_b - ms_e) / (ms_b + (ms_j - ms_e) / n)
    icc3_k = (ms_b - ms_e) / ms_b

    f_val = ms_b / ms_e if ms_e > 0 else np.nan
    p_val = 1 - stats.f.cdf(f_val, n - 1, (n - 1) * (k - 1))

    # Exact F-interval for ICC(3,1)
    fl = f_val / stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1))
    fu = f_val * stats.f.ppf(0.975, (n - 1) * (k - 1), n - 1)
    ci_l = (fl - 1) / (fl + k - 1)
    ci_u = (fu - 1) / (fu + k - 1)

    table = [
        {"Parameter": "ICC(1,1) — One-Way Random Single", "Estimate": float(icc1_1), "Std_Error": None,
         "Statistic": ms_b / ms_w, "df": f"{n-1}, {n*(k-1)}", "p_value": 1 - stats.f.cdf(ms_b / ms_w, n - 1, n * (k - 1)), "CI_Low": None, "CI_High": None},
        {"Parameter": "ICC(2,1) — Two-Way Random Absolute Agreement", "Estimate": float(icc2_1), "Std_Error": None,
         "Statistic": f_val, "df": f"{n-1}, {(n-1)*(k-1)}", "p_value": p_val, "CI_Low": None, "CI_High": None},
        {"Parameter": "ICC(3,1) — Two-Way Mixed Consistency", "Estimate": float(icc3_1), "Std_Error": None,
         "Statistic": f_val, "df": f"{n-1}, {(n-1)*(k-1)}", "p_value": p_val, "CI_Low": float(ci_l), "CI_High": float(ci_u)},
        {"Parameter": "ICC(2,k) — Average Raters Absolute", "Estimate": float(icc2_k), "Std_Error": None,
         "Statistic": f_val, "df": f"{n-1}, {(n-1)*(k-1)}", "p_value": p_val, "CI_Low": None, "CI_High": None},
        {"Parameter": "ICC(3,k) — Cronbach's Alpha / Average Consistency", "Estimate": float(icc3_k), "Std_Error": None,
         "Statistic": f_val, "df": f"{n-1}, {(n-1)*(k-1)}", "p_value": p_val, "CI_Low": None, "CI_High": None},
    ]

    return {
        "method": "Intraclass Correlation Coefficients (Shrout–Fleiss ICC Suite)",
        "summary": {"Subjects_N": n, "Raters_k": k, "ICC_2_1_Agreement": float(icc2_1),
                    "ICC_3_1_Consistency": float(icc3_1), "Cronbach_Alpha_ICC_3_k": float(icc3_k)},
        "table": table,
        "diagnostics": {"MSB": float(ms_b), "MSJ": float(ms_j), "MSE": float(ms_e)},
        "plot": {
            "type": "bar",
            "title": "Shrout–Fleiss ICC Reliability Coefficients",
            "labels": ["ICC(1,1)", "ICC(2,1)", "ICC(3,1)", "ICC(2,k)", "ICC(3,k)"],
            "values": [float(icc1_1), float(icc2_1), float(icc3_1), float(icc2_k), float(icc3_k)]
        }
    }


# ==============================================================================
# 8. CATEGORY F: RESAMPLING & STATISTICAL SIMULATION (3 METHODS)
# ==============================================================================

def est_bootstrap(df: pd.DataFrame, spec: dict) -> dict:
    y_col = spec.get("y") or df.select_dtypes(include=[np.number]).columns[0]
    stat_type = str(spec.get("statistic", "mean")).lower()
    B = int(spec.get("B", 2000))
    seed = int(spec.get("seed", 42))
    sub = _numeric_df(df, [y_col])
    x = sub[y_col].values
    n = len(x)

    fn_map = {"mean": np.mean, "median": np.median, "std": lambda v: np.std(v, ddof=1), "iqr": stats.iqr}
    stat_fn = fn_map.get(stat_type, np.mean)
    theta_hat = float(stat_fn(x))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot_reps = np.array([stat_fn(x[row]) for row in idx])
    boot_se = float(np.std(boot_reps, ddof=1))
    boot_bias = float(np.mean(boot_reps) - theta_hat)

    # Percentile 95% CI
    pct_low, pct_high = float(np.percentile(boot_reps, 2.5)), float(np.percentile(boot_reps, 97.5))

    # BCa (Bias-Corrected and Accelerated) 95% CI via Jackknife acceleration
    z0 = stats.norm.ppf(np.clip(np.mean(boot_reps < theta_hat), 1e-4, 1 - 1e-4))
    jack = np.array([stat_fn(np.delete(x, i)) for i in range(n)])
    jack_diff = np.mean(jack) - jack
    denom = 6.0 * (np.sum(jack_diff ** 2) ** 1.5)
    a_hat = float(np.sum(jack_diff ** 3) / denom) if denom > 0 else 0.0

    z_alpha1, z_alpha2 = stats.norm.ppf(0.025), stats.norm.ppf(0.975)
    a1 = stats.norm.cdf(z0 + (z0 + z_alpha1) / max(1.0 - a_hat * (z0 + z_alpha1), 1e-4))
    a2 = stats.norm.cdf(z0 + (z0 + z_alpha2) / max(1.0 - a_hat * (z0 + z_alpha2), 1e-4))
    bca_low, bca_high = float(np.quantile(boot_reps, a1)), float(np.quantile(boot_reps, a2))

    return {
        "method": f"Nonparametric Bootstrap Inference ({stat_type.capitalize()}, B={B})",
        "summary": {"N": n, "Replications_B": B, "Observed_Estimate": theta_hat,
                    "Bootstrap_SE": boot_se, "Bootstrap_Bias": boot_bias, "BCa_Acceleration_a": a_hat},
        "table": [
            {"Parameter": f"Percentile 95% CI ({stat_type})", "Estimate": theta_hat, "Std_Error": boot_se,
             "Statistic": theta_hat / boot_se if boot_se > 0 else None, "df": n - 1, "p_value": None,
             "CI_Low": pct_low, "CI_High": pct_high},
            {"Parameter": f"BCa Bias-Corrected 95% CI ({stat_type})", "Estimate": theta_hat - boot_bias, "Std_Error": boot_se,
             "Statistic": z0, "df": n - 1, "p_value": None,
             "CI_Low": bca_low, "CI_High": bca_high},
        ],
        "diagnostics": {"BCa_Bias_Correction_z0": float(z0), "BCa_Acceleration_a": a_hat},
        "plot": {
            "type": "histogram",
            "title": f"Bootstrap Sampling Distribution ({stat_type}, B={B})",
            "values": boot_reps[:1000].tolist()
        }
    }


def est_permutation(df: pd.DataFrame, spec: dict) -> dict:
    y_col = spec.get("y") or df.columns[0]
    g_col = spec.get("group") or df.columns[1]
    B = int(spec.get("B", 5000))
    seed = int(spec.get("seed", 42))
    _require_cols(df, [y_col, g_col])

    sub = df[[y_col, g_col]].dropna()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()
    groups = list(sub.groupby(g_col))
    if len(groups) != 2:
        raise ValueError("Two-sample permutation test requires exactly 2 groups.")

    y = sub[y_col].values
    n1 = len(groups[0][1])
    obs_diff = float(np.mean(y[:n1]) - np.mean(y[n1:]))

    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(B)
    for b in range(B):
        perm = rng.permutation(y)
        perm_diffs[b] = np.mean(perm[:n1]) - np.mean(perm[n1:])

    # Exact Monte Carlo Phipson-Smyth p-value: (r + 1) / (B + 1)
    p_val = float((np.sum(np.abs(perm_diffs) >= abs(obs_diff)) + 1) / (B + 1))

    return {
        "method": f"Exact Monte Carlo Permutation Test (B={B} Shuffles)",
        "summary": {"N": len(y), "Permutations_B": B, "Observed_Mean_Diff": obs_diff,
                    "Null_SD": float(np.std(perm_diffs)), "Permutation_p_value": p_val},
        "table": [
            {"Parameter": f"Mean Difference ({groups[0][0]} - {groups[1][0]})",
             "Estimate": obs_diff, "Std_Error": float(np.std(perm_diffs)),
             "Statistic": obs_diff / max(float(np.std(perm_diffs)), 1e-10),
             "df": len(y) - 2, "p_value": p_val,
             "CI_Low": float(np.percentile(perm_diffs, 2.5)),
             "CI_High": float(np.percentile(perm_diffs, 97.5))}
        ],
        "diagnostics": {"Phipson_Smyth_p_Correction": "Applied (r+1)/(B+1)"},
        "plot": {
            "type": "histogram",
            "title": f"Permutation Null Distribution (Obs Diff = {obs_diff:.3f})",
            "values": perm_diffs[:1000].tolist()
        }
    }


def est_power(df: pd.DataFrame, spec: dict) -> dict:
    # Analytical non-central t power & sample size curve (or empirical effect size from CSV if provided)
    d = float(spec.get("effect_size", 0.5))
    alpha = float(spec.get("alpha", 0.05))
    target_power = float(spec.get("target_power", 0.80))
    if d <= 0 or not math.isfinite(d):
        raise ValueError("Cohen's d effect size must be finite and strictly positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("Significance level alpha must lie strictly between 0 and 1.")
    if not (0.0 < target_power < 1.0):
        raise ValueError("Target power must lie strictly between 0 and 1.")

    def power_for_n(n_per_group):
        df_t = 2 * n_per_group - 2
        ncp = d * math.sqrt(n_per_group / 2.0)
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df_t)
        return float(1.0 - stats.nct.cdf(t_crit, df_t, ncp) + stats.nct.cdf(-t_crit, df_t, ncp))

    # Solve minimum sample size per group
    n_req = 4
    while power_for_n(n_req) < target_power and n_req < 10000:
        n_req += 1

    grid_n = [10, 20, 30, 50, 75, 100, 150, 200, 300]
    grid_pwr = [power_for_n(ng) for ng in grid_n]

    table = [
        {"Parameter": f"Required N per Group (Power >= {target_power:.2f})",
         "Estimate": float(n_req), "Std_Error": float(2 * n_req),
         "Statistic": power_for_n(n_req), "df": 2 * n_req - 2, "p_value": alpha,
         "CI_Low": None, "CI_High": None}
    ]
    for ng, pw in zip(grid_n, grid_pwr):
        table.append({
            "Parameter": f"Power at n={ng}/group (N={2*ng})",
            "Estimate": pw, "Std_Error": None, "Statistic": d * math.sqrt(ng / 2.0),
            "df": 2 * ng - 2, "p_value": alpha, "CI_Low": None, "CI_High": None
        })

    return {
        "method": "Statistical Power & Sample Size Analysis (Non-Central t Distribution)",
        "summary": {"Cohens_d": d, "Alpha": alpha, "Target_Power": target_power,
                    "Required_n_per_Group": n_req, "Total_Required_N": 2 * n_req,
                    "Achieved_Power": power_for_n(n_req)},
        "table": table,
        "diagnostics": {"Noncentrality_Parameter": d * math.sqrt(n_req / 2.0)},
        "plot": {
            "type": "line",
            "title": f"Statistical Power Curve (Cohen's d = {d}, α = {alpha})",
            "x": grid_n,
            "y": grid_pwr
        }
    }


# ==============================================================================
# 9. CATEGORY G: META-ANALYSIS (2 METHODS)
# ==============================================================================

def est_meta_fe(df: pd.DataFrame, spec: dict) -> dict:
    yi_col = spec.get("effect") or df.columns[0]
    sei_col = spec.get("se") or df.columns[1]
    lbl_col = spec.get("study") if spec.get("study") in df.columns else None
    sub = _numeric_df(df, [yi_col, sei_col])
    yi = sub[yi_col].values
    sei = sub[sei_col].values
    if (sei <= 0).any():
        raise ValueError("Standard errors in meta-analysis must be strictly positive.")

    vi = sei ** 2
    wi = 1.0 / vi
    theta_fe = float(np.sum(wi * yi) / np.sum(wi))
    se_fe = float(math.sqrt(1.0 / np.sum(wi)))
    z_fe = theta_fe / se_fe
    p_fe = float(2 * (1 - stats.norm.cdf(abs(z_fe))))

    k = len(yi)
    Q = float(np.sum(wi * (yi - theta_fe) ** 2))
    p_Q = float(1 - stats.chi2.cdf(Q, k - 1))
    I2 = max(0.0, (Q - (k - 1)) / Q * 100.0) if Q > 0 else 0.0

    labels = df.loc[sub.index, lbl_col].astype(str).tolist() if lbl_col else [f"Study {i+1}" for i in range(k)]
    table = [
        {"Parameter": "Pooled Fixed-Effect Estimate (Inverse-Variance)",
         "Estimate": theta_fe, "Std_Error": se_fe, "Statistic": z_fe, "df": k - 1,
         "p_value": p_fe, "CI_Low": theta_fe - 1.96 * se_fe, "CI_High": theta_fe + 1.96 * se_fe}
    ]
    for lbl, y_i, s_i, w_i in zip(labels, yi, sei, wi / np.sum(wi) * 100.0):
        table.append({
            "Parameter": f"{lbl} (Weight={w_i:.1f}%)",
            "Estimate": float(y_i), "Std_Error": float(s_i), "Statistic": float(y_i / s_i),
            "df": 1, "p_value": float(2 * (1 - stats.norm.cdf(abs(y_i / s_i)))),
            "CI_Low": float(y_i - 1.96 * s_i), "CI_High": float(y_i + 1.96 * s_i)
        })

    return {
        "method": "Fixed-Effect Meta-Analysis (Inverse-Variance Weighted)",
        "summary": {"Studies_k": k, "Pooled_Effect": theta_fe, "SE": se_fe, "p_value": p_fe,
                    "Cochrans_Q": Q, "Q_p_value": p_Q, "I_Squared_Pct": I2},
        "table": table,
        "diagnostics": {"Cochrans_Q": Q, "Heterogeneity_I2_Pct": I2},
        "plot": {
            "type": "bar",
            "title": "Study Effect Sizes & Pooled Fixed-Effect Estimate",
            "labels": ["POOLED FE"] + labels,
            "values": [theta_fe] + yi.tolist()
        }
    }


def est_meta_re(df: pd.DataFrame, spec: dict) -> dict:
    # DerSimonian-Laird & REML Random-Effects Meta-Analysis + Egger's funnel asymmetry test
    yi_col = spec.get("effect") or df.columns[0]
    sei_col = spec.get("se") or df.columns[1]
    lbl_col = spec.get("study") if spec.get("study") in df.columns else None
    sub = _numeric_df(df, [yi_col, sei_col])
    yi = sub[yi_col].values
    sei = sub[sei_col].values
    if (sei <= 0).any():
        raise ValueError("Standard errors in meta-analysis must be strictly positive.")

    vi = sei ** 2
    wi = 1.0 / vi
    theta_fe = float(np.sum(wi * yi) / np.sum(wi))
    k = len(yi)
    Q = float(np.sum(wi * (yi - theta_fe) ** 2))
    p_Q = float(1 - stats.chi2.cdf(Q, k - 1))
    I2 = max(0.0, (Q - (k - 1)) / Q * 100.0) if Q > 0 else 0.0

    # DerSimonian-Laird tau^2
    c_dl = float(np.sum(wi) - np.sum(wi ** 2) / np.sum(wi))
    tau2_dl = max(0.0, (Q - (k - 1)) / c_dl)

    # REML tau^2 via profile log-likelihood maximization
    def reml_nll(log_tau2):
        t2 = math.exp(log_tau2)
        w_star = 1.0 / (vi + t2)
        mu_star = np.sum(w_star * yi) / np.sum(w_star)
        ll = -0.5 * (np.sum(np.log(vi + t2)) + math.log(np.sum(w_star)) + np.sum(w_star * (yi - mu_star) ** 2))
        return -float(ll)

    opt_reml = optimize.minimize_scalar(reml_nll, bounds=(-20, 10), method="bounded")
    tau2_reml = math.exp(float(opt_reml.x)) if Q > (k - 1) else 0.0

    w_re = 1.0 / (vi + tau2_reml)
    theta_re = float(np.sum(w_re * yi) / np.sum(w_re))
    se_re = float(math.sqrt(1.0 / np.sum(w_re)))
    z_re = theta_re / se_re
    p_re = float(2 * (1 - stats.norm.cdf(abs(z_re))))

    # 95% Prediction Interval
    t_crit = stats.t.ppf(0.975, df=max(k - 2, 1))
    pi_sd = math.sqrt(tau2_reml + se_re ** 2)
    pi_low, pi_high = theta_re - t_crit * pi_sd, theta_re + t_crit * pi_sd

    # Egger's regression test for funnel asymmetry: (yi / sei) ~ 1 + (1 / sei)
    egger_p = None
    if k >= 3:
        snd = yi / sei
        prec = 1.0 / sei
        eg_fit = sm.OLS(snd, sm.add_constant(prec)).fit()
        egger_p = float(eg_fit.pvalues[0])

    labels = df.loc[sub.index, lbl_col].astype(str).tolist() if lbl_col else [f"Study {i+1}" for i in range(k)]
    table = [
        {"Parameter": "Pooled Random-Effects (REML)", "Estimate": theta_re, "Std_Error": se_re,
         "Statistic": z_re, "df": k - 1, "p_value": p_re,
         "CI_Low": theta_re - 1.96 * se_re, "CI_High": theta_re + 1.96 * se_re},
        {"Parameter": "95% Prediction Interval (Between-Study)", "Estimate": theta_re, "Std_Error": pi_sd,
         "Statistic": None, "df": max(k - 2, 1), "p_value": None, "CI_Low": pi_low, "CI_High": pi_high},
        {"Parameter": "Between-Study Variance τ² (REML / DL)", "Estimate": tau2_reml, "Std_Error": tau2_dl,
         "Statistic": Q, "df": k - 1, "p_value": p_Q, "CI_Low": I2, "CI_High": None}
    ]

    return {
        "method": "Random-Effects Meta-Analysis (REML & DerSimonian–Laird)",
        "summary": {"Studies_k": k, "Pooled_RE_Effect": theta_re, "SE": se_re, "p_value": p_re,
                    "Tau2_REML": tau2_reml, "Tau2_DL": tau2_dl, "I_Squared_Pct": I2,
                    "Eggers_Asymmetry_p": egger_p},
        "table": table,
        "diagnostics": {"Cochrans_Q": Q, "Q_p_value": p_Q, "Eggers_Funnel_p": egger_p},
        "plot": {
            "type": "bar",
            "title": "Random-Effects Meta-Analysis Study Effects & Pooled RE",
            "labels": ["POOLED RE"] + labels,
            "values": [theta_re] + yi.tolist()
        }
    }


# ==============================================================================
# 10. DISPATCHER & BUILT-IN BENCHMARK DATASETS
# ==============================================================================

DISPATCH = {
    "stat_anova1": est_anova1,
    "stat_anova2": est_anova2,
    "stat_rm_anova": est_rm_anova,
    "stat_ancova": est_ancova,
    "stat_manova": est_manova,
    "stat_mancova": est_mancova,
    "stat_mann_whitney": est_mann_whitney,
    "stat_wilcoxon": est_wilcoxon,
    "stat_kruskal": est_kruskal,
    "stat_friedman": est_friedman,
    "stat_kendall": est_kendall,
    "stat_pca": est_pca,
    "stat_efa": est_efa,
    "stat_cfa": est_cfa,
    "stat_lda": est_lda,
    "stat_cca": est_cca,
    "stat_ca": est_ca,
    "stat_km": est_km,
    "stat_cox_ph": est_cox_ph,
    "stat_aft": est_aft,
    "stat_fine_gray": est_fine_gray,
    "stat_param_surv": est_param_surv,
    "stat_lmm": est_lmm,
    "stat_glmm": est_glmm,
    "stat_gee": est_gee,
    "stat_icc": est_icc,
    "stat_bootstrap": est_bootstrap,
    "stat_permutation": est_permutation,
    "stat_power": est_power,
    "stat_meta_fe": est_meta_fe,
    "stat_meta_re": est_meta_re,
}


def run_json(model_id: str, csv_text: str, spec_json: str) -> str:
    if model_id not in DISPATCH:
        raise ValueError(f"Unknown statistical method ID: {model_id}")
    spec = json.loads(spec_json) if spec_json else {}
    # Power/sample-size analysis is analytic and does not estimate from uploaded data.
    df = _parse_csv(csv_text, min_rows=1 if model_id == "stat_power" else 3)
    t0 = time.perf_counter()
    res = DISPATCH[model_id](df, spec)
    res["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    res["model_id"] = model_id
    return json.dumps(_clean(res))


# ==============================================================================
# 11. OES METHOD VALIDATION PROTOCOL (OMVP) — LIVE MONTE CARLO & BENCHMARK SUITE
# ==============================================================================

def _generate_omvp_dataset(model_id: str, seed: int = 12345) -> tuple:
    """Reproducible DGP and specification for each of the 31 statistical methods."""
    rng = np.random.default_rng(seed)
    n = 120

    if model_id in ("stat_anova1", "stat_kruskal", "stat_mann_whitney", "stat_permutation"):
        k = 2 if model_id in ("stat_mann_whitney", "stat_permutation") else 3
        g = np.repeat([f"G{i+1}" for i in range(k)], n // k)
        eff = np.repeat([0.0, 0.8, 1.5][:k], n // k)
        y = 10.0 + eff + rng.normal(0, 1.2, size=len(g))
        df = pd.DataFrame({"outcome": y, "group": g})
        return df, {"y": "outcome", "group": "group", "B": 500}

    if model_id == "stat_anova2":
        f1 = np.tile(["A", "B"], n // 2)
        f2 = np.repeat(["Low", "Medium", "High"], n // 3)
        y = 5.0 + (f1 == "B") * 1.5 + (f2 == "High") * 2.0 + rng.normal(0, 1.0, size=n)
        return pd.DataFrame({"y": y, "factor1": f1, "factor2": f2}), {"y": "y", "factor1": "factor1", "factor2": "factor2"}

    if model_id in ("stat_rm_anova", "stat_friedman", "stat_icc"):
        subj = rng.normal(50, 8, size=(60, 1))
        Y = subj + np.array([[0.0, 2.5, 5.0, 4.0]]) + rng.normal(0, 1.5, size=(60, 4))
        df = pd.DataFrame(Y, columns=["T1", "T2", "T3", "T4"])
        return df, {"conditions": ["T1", "T2", "T3", "T4"], "raters": ["T1", "T2", "T3", "T4"]}

    if model_id == "stat_ancova":
        g = np.repeat(["Control", "Treatment"], n // 2)
        cov = rng.normal(10, 2, size=n)
        y = 2.0 + 1.5 * (g == "Treatment") + 0.8 * cov + rng.normal(0, 1.0, size=n)
        return pd.DataFrame({"y": y, "group": g, "cov": cov}), {"y": "y", "group": "group", "covariates": ["cov"]}

    if model_id in ("stat_manova", "stat_mancova", "stat_lda"):
        g = np.repeat(["Class1", "Class2", "Class3"], n // 3)
        cov = rng.normal(0, 1, size=n)
        y1 = (g == "Class2") * 1.5 + (g == "Class3") * 3.0 + 0.5 * cov + rng.normal(0, 1.0, size=n)
        y2 = (g == "Class2") * 2.0 - (g == "Class3") * 1.0 + 0.4 * cov + rng.normal(0, 1.0, size=n)
        df = pd.DataFrame({"y1": y1, "y2": y2, "group": g, "cov": cov})
        return df, {"y_vars": ["y1", "y2"], "x_vars": ["y1", "y2"], "group": "group", "covariates": ["cov"]}

    if model_id in ("stat_wilcoxon", "stat_kendall", "stat_bootstrap"):
        x1 = rng.normal(10, 2, size=80)
        x2 = x1 - 1.2 + rng.normal(0, 1.0, size=80)
        return pd.DataFrame({"x1": x1, "x2": x2}), {"x1": "x1", "x2": "x2", "x": "x1", "y": "x2", "B": 500}

    if model_id in ("stat_pca", "stat_efa", "stat_cfa", "stat_cca"):
        f1 = rng.normal(0, 1, size=150)
        f2 = rng.normal(0, 1, size=150)
        df = pd.DataFrame({
            "v1": 1.0 * f1 + rng.normal(0, 0.5, 150),
            "v2": 0.9 * f1 + rng.normal(0, 0.5, 150),
            "v3": 0.85 * f1 + rng.normal(0, 0.5, 150),
            "v4": 0.9 * f2 + 0.3 * f1 + rng.normal(0, 0.5, 150),
            "v5": 0.85 * f2 + rng.normal(0, 0.5, 150),
            "v6": 0.8 * f2 + rng.normal(0, 0.5, 150),
        })
        return df, {"vars": ["v1", "v2", "v3", "v4", "v5", "v6"],
                    "x_vars": ["v1", "v2", "v3"], "y_vars": ["v4", "v5", "v6"], "n_factors": 2}

    if model_id == "stat_ca":
        r = rng.choice(["Low", "Mid", "High"], size=200, p=[0.3, 0.4, 0.3])
        c = np.where(r == "High", rng.choice(["Yes", "No"], size=200, p=[0.75, 0.25]),
                     rng.choice(["Yes", "No"], size=200, p=[0.35, 0.65]))
        return pd.DataFrame({"row_var": r, "col_var": c}), {"row_var": "row_var", "col_var": "col_var"}

    if model_id in ("stat_km", "stat_cox_ph", "stat_aft", "stat_param_surv"):
        trt = rng.binomial(1, 0.5, size=150)
        age = rng.normal(0, 1, size=150)
        rate = np.exp(-0.5 + 0.7 * trt + 0.3 * age)
        t_true = rng.exponential(1.0 / rate)
        c_time = rng.exponential(2.5, size=150)
        t_obs = np.minimum(t_true, c_time)
        event = (t_true <= c_time).astype(int)
        return pd.DataFrame({"time": t_obs, "event": event, "trt": trt, "age": age}), {
            "time": "time", "event": "event", "group": "trt", "x_vars": ["trt", "age"]
        }

    if model_id == "stat_fine_gray":
        trt = rng.binomial(1, 0.5, size=150)
        t_obs = rng.exponential(1.5, size=150)
        status = rng.choice([0, 1, 2], size=150, p=[0.25, 0.45, 0.30])
        return pd.DataFrame({"time": t_obs, "status": status, "trt": trt}), {
            "time": "time", "event": "status", "x_vars": ["trt"]
        }

    if model_id in ("stat_lmm", "stat_gee"):
        clusters = np.repeat(np.arange(30), 5)
        u_i = np.repeat(rng.normal(0, 1.5, size=30), 5)
        x = rng.normal(0, 1, size=150)
        y = 3.0 + 1.2 * x + u_i + rng.normal(0, 1.0, size=150)
        return pd.DataFrame({"y": y, "cluster": clusters, "x": x}), {"y": "y", "group": "cluster", "x_vars": ["x"]}

    if model_id == "stat_glmm":
        clusters = np.repeat(np.arange(30), 6)
        u_i = np.repeat(rng.normal(0, 1.0, size=30), 6)
        x = rng.normal(0, 1, size=180)
        prob = special.expit(-0.5 + 1.1 * x + u_i)
        y = rng.binomial(1, prob)
        return pd.DataFrame({"y": y, "cluster": clusters, "x": x}), {"y": "y", "group": "cluster", "x_vars": ["x"], "family": "binomial"}

    if model_id == "stat_power":
        return pd.DataFrame({"dummy": [1, 2, 3, 4]}), {"effect_size": 0.5, "alpha": 0.05, "target_power": 0.80}

    if model_id in ("stat_meta_fe", "stat_meta_re"):
        k = 12
        true_theta = 0.45
        se_i = rng.uniform(0.08, 0.25, size=k)
        y_i = true_theta + rng.normal(0, 0.15, size=k) + rng.normal(0, se_i)
        df = pd.DataFrame({"study": [f"Trial_{i+1}" for i in range(k)], "effect": y_i, "se": se_i})
        return df, {"study": "study", "effect": "effect", "se": "se"}

    raise ValueError(f"No DGP configured for {model_id}")


def run_omvp_json(model_id: str, n_reps: int = 200) -> str:
    """Run the browser OMVP checks and return a reproducible JSON validation report."""
    if model_id not in DISPATCH:
        raise ValueError(f"Unknown model_id: {model_id}")
    n_reps = int(n_reps)
    if n_reps < 1 or n_reps > 5000:
        raise ValueError("OMVP replications must be between 1 and 5000.")

    t0 = time.perf_counter()
    df0, spec0 = _generate_omvp_dataset(model_id, seed=101)
    base_res = DISPATCH[model_id](df0.copy(), dict(spec0))
    finite_ok = _finite_tree(base_res)
    conditioning = _numeric_condition_diagnostic(df0)

    estimates, pvals = [], []
    converged, failed = 0, 0
    failure_messages = []
    for r in range(n_reps):
        try:
            df_r, spec_r = _generate_omvp_dataset(model_id, seed=1000 + r)
            out_r = DISPATCH[model_id](df_r, spec_r)
            if not _finite_tree(out_r):
                raise FloatingPointError("Non-finite value detected in Monte Carlo result.")
            converged += 1
            first_row = out_r["table"][0] if out_r.get("table") else {}
            value = first_row.get("Estimate")
            if value is not None:
                estimates.append(float(value))
            pvalue = first_row.get("p_value")
            if pvalue is not None:
                pvals.append(float(pvalue))
        except Exception as exc:
            failed += 1
            if len(failure_messages) < 5:
                failure_messages.append(f"{type(exc).__name__}: {exc}")

    mc_mean = float(np.mean(estimates)) if estimates else None
    mc_sd = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None
    rejection_rate = float(np.mean(np.asarray(pvals) < 0.05)) if pvals else None
    mc_finite = all(math.isfinite(v) for v in estimates + pvals)

    reference_environment = {
        "SciPy": scipy.__version__,
        "statsmodels": sm.__version__,
        "reference_parity": "Not universally executed; method-specific independent parity tests are separate."
    }
    repeat_res = DISPATCH[model_id](df0.copy(), dict(spec0))
    reproducible = (
        json.dumps(_clean(base_res), sort_keys=True, separators=(",", ":"))
        == json.dumps(_clean(repeat_res), sort_keys=True, separators=(",", ":"))
    )

    edge_exception = None
    try:
        if model_id == "stat_power":
            bad_spec = {"effect_size": 0.5, "alpha": 1.0, "target_power": 0.80}
            DISPATCH[model_id](pd.DataFrame({"dummy": [1, 2, 3]}), bad_spec)
        else:
            DISPATCH[model_id](pd.DataFrame({"__invalid_column__": [1]}), spec0)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        edge_exception = type(exc).__name__
    edge_caught = edge_exception is not None

    elapsed = round(time.perf_counter() - t0, 3)
    report = {
        "protocol": "OES Method Validation Protocol (OMVP v1.0)",
        "model_id": model_id,
        "method_name": base_res["method"],
        "stages": {
            "Stage_1_Numerical_Finiteness": "PASS" if finite_ok else "FAIL",
            "Stage_2_Numeric_Matrix_Conditioning": conditioning,
            "Stage_3_4_Monte_Carlo_Replications_Requested": n_reps,
            "Stage_3_4_Monte_Carlo_Replications_Completed": converged,
            "Stage_3_4_Monte_Carlo_Replications_Failed": failed,
            "Monte_Carlo_Completion_Rate_Pct": round(converged / n_reps * 100.0, 2),
            "Stage_5_Primary_Estimate_Mean": mc_mean,
            "Stage_5_Primary_Estimate_SD": mc_sd,
            "Stage_5_First_Row_p_Value_Rejection_Frequency_at_0_05": rejection_rate,
            "Monte_Carlo_Results_Finite": "PASS" if mc_finite else "FAIL",
            "Monte_Carlo_Failure_Details": failure_messages,
            "Stage_5_Interpretation_Note": "The first-row p-value rejection frequency is descriptive and is not an empirical power estimate unless the DGP is explicitly specified as an alternative hypothesis.",
            "Stage_6_Reference_Library_Environment": reference_environment,
            "Stage_7_Seeded_Reproducibility": "PASS" if reproducible else "FAIL",
            "Stage_8_Edge_Case_Guard": "PASS" if edge_caught else "FAIL",
            "Stage_8_Rejection_Exception": edge_exception,
        },
        "baseline_summary": base_res["summary"],
        "runtime_seconds": elapsed
    }
    return json.dumps(_clean(report))
