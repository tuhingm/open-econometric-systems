"""Open Econometric Systems estimation engine.

Pure Python (NumPy, pandas, SciPy, statsmodels, scikit-learn, XGBoost), so the same code runs on a server, on a laptop
and in the browser under Pyodide. Entry points:

    run(model_id, csv_text, spec)   -> dict (or JSON string with run_json) in the common result format (core.py)
    catalog()                       -> list of the models with their input fields and the guide example
    prepare(df, ops)                -> data frame after the data operations (lags, differences, logs, growth, sample)
"""
import json
import warnings

import numpy as np
import pandas as pd

from . import bayes, causal, lp, markov, ml, panel, pvar, statespace_var, timeseries
from .core import OESError, read_csv, time_order
from .report import frame, plot, show, summary, tables

__version__ = "1.2.0"

# field helpers -------------------------------------------------------------------------------------------------
def col(key, label, required=True, help=""):
    return {"key": key, "label": label, "type": "col", "required": required, "help": help}


def cols(key, label, required=True, help=""):
    return {"key": key, "label": label, "type": "cols", "required": required, "help": help}


def num(key, label, default, help=""):
    return {"key": key, "label": label, "type": "num", "default": default, "help": help}


def sel(key, label, options, default, help=""):
    return {"key": key, "label": label, "type": "select", "options": options, "default": default, "help": help}


TIME = col("time", "Time / date column (sorts the data)", required=False)
UNIT = col("unit", "Unit (entity) identifier")
PTIME = col("time", "Time identifier")
PCOV = sel("cov", "Standard errors", [["clustered", "Clustered by unit"], ["robust", "Robust (White)"], ["unadjusted", "Conventional"],
                                      ["dk", "Driscoll-Kraay"]], "clustered")
BW = num("bandwidth", "Driscoll-Kraay bandwidth (blank = automatic)", "")

TS = "research-guide/examples/"
PD = "panel-data-guide/examples/"
CI = "causal-inference-guide/examples/"
MLD = "ml-guide/examples/"
LAG4 = [1, 2, 3, 4]

MODELS = [
    # ------------------------------------------------------------------ time series
    dict(id="ols", group="Time series", name="OLS with diagnostics", guide="research-guide/ols.html", fn=timeseries.ols,
         fields=[col("y", "Dependent variable"), cols("x", "Explanatory variables"), TIME,
                 sel("cov", "Standard errors", [["HC1", "Robust HC1"], ["HC0", "Robust HC0"], ["HC2", "Robust HC2"], ["HC3", "Robust HC3"],
                                                 ["HAC", "Newey-West HAC"], ["nonrobust", "Conventional"]], "HC1"),
                 num("hac_lags", "HAC lags", 4)],
         example=dict(file=TS + "returns.csv", spec=dict(y="sp500_return", x=["ip_growth", "d_fedfunds", "d_credit_spread"], time="date", cov="HC1"))),
    dict(id="ardl", group="Time series", name="ARDL bounds test", guide="research-guide/ardl.html", fn=timeseries.ardl,
         fields=[col("y", "Dependent variable"), cols("x", "Regressors"), TIME, num("max_lag", "Maximum lag of y", 4),
                 num("max_order", "Maximum lag of regressors", 4), sel("ic", "Lag selection", [["aic", "AIC"], ["bic", "BIC"], ["hqic", "HQIC"]], "aic"),
                 sel("case", "Bounds-test case", [["3", "3: unrestricted constant"], ["2", "2: restricted constant"], ["5", "5: unrestricted trend"],
                                                  ["4", "4: restricted trend"], ["1", "1: no deterministic terms"]], "3")],
         example=dict(file=TS + "yield.csv", spec=dict(y="yield_10y", x=["inflation", "gdp_growth", "unemployment"], time="date"))),
    dict(id="nardl", group="Time series", name="Nonlinear ARDL (NARDL)", guide="research-guide/nardl.html", fn=timeseries.nardl,
         fields=[col("y", "Dependent variable"), cols("x", "Asymmetric regressor(s)"), cols("controls", "Symmetric controls", required=False), TIME,
                 num("max_lag", "Maximum lag of y", 4), num("max_order", "Maximum lag of regressors", 4),
                 sel("ic", "Lag selection", [["aic", "AIC"], ["bic", "BIC"]], "aic"), sel("case", "Bounds-test case", [["3", "3"], ["2", "2"], ["5", "5"]], "3")],
         example=dict(file=TS + "passthrough.csv", spec=dict(y="baa_yield", x=["fedfunds"], time="date", max_lag=6, max_order=6))),
    dict(id="var", group="Time series", name="Vector autoregression (VAR)", guide="research-guide/var.html", fn=timeseries.var,
         fields=[cols("vars", "Variables (Cholesky order)"), TIME, sel("transform", "Transformation", [["levels", "Levels (as given)"], ["diff", "First differences"]], "levels"),
                 num("max_lags", "Maximum lag for selection", 8), num("lags", "Lag order (blank = selected)", ""),
                 sel("ic", "Lag selection", [["aic", "AIC"], ["bic", "BIC"], ["hqic", "HQIC"], ["fpe", "FPE"]], "aic"), num("horizon", "IRF horizon", 12)],
         example=dict(file=TS + "prices.csv", spec=dict(vars=["log_sp500", "log_indpro"], time="date", transform="diff", max_lags=12))),
    dict(id="vecm", group="Time series", name="Vector error correction (VECM)", guide="research-guide/vecm.html", fn=timeseries.vecm,
         fields=[cols("vars", "Variables"), TIME,
                 sel("deterministic", "Deterministic terms", [["ci", "Constant in the cointegrating relation"], ["co", "Constant outside"],
                                                               ["li", "Trend in the cointegrating relation"], ["lo", "Linear trend outside"], ["n", "None"]], "ci"),
                 num("max_lags", "Maximum lag for selection", 8), num("lags", "Lags in differences (blank = selected)", ""),
                 num("rank", "Cointegration rank (blank = trace test)", ""), num("horizon", "IRF horizon", 12)],
         example=dict(file=TS + "rates.csv", spec=dict(vars=["yield_10y", "yield_1y"], time="date", deterministic="ci"))),
    dict(id="iv_gmm", group="Time series", name="2SLS and GMM", guide="research-guide/gmm.html", fn=timeseries.iv_gmm,
         fields=[col("y", "Dependent variable"), cols("endog", "Endogenous regressors"), cols("instruments", "Excluded instruments"),
                 cols("exog", "Exogenous regressors (included)", required=False), TIME,
                 sel("method", "Estimator", [["2sls", "2SLS"], ["gmm", "Two-step efficient GMM"]], "2sls"),
                 sel("cov", "Standard errors", [["kernel", "HAC (Bartlett)"], ["robust", "Robust"], ["unadjusted", "Conventional"]], "kernel"),
                 num("bandwidth", "HAC bandwidth", 4)],
         example=dict(file=TS + "taylor.csv",
                      ops=[dict(op="lag", vars=["fedfunds", "inflation", "output_gap"], lags=LAG4, time="date"),
                           dict(op="sample", time="date", start="1979Q3", end="2007Q4")],
                      spec=dict(y="fedfunds", endog=["inflation", "output_gap"], exog=["fedfunds_l1"], time="date",
                                instruments=["inflation_l1", "inflation_l2", "inflation_l3", "inflation_l4", "output_gap_l1", "output_gap_l2",
                                             "output_gap_l3", "output_gap_l4", "fedfunds_l2"], method="gmm", cov="kernel", bandwidth=4))),
    dict(id="lp", group="Time series", name="Local projections (impulse responses)", guide="research-guide/lp.html", fn=lp.lp,
         fields=[col("y", "Response variable"), col("shock", "Shock variable"),
                 cols("contemp", "Contemporaneous controls (ordered before the shock)", required=False),
                 cols("controls", "Other controls (lags only)", required=False),
                 cols("instrument", "Instrument(s) for the shock (LP-IV)", required=False), TIME,
                 num("lags", "Lags of every variable", 4), num("horizon", "Horizon", 12),
                 sel("outcome", "Outcome", [["level", "Level: y(t+h)"], ["diff", "Change since t-1: y(t+h) - y(t-1)"],
                                            ["cumsum", "Cumulative sum: y(t) + ... + y(t+h)"]], "level"),
                 sel("scale", "Size of the shock", [["unit", "One unit"], ["sd", "One standard deviation"]], "unit"),
                 sel("cov", "Standard errors", [["hac", "Newey-West HAC"], ["robust", "Robust (HC1)"]], "hac"),
                 num("bandwidth", "HAC bandwidth (blank = horizon + 1)", ""),
                 sel("lag_augment", "Lag augmentation", [["no", "No"], ["yes", "Yes (one extra lag, robust errors)"]], "no")],
         example=dict(file=TS + "returns.csv",
                      spec=dict(y="ip_growth", shock="d_credit_spread", contemp=["sp500_return"], controls=["d_fedfunds"], outcome="cumsum",
                                scale="sd", time="date", lags=6, horizon=24))),
    # ------------------------------------------------------------------ panel data
    dict(id="pols", group="Panel data", name="Pooled OLS", guide="panel-data-guide/pols.html", fn=panel.pols,
         fields=[col("y", "Dependent variable"), cols("x", "Regressors"), UNIT, PTIME, PCOV, BW],
         example=dict(file=PD + "crime4.csv", spec=dict(y="lcrmrte", x=["lprbarr", "lprbconv", "lprbpris", "lavgsen", "lpolpc"], unit="county", time="year"))),
    dict(id="fe", group="Panel data", name="Fixed effects", guide="panel-data-guide/fe.html", fn=panel.fe,
         fields=[col("y", "Dependent variable"), cols("x", "Regressors (time-varying)"), UNIT, PTIME,
                 sel("time_effects", "Time effects", [["no", "No"], ["yes", "Yes (two-way FE)"]], "no"), PCOV, BW],
         example=dict(file=PD + "wagepan.csv", spec=dict(y="lwage", x=["union", "married", "expersq"], unit="nr", time="year", time_effects="yes"))),
    dict(id="re", group="Panel data", name="Random effects", guide="panel-data-guide/re.html", fn=panel.re,
         fields=[col("y", "Dependent variable"), cols("x", "Regressors"), UNIT, PTIME, PCOV, BW],
         example=dict(file=PD + "laborsupply.csv", spec=dict(y="lnhr", x=["lnwg", "kids", "age", "disab"], unit="id", time="year"))),
    dict(id="fd", group="Panel data", name="Panel in first differences", guide="panel-data-guide/diff.html", fn=panel.fd,
         fields=[col("y", "Dependent variable (levels)"), cols("x", "Regressors (levels)"), UNIT, PTIME,
                 sel("cov", "Standard errors", [["dk", "Driscoll-Kraay"], ["clustered", "Clustered by unit"], ["robust", "Robust"], ["unadjusted", "Conventional"]], "dk"),
                 num("bandwidth", "Driscoll-Kraay bandwidth (blank = automatic)", 2)],
         example=dict(file=PD + "gasoline.csv", spec=dict(y="lgaspcar", x=["lincomep", "lrpmg", "lcarpcap"], unit="country", time="year", bandwidth=2))),
    dict(id="fedk", group="Panel data", name="Fixed effects with Driscoll-Kraay errors", guide="panel-data-guide/fedk.html", fn=panel.fedk,
         fields=[col("y", "Dependent variable"), cols("x", "Regressors"), UNIT, PTIME,
                 sel("time_effects", "Time effects", [["no", "No"], ["yes", "Yes (two-way FE)"]], "no"), num("bandwidth", "Bandwidth (blank = automatic)", 3)],
         example=dict(file=PD + "garrett.csv", ops=[dict(op="diff", vars=["trade"], unit="country", time="year")],
                      spec=dict(y="gdp_growth", x=["unem", "d_trade"], unit="country", time="year", bandwidth=3))),
    dict(id="diff_gmm", group="Panel data", name="Difference GMM (Arellano-Bond)", guide="panel-data-guide/gmm.html", fn=panel.diff_gmm,
         fields=[col("y", "Dependent variable"), cols("x", "Strictly exogenous regressors", required=False), UNIT, PTIME,
                 num("lags_y", "Lags of the dependent variable", 1), num("gmm_first", "First instrument lag", 2),
                 num("gmm_last", "Last instrument lag (blank = all)", ""), sel("collapse", "Collapse instruments", [["no", "No"], ["yes", "Yes"]], "no")],
         example=dict(file=PD + "employment.csv",
                      ops=[dict(op="lag", vars=["log_wage", "log_output"], lags=[1, 2], unit="firm", time="year"),
                           dict(op="lag", vars=["log_capital"], lags=[1, 2], unit="firm", time="year")],
                      spec=dict(y="log_emp", x=["log_wage", "log_wage_l1", "log_capital", "log_capital_l1", "log_capital_l2", "log_output",
                                                "log_output_l1", "log_output_l2"], unit="firm", time="year", lags_y=2))),
    dict(id="mg", group="Panel data", name="Mean group and CCE mean group", guide="panel-data-guide/mg.html", fn=panel.mg,
         fields=[col("y", "Dependent variable"), cols("x", "Regressors"), UNIT, PTIME,
                 sel("method", "Estimator", [["mg", "Mean group (Pesaran-Smith)"], ["cce", "CCE mean group (Pesaran 2006)"]], "mg"),
                 sel("trend", "Unit trends (MG)", [["no", "No"], ["yes", "Yes"]], "no")],
         example=dict(file=PD + "growth_saving.csv", ops=[dict(op="growth", vars=["gdp"], unit="country", time="year")],
                      spec=dict(y="g_gdp", x=["sr"], unit="country", time="year", method="mg"))),
    dict(id="pmg", group="Panel data", name="Panel ARDL: PMG, MG and DFE", guide="panel-data-guide/pmg.html", fn=panel.pmg,
         fields=[col("y", "Dependent variable"), cols("x", "Regressors"), UNIT, PTIME,
                 sel("method", "Main estimator", [["pmg", "Pooled mean group"], ["mg", "Mean group"], ["dfe", "Dynamic fixed effects"]], "pmg")],
         example=dict(file=PD + "cigar.csv", ops=[dict(op="demean", vars=["log_sales", "log_price", "log_income", "log_pimin"], time="year")],
                      spec=dict(y="dm_log_sales", x=["dm_log_price", "dm_log_income", "dm_log_pimin"], unit="state", time="year"))),
    dict(id="csardl", group="Panel data", name="Cross-sectionally augmented ARDL (CS-ARDL)", guide="panel-data-guide/csardl.html", fn=panel.csardl,
         fields=[col("y", "Dependent variable"), cols("x", "Regressors"), UNIT, PTIME, num("pt", "Lags of the averages (blank = T^(1/3))", "")],
         example=dict(file=PD + "cigar.csv", spec=dict(y="log_sales", x=["log_price", "log_income", "log_pimin"], unit="state", time="year", pt=1))),
    dict(id="plp", group="Panel data", name="Panel local projections", guide="panel-data-guide/plp.html", fn=lp.plp,
         fields=[col("y", "Response variable"), col("shock", "Shock variable"),
                 cols("contemp", "Contemporaneous controls", required=False), cols("controls", "Other controls (lags only)", required=False), UNIT, PTIME,
                 num("lags", "Lags of every variable", 2), num("horizon", "Horizon", 8),
                 sel("effects", "Fixed effects", [["unit", "Unit effects"], ["twoway", "Unit and time effects"], ["none", "None (pooled)"]], "unit"),
                 sel("outcome", "Outcome", [["level", "Level: y(t+h)"], ["diff", "Change since t-1: y(t+h) - y(t-1)"],
                                            ["cumsum", "Cumulative sum: y(t) + ... + y(t+h)"]], "level"),
                 sel("scale", "Size of the shock", [["unit", "One unit"], ["sd", "One standard deviation"]], "unit"),
                 sel("cov", "Standard errors", [["clustered", "Clustered by unit"], ["dk", "Driscoll-Kraay"]], "clustered"), BW],
         example=dict(file=PD + "cigar.csv", ops=[dict(op="diff", vars=["log_price", "log_income"], unit="state", time="year")],
                      spec=dict(y="log_sales", shock="d_log_price", contemp=["d_log_income"], unit="state", time="year", lags=2, horizon=8,
                                effects="unit", cov="clustered", scale="sd"))),
    # ------------------------------------------------------------------ causal inference
    dict(id="did", group="Causal inference", name="Difference-in-differences", guide="causal-inference-guide/did.html", fn=causal.did,
         fields=[col("outcome", "Outcome"), col("treat", "Treatment indicator (0/1, on when treated)"), col("unit", "Unit identifier"), col("time", "Time variable"),
                 num("boot", "Bootstrap replications", 199)],
         example=dict(file=CI + "castle_2007.csv", spec=dict(outcome="l_homicide", treat="castle_law", unit="state", time="year"))),
    dict(id="staggered", group="Causal inference", name="Staggered DiD (Callaway-Sant'Anna)", guide="causal-inference-guide/staggered.html", fn=causal.staggered,
         fields=[col("outcome", "Outcome"), col("treat", "Treatment indicator (0/1)"), col("unit", "Unit identifier"), col("time", "Time variable"),
                 num("boot", "Bootstrap replications", 199)],
         example=dict(file=CI + "castle.csv", spec=dict(outcome="l_homicide", treat="castle_law", unit="state", time="year"))),
    dict(id="its", group="Causal inference", name="Interrupted time series", guide="causal-inference-guide/its.html", fn=causal.its,
         fields=[col("outcome", "Outcome"), col("treat", "Intervention indicator (0 before, 1 after)"), col("time", "Time variable"),
                 col("unit", "Unit identifier (optional)", required=False)],
         example=dict(file=CI + "seatbelt.csv", spec=dict(outcome="deaths", treat="law", time="time"))),
    dict(id="iv", group="Causal inference", name="Instrumental variables (2SLS)", guide="causal-inference-guide/iv.html", fn=causal.iv,
         fields=[col("outcome", "Outcome"), col("treat", "Treatment (endogenous)"), cols("instruments", "Instrument(s)"), cols("controls", "Controls", required=False)],
         example=dict(file=CI + "card.csv", spec=dict(outcome="lwage", treat="educ", instruments=["nearc4"], controls=["exper", "black", "smsa", "south", "married"]))),
    dict(id="rdd", group="Causal inference", name="Regression discontinuity", guide="causal-inference-guide/rdd.html", fn=causal.rdd,
         fields=[col("outcome", "Outcome"), col("running", "Running variable"), num("cutoff", "Cutoff", 0),
                 col("treat", "Treatment (only for a fuzzy design)", required=False), cols("controls", "Covariates to check for continuity", required=False)],
         example=dict(file=CI + "close_elections.csv", spec=dict(outcome="score", running="demvoteshare", cutoff=0.5, controls=["lagdemvoteshare"]))),
    dict(id="scm", group="Causal inference", name="Synthetic control", guide="causal-inference-guide/scm.html", fn=causal.scm,
         fields=[col("outcome", "Outcome"), col("treat", "Treatment indicator (0/1)"), col("unit", "Unit identifier"), col("time", "Time variable")],
         example=dict(file=CI + "texas.csv", spec=dict(outcome="bmprison", treat="expansion", unit="state", time="year"))),
    dict(id="gsc", group="Causal inference", name="Generalized synthetic control", guide="causal-inference-guide/gsc.html", fn=causal.gsc,
         fields=[col("outcome", "Outcome"), col("treat", "Treatment indicator (0/1)"), col("unit", "Unit identifier"), col("time", "Time variable"),
                 num("boot", "Bootstrap replications", 199)],
         example=dict(file=CI + "castle_murder.csv", spec=dict(outcome="l_murder", treat="castle_law", unit="state", time="year"))),
    # ------------------------------------------------------------------ machine learning
    dict(id="lasso", group="Machine learning", name="LASSO", guide="ml-guide/lasso.html", fn=ml.lasso,
         fields=[col("y", "Outcome to predict"), cols("x", "Predictors"), col("time", "Time column (time-ordered validation)", required=False),
                 num("test_share", "Test share", 0.2)],
         example=dict(file=MLD + "econmath.csv", spec=dict(y="score", x=["age", "work", "study", "econhs", "colgpa", "hsgpa", "acteng", "actmth", "male",
                                                                        "calculus", "attexc", "attgood", "fathcoll", "mothcoll"]))),
    dict(id="ridge", group="Machine learning", name="Ridge", guide="ml-guide/ridge.html", fn=ml.ridge,
         fields=[col("y", "Outcome to predict"), cols("x", "Predictors"), col("time", "Time column (time-ordered validation)", required=False),
                 num("test_share", "Test share", 0.2)],
         example=dict(file=MLD + "wage2.csv", spec=dict(y="lwage", x=["hours", "IQ", "KWW", "educ", "exper", "tenure", "age", "married", "black", "south",
                                                                     "urban", "sibs", "brthord", "meduc", "feduc"]))),
    dict(id="enet", group="Machine learning", name="Elastic net", guide="ml-guide/enet.html", fn=ml.enet,
         fields=[col("y", "Outcome to predict"), cols("x", "Predictors"), col("time", "Time column (time-ordered validation)", required=False),
                 num("test_share", "Test share", 0.2)],
         example=dict(file=MLD + "macro.csv", ops=[dict(op="diff", vars=["tbilrate_l1", "tbilrate_l2", "tbilrate_l3", "tbilrate_l4"], time="quarter")],
                      spec=dict(y="infl", time="quarter", x=[f"{v}_l{k}" for v in ("infl", "unemp", "gdp_growth", "m1_growth") for k in LAG4]
                                + [f"d_tbilrate_l{k}" for k in LAG4]))),
    dict(id="rf", group="Machine learning", name="Random forest", guide="ml-guide/rf.html", fn=ml.rf,
         fields=[col("y", "Outcome to predict"), cols("x", "Predictors"), col("time", "Time column (time-ordered validation)", required=False),
                 num("test_share", "Test share", 0.2), num("n_estimators", "Number of trees", 300), num("min_leaf", "Minimum leaf size", 5)],
         example=dict(file=MLD + "mlb1.csv", spec=dict(y="lsalary", x="__all__"))),
    dict(id="xgb", group="Machine learning", name="XGBoost", guide="ml-guide/xgb.html", fn=ml.xgb,
         fields=[col("y", "Outcome to predict"), cols("x", "Predictors"), col("time", "Time column (time-ordered validation)", required=False),
                 num("test_share", "Test share", 0.2), num("n_estimators", "Number of trees", 400), num("learning_rate", "Learning rate", 0.05),
                 num("max_depth", "Maximum depth", 4)],
         example=dict(file=MLD + "injury.csv", spec=dict(y="ldurat", x="__all__"))),
    dict(id="dml_plr", group="Machine learning", name="Double ML: partially linear model", guide="ml-guide/dml-plr.html", fn=ml.dml_plr,
         fields=[col("y", "Outcome"), col("treat", "Treatment (continuous or binary)"), cols("x", "Controls"),
                 sel("learner", "Learners", [["both", "LASSO and random forest"], ["lasso", "LASSO"], ["rf", "Random forest"]], "both"),
                 num("folds", "Folds", 5), num("reps", "Repetitions", 2), num("trees", "Trees in the random forests", 100)],
         example=dict(file=MLD + "growth.csv", spec=dict(y="growth", treat="gdpsh465", x="__all__", reps=1))),
    dict(id="dml_irm", group="Machine learning", name="Double ML: interactive model (binary treatment)", guide="ml-guide/dml-irm.html", fn=ml.dml_irm,
         fields=[col("y", "Outcome"), col("treat", "Binary treatment (0/1)"), cols("x", "Controls"),
                 sel("learner", "Learners", [["both", "LASSO and random forest"], ["lasso", "LASSO"], ["rf", "Random forest"]], "both"),
                 num("folds", "Folds", 5), num("reps", "Repetitions", 1), num("trees", "Trees in the random forests", 100)],
         example=dict(file=MLD + "k401_irm.csv", spec=dict(y="net_tfa", treat="e401", x="__all__", learner="rf", reps=1))),
    dict(id="dml_iv", group="Machine learning", name="Double ML: partially linear IV", guide="ml-guide/dml-iv.html", fn=ml.dml_iv,
         fields=[col("y", "Outcome"), col("treat", "Treatment (endogenous)"), col("instrument", "Instrument"), cols("x", "Controls"),
                 sel("learner", "Learners", [["both", "LASSO and random forest"], ["lasso", "LASSO"], ["rf", "Random forest"]], "both"),
                 num("folds", "Folds", 5), num("reps", "Repetitions", 1), num("trees", "Trees in the random forests", 100)],
         example=dict(file=MLD + "k401_iv.csv", spec=dict(y="net_tfa", treat="p401", instrument="e401", x="__all__", learner="rf", reps=1))),
    dict(id="causal_forest", group="Machine learning", name="Causal forest (heterogeneous effects)", guide="ml-guide/causal-forest.html", fn=ml.causal_forest,
         fields=[col("y", "Outcome"), col("treat", "Binary treatment (0/1)"), cols("x", "Controls"), num("folds", "Folds", 5),
                 num("n_estimators", "Number of trees", 300), num("min_leaf", "Minimum leaf size", 20),
                 num("trees", "Trees in the nuisance forests", 100)],
         example=dict(file=MLD + "k401_irm.csv", spec=dict(y="net_tfa", treat="e401", x="__all__"))),
]

# ------------------------------------------------------------------ Bayesian
OX = "oes/examples/"
BV_PRIOR = sel("prior", "Prior", [["minnesota", "Minnesota (Litterman)"], ["nw", "Normal-Wishart"], ["inw", "Independent Normal-Wishart"],
                                  ["nd", "Normal-diffuse"]], "minnesota")
HYP = [num("lambda1", "Overall tightness lambda1", 0.1), num("lambda2", "Cross-variable tightness lambda2", 0.5),
       num("lambda3", "Lag decay lambda3", 1), num("lambda4", "Constant tightness lambda4", 100)]
DRAWS = [num("draws", "Posterior draws", 1000), num("burn", "Burn-in (Gibbs samplers)", 500), num("seed", "Random seed", 0)]
MODELS += [
    dict(id="bayes_reg", group="Bayesian", name="Bayesian linear regression", guide="bayesian-guide/bayes-reg.html", fn=bayes.bayes_reg,
         fields=[col("y", "Dependent variable"), cols("x", "Explanatory variables"), TIME,
                 sel("method", "Method", [["conjugate", "Exact (Normal-inverse-gamma)"], ["gibbs", "Gibbs sampler (independent prior)"]], "conjugate"),
                 sel("prior", "Prior on the slopes", [["gprior", "Zellner g-prior"], ["normal", "Normal, common s.d."]], "gprior"),
                 num("g", "g (blank = number of observations)", ""), num("prior_sd", "Prior s.d. (normal prior)", 10),
                 num("prior_mean", "Prior means of the slopes (blank = 0; one or a comma list)", ""),
                 num("draws", "Draws (Gibbs)", 5000), num("burn", "Burn-in (Gibbs)", 1000), num("seed", "Random seed", 0)],
         example=dict(file=MLD + "wage2.csv", spec=dict(y="lwage", x=["educ", "exper", "tenure", "IQ", "married", "black", "south", "urban"]))),
    dict(id="bvar", group="Bayesian", name="Bayesian VAR (Minnesota, Normal-Wishart, ...)", guide="bayesian-guide/bvar.html", fn=bayes.bvar,
         fields=[cols("vars", "Variables (Cholesky order)"), TIME, num("lags", "Lags", 4), BV_PRIOR] + HYP +
                [num("delta", "Prior mean of own first lag (1 = random walk; one or a comma list)", 1),
                 cols("block", "Block-exogenous variables (not affected by the others)", required=False),
                 num("horizon", "IRF horizon", 20), num("fhorizon", "Forecast horizon", 8)] + DRAWS,
         example=dict(file=TS + "money.csv", spec=dict(vars=["log_price", "log_m2", "log_gdp"], time="date", lags=4, prior="minnesota"))),
    dict(id="favar", group="Bayesian", name="Factor-augmented VAR (FAVAR)", guide="bayesian-guide/favar.html", fn=bayes.favar,
         fields=[cols("y", "Observed VAR variables (policy variable last)"), cols("x", "Informational series (for the factors)"), TIME,
                 num("factors", "Number of factors", 2), num("lags", "Lags", 2),
                 sel("prior", "Prior", [["nw", "Normal-Wishart"], ["minnesota", "Minnesota"], ["inw", "Independent Normal-Wishart"], ["nd", "Normal-diffuse"]], "nw"),
                 num("lambda1", "Overall tightness lambda1", 0.1),
                 num("delta", "Prior mean of own first lag of the observed variables (1 = random walk; one or a comma list)", 0),
                 num("horizon", "IRF horizon", 20)] + DRAWS,
         example=dict(file=OX + "us_macro.csv", spec=dict(y=["inflation", "unemp", "tbilrate"], time="date", factors=2,
                                                          x=["gdp_growth", "cons_growth", "inv_growth", "gov_growth", "income_growth", "m1_growth"]))),
    dict(id="mabvar", group="Bayesian", name="Mean-adjusted (steady-state) BVAR", guide="bayesian-guide/steady-state.html", fn=bayes.mabvar,
         fields=[cols("vars", "Variables (Cholesky order)"), TIME, num("lags", "Lags", 2),
                 num("ss_mean", "Prior means of the steady states (comma list; blank = sample means)", ""),
                 num("ss_sd", "Prior s.d. of the steady states (comma list; blank = sample s.d.)", ""),
                 num("lambda1", "Overall tightness lambda1", 0.1), num("lambda2", "Cross-variable tightness lambda2", 0.5),
                 num("horizon", "IRF horizon", 20), num("fhorizon", "Forecast horizon", 20)] + DRAWS,
         example=dict(file=OX + "us_macro.csv", spec=dict(vars=["gdp_growth", "inflation", "tbilrate"], time="date", lags=2,
                                                          ss_mean="3, 3.5, 5", ss_sd="0.5, 1, 1"))),
    dict(id="pvar", group="Bayesian", name="Panel VAR (6 estimators)", guide="bayesian-guide/panel-var.html", fn=pvar.pvar,
         fields=[cols("vars", "Variables (Cholesky order)"), UNIT, PTIME, num("lags", "Lags", 1),
                 sel("estimator", "Estimator", [["mg", "OLS mean group"], ["pooled", "Bayesian pooled"], ["zh", "Zellner-Hong random effects"],
                                                ["hierarchical", "Hierarchical (Jarocinski)"], ["static", "Static factor (Canova-Ciccarelli)"],
                                                ["dynamic", "Dynamic factor (Canova-Ciccarelli)"]], "mg"),
                 num("lambda1", "lambda1 (pooled, Zellner-Hong)", 0.1), num("prior_var", "Prior variance of the factors", 1),
                 num("horizon", "IRF horizon", 12)] + DRAWS,
         example=dict(file=PD + "garrett.csv", spec=dict(vars=["gdp_growth", "unem"], unit="country", time="year", lags=1, estimator="mg"))),
    dict(id="tvpvar", group="Bayesian", name="Time-varying parameter VAR (TVP-VAR)", guide="bayesian-guide/tvp-var.html", fn=statespace_var.tvpvar,
         fields=[cols("vars", "Variables (Cholesky order)"), TIME, num("lags", "Lags", 2), num("training", "Training-sample length", 40),
                 num("kQ", "kQ (amount of time variation)", 0.01), num("horizon", "IRF horizon", 20),
                 num("draws", "Posterior draws", 500), num("burn", "Burn-in", 300), num("seed", "Random seed", 0)],
         example=dict(file=OX + "us_macro.csv", spec=dict(vars=["inflation", "unemp", "tbilrate"], time="date", lags=2))),
    dict(id="svvar", group="Bayesian", name="Stochastic-volatility VAR (SV-VAR)", guide="bayesian-guide/sv-var.html", fn=statespace_var.svvar,
         fields=[cols("vars", "Variables (Cholesky order)"), TIME, num("lags", "Lags", 2), num("lambda1", "Overall tightness lambda1", 0.1),
                 num("lambda2", "Cross-variable tightness lambda2", 0.5), num("delta", "Prior mean of own first lag", 0),
                 num("horizon", "IRF horizon", 20)] + DRAWS,
         example=dict(file=OX + "us_macro.csv", spec=dict(vars=["inflation", "unemp", "tbilrate"], time="date", lags=2))),
    dict(id="mfbvar", group="Bayesian", name="Mixed-frequency BVAR (monthly + quarterly)", guide="bayesian-guide/mixed-frequency.html", fn=statespace_var.mfbvar,
         fields=[cols("monthly", "Monthly variables"), cols("quarterly", "Quarterly variables (value in the quarter's last month, other months blank)"),
                 col("time", "Month column", required=False), num("lags", "Lags (months)", 5),
                 sel("aggregation", "Quarterly aggregation", [["triangle", "Growth rates: triangle weights (1,2,3,2,1)/9"], ["average", "Levels: 3-month average"]], "triangle"),
                 num("lambda1", "Overall tightness lambda1", 0.2), num("delta", "Prior mean of own first lag", 0), num("horizon", "IRF horizon", 24),
                 num("draws", "Posterior draws", 300), num("burn", "Burn-in", 200), num("seed", "Random seed", 0)],
         example=dict(file=OX + "mixed_frequency.csv", spec=dict(monthly=["ip_growth", "fedfunds"], quarterly=["gdp_growth"], time="date", lags=5))),
    dict(id="tvar", group="Bayesian", name="Threshold VAR (Bayesian)", guide="bayesian-guide/threshold-var.html", fn=bayes.tvar,
         fields=[cols("vars", "Variables (Cholesky order)"), col("threshold_var", "Threshold variable"), TIME, num("lags", "Lags", 2),
                 num("max_delay", "Maximum delay d", 2), num("trim", "Minimum share in each regime", 0.15), num("grid", "Threshold grid points", 40),
                 num("lambda1", "Overall tightness lambda1", 0.2), num("horizon", "IRF horizon", 20), num("draws", "Posterior draws", 1000), num("seed", "Random seed", 0)],
         example=dict(file=OX + "us_macro.csv", spec=dict(vars=["gdp_growth", "inflation", "tbilrate"], threshold_var="inflation", time="date", lags=2))),
    # ------------------------------------------------------------------ Markov switching
    dict(id="ms_reg", group="Markov switching", name="Markov-switching regression", guide="markov-guide/ms-regression.html", fn=markov.ms_reg,
         fields=[col("y", "Dependent variable"), cols("x", "Regressors (optional)", required=False), TIME, num("regimes", "Number of regimes", 2),
                 sel("switching_variance", "Switching variance", [["yes", "Yes"], ["no", "No"]], "yes"),
                 sel("switching_exog", "Switching slopes", [["yes", "Yes"], ["no", "No"]], "yes"), col("reference", "Reference 0/1 series, e.g. recession dates (optional)", required=False, help="Compared with the estimated regimes")],
         example=dict(file=OX + "fedfunds.csv", spec=dict(y="fedfunds", time="date", switching_variance="no"))),
    dict(id="ms_ar", group="Markov switching", name="Markov-switching autoregression (Hamilton)", guide="markov-guide/ms-ar.html", fn=markov.ms_ar,
         fields=[col("y", "Variable"), TIME, num("regimes", "Number of regimes", 2), num("order", "AR order", 4),
                 sel("switching_ar", "Switching AR coefficients", [["no", "No"], ["yes", "Yes"]], "no"),
                 sel("switching_variance", "Switching variance", [["no", "No"], ["yes", "Yes"]], "no"), col("reference", "Reference 0/1 series, e.g. recession dates (optional)", required=False, help="Compared with the estimated regimes")],
         example=dict(file=OX + "hamilton_gnp.csv", spec=dict(y="gnp_growth", time="date", order=4, reference="nber_recession"))),
    dict(id="msvar", group="Markov switching", name="Markov-switching VAR", guide="markov-guide/ms-var.html", fn=markov.msvar,
         fields=[cols("vars", "Variables (Cholesky order)"), TIME, num("regimes", "Number of regimes", 2), num("lags", "Lags", 1),
                 sel("switching", "What switches", [["MSIH", "MSIH: intercepts and covariances"], ["MSI", "MSI: intercepts only"],
                                                     ["MSIAH", "MSIAH: intercepts, lags and covariances"]], "MSIH"),
                 num("starts", "Starting values", 6), num("horizon", "IRF horizon", 20), col("reference", "Reference 0/1 series, e.g. recession dates (optional)", required=False, help="Compared with the estimated regimes")],
         example=dict(file=OX + "us_macro.csv", spec=dict(vars=["gdp_growth", "inflation"], time="date", lags=1, switching="MSIH"))),
    dict(id="mspvar", group="Markov switching", name="Markov-switching panel VAR", guide="markov-guide/ms-panel-var.html", fn=markov.mspvar,
         fields=[cols("vars", "Variables (Cholesky order)"), UNIT, PTIME, num("regimes", "Number of regimes", 2), num("lags", "Lags", 1),
                 sel("switching", "What switches", [["MSIH", "MSIH: intercepts and covariances"], ["MSI", "MSI: intercepts only"],
                                                     ["MSIAH", "MSIAH: intercepts, lags and covariances"]], "MSIH"),
                 sel("demean", "Remove unit means", [["yes", "Yes"], ["no", "No"]], "yes"), num("starts", "Starting values", 6), num("horizon", "IRF horizon", 20)],
         example=dict(file=PD + "garrett.csv", spec=dict(vars=["gdp_growth", "unem"], unit="country", time="year", lags=1, switching="MSIH"))),
]

BY_ID = {m["id"]: m for m in MODELS}
GROUPS = ["Time series", "Panel data", "Causal inference", "Machine learning", "Bayesian", "Markov switching"]


def packages(model_id):
    """Pyodide packages a model needs beyond NumPy, pandas, SciPy and statsmodels."""
    m = BY_ID[model_id]
    if m["group"] != "Machine learning":
        return []
    return ["scikit-learn", "xgboost"] if model_id == "xgb" else ["scikit-learn"]


SLOW = {"dml_irm", "dml_iv", "causal_forest", "staggered", "tvpvar", "mfbvar", "svvar"}


def catalog():
    return [dict({k: v for k, v in m.items() if k != "fn"}, packages=packages(m["id"]), slow=m["id"] in SLOW) for m in MODELS]


def catalog_json():
    return json.dumps(catalog())


# data operations ------------------------------------------------------------------------------------------------
def _sorted(df, op):
    unit, time = op.get("unit"), op.get("time")
    for c in (unit, time):
        if c and c not in df.columns:
            raise OESError(f"Column {c} not found for the data operation.")
    if time:
        if unit:
            df = df.assign(_ord=np.arange(len(df)))
            df = time_order(df, time)
            df = df.sort_values(unit, kind="stable")
        else:
            df = time_order(df, time)
    return df.reset_index(drop=True), (df.groupby(unit, sort=False) if unit else None)


def prepare(df, ops):
    """Apply data operations in order. Each op is a dict with op in {lag, diff, log, growth, sample} and vars."""
    for op in ops or []:
        kind = op.get("op")
        vars_ = [v for v in (op.get("vars") or []) if v]
        miss = [v for v in vars_ if v not in df.columns]
        if miss:
            raise OESError("Column(s) not found for the data operation: " + ", ".join(miss))
        if kind == "sample":
            t = op.get("time")
            if not t or t not in df.columns:
                raise OESError("Choose the time column for the sample restriction.")
            df = time_order(df, t).reset_index(drop=True)
            lab = df[t].astype(str)
            s, e = op.get("start"), op.get("end")
            keep = np.ones(len(df), bool)
            num = pd.to_numeric(df[t], errors="coerce")
            if s not in (None, ""):
                keep &= (num >= float(s)).to_numpy() if num.notna().all() and _isnum(s) else (lab >= str(s)).to_numpy()
            if e not in (None, ""):
                keep &= (num <= float(e)).to_numpy() if num.notna().all() and _isnum(e) else (lab <= str(e)).to_numpy()
            df = df[keep].reset_index(drop=True)
            continue
        df, g = _sorted(df, op)
        for v in vars_:
            x = pd.to_numeric(df[v], errors="coerce")
            sh = (lambda k: x.groupby(df[op["unit"]], sort=False).shift(k)) if op.get("unit") else (lambda k: x.shift(k))
            if kind == "lag":
                for k in op.get("lags") or [1]:
                    df[f"{v}_l{int(k)}"] = sh(int(k))
            elif kind == "diff":
                df[f"d_{v}"] = x - sh(1)
            elif kind == "log":
                df[f"ln_{v}"] = np.log(x.where(x > 0))
            elif kind == "demean":
                df[f"dm_{v}"] = x - x.groupby(df[op["time"]]).transform("mean")
            elif kind == "growth":
                lx = np.log(x.where(x > 0))
                lsh = lx.groupby(df[op["unit"]], sort=False).shift(1) if op.get("unit") else lx.shift(1)
                df[f"g_{v}"] = 100 * (lx - lsh)
            else:
                raise OESError(f"Unknown data operation {kind}.")
    return df


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# entry point ----------------------------------------------------------------------------------------------------
def run(model_id, csv_text, spec=None):
    warnings.simplefilter("ignore")
    if isinstance(spec, str):
        spec = json.loads(spec) if spec.strip() else {}
    spec = dict(spec or {})
    m = BY_ID.get(model_id)
    if m is None:
        return {"error": f"Unknown model {model_id}."}
    try:
        df = read_csv(csv_text) if isinstance(csv_text, str) else csv_text.copy()
        df = prepare(df, spec.pop("ops", None))
        if spec.get("x") == "__all__":
            used = {spec.get(k) for k in ("y", "treat", "instrument", "time", "unit")}
            spec["x"] = [c for c in df.columns if c not in used and pd.api.types.is_numeric_dtype(df[c])]
        for k in ("case",):
            if k in spec and spec[k] not in (None, ""):
                spec[k] = int(spec[k])
        out = m["fn"](df, spec)
        out["guide"] = m["guide"]
        out["engine"] = __version__
        return _clean(out)
    except OESError as e:
        return {"error": str(e)}
    except np.linalg.LinAlgError as e:
        return {"error": f"The model could not be estimated: a matrix is singular ({e}). Check for collinear or constant variables."}
    except Exception as e:  # report rather than crash the page
        return {"error": f"The estimation failed: {type(e).__name__}: {e}"}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def run_json(model_id, csv_text, spec_json="{}"):
    return json.dumps(run(model_id, csv_text, spec_json))
