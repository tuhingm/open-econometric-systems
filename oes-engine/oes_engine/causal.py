"""Causal-inference designs of the Causal Inference Research Guide.

The estimators are those of the guide's own file (_causal_core.py, identical to causal-inference-guide/examples/
causal_tests.py); here each design can be chosen directly instead of being selected from the data.
"""
import numpy as np

from . import _causal_core as C
from .core import (OESError, as_list, chart, coef_table, distinct, f, int_option, need, result, stat, table, test_row,
                   tests_table)


def _roles(spec, *keys):
    r = {k: spec.get(k) for k in ("outcome", "treat", "unit", "time", "running", "cutoff")}
    r["instruments"] = as_list(spec.get("instruments"))
    r["controls"] = as_list(spec.get("controls"))
    for k in keys:
        need(spec, k, {"outcome": "outcome", "treat": "treatment variable", "unit": "unit identifier", "time": "time variable",
                       "running": "running variable", "instruments": "instrument(s)"}[k])
    distinct(*[[r[k]] for k in ("outcome", "treat", "unit", "time", "running") if r.get(k)], r["instruments"], r["controls"])
    return r


def _check(df, roles):
    cols = [roles.get(k) for k in ("outcome", "treat", "unit", "time", "running") if roles.get(k)] + roles["instruments"] + roles["controls"]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise OESError("Column(s) not found in the data: " + ", ".join(miss))


def _boot(spec):
    C.BOOT = int_option(spec, "boot", 199, 49, 999, "Bootstrap replications")


# ------------------------------------------------------------------------------------------------ RDD
def rdd(df, spec):
    roles = _roles(spec, "outcome", "running")
    roles["cutoff"] = float(spec.get("cutoff") or 0)
    _check(df, roles)
    r = C._rdd(df, roles)
    if "error" in r:
        raise OESError(r["error"])
    rd = r["rd"]
    kind = "Fuzzy" if r["fuzzy"] else "Sharp"
    tabs = [coef_table([f"RD effect at {r['running']} = {r['cutoff']:g}"], [rd["est"]], [rd["se"]], dist="z",
                       title=f"{kind} RD: local linear estimate (triangular kernel, IK bandwidth h = {r['h']})",
                       note=f"{rd['n']} observations within the bandwidth." + (" Fuzzy RD: the jump in the outcome divided by the jump in treatment." if r["fuzzy"] else ""))]
    tabs.append(table("Bandwidth sensitivity", ["Bandwidth multiple", "h", "Estimate", "Std. err.", "Obs."],
                      [[s["mult"], s["h"], s["est"], s["se"], s["n"]] for s in r["sensitivity"]]))
    rows = []
    den = r.get("density")
    if den:
        rows.append(test_row("McCrary density test (log difference)", den["z"], den["p"], f"theta = {den['theta']}; H0: no manipulation of the running variable"))
    for pl in r.get("placebo", []):
        rows.append(test_row(f"Placebo cutoff at {pl['cutoff']} ({pl['side']})", pl["est"] / pl["se"] if pl["se"] else None, pl["p"], f"estimate {pl['est']}; H0: no jump"))
    for cv in r.get("covariates", []):
        rows.append(test_row(f"Covariate continuity: {cv['var']}", cv["est"] / cv["se"] if cv["se"] else None, cv["p"], f"jump {cv['est']}; H0: no jump"))
    tabs.append(tests_table(rows, "Validity checks"))
    charts = []
    if r.get("bins"):
        bx = [b[0] for b in r["bins"]]
        charts.append(chart(f"Binned means of {r['outcome']} around the cutoff", bx, [("binned mean", [b[1] for b in r["bins"]])],
                            r["running"], r["outcome"], kind="scatter"))
    summary = [stat("Observations", r["n"]), stat("Left of cutoff", r["n_left"]), stat("Right of cutoff", r["n_right"]),
               stat("Bandwidth", r["h"]), stat("RD estimate", rd["est"]), stat("p-value", rd["p"])]
    notes = [f"Flags raised by the checks: {', '.join(r['flags'])}." if r.get("flags") else "No validity check is rejected at 5%."]
    return result("rdd", f"Regression discontinuity: {r['outcome']}", r["n"], summary, tabs, charts, notes)


# ------------------------------------------------------------------------------------------------ IV
def iv(df, spec):
    roles = _roles(spec, "outcome", "treat", "instruments")
    _check(df, roles)
    r = C._iv(df, roles)
    if "error" in r:
        raise OESError(r["error"])
    tabs = [coef_table([f"{r['treat']} (2SLS)", f"{r['treat']} (OLS)"], [r["tsls"]["est"], r["ols"]["est"]], [r["tsls"]["se"], r["ols"]["se"]],
                       dist="z", title="Effect of the treatment", note="Heteroskedasticity-robust (HC1) standard errors."),
            coef_table([c["var"] for c in r["first_stage"]["coefs"]], [c["est"] for c in r["first_stage"]["coefs"]],
                       [c["se"] for c in r["first_stage"]["coefs"]], dist="z", title=f"First stage: {r['treat']} on the instruments"),
            coef_table([c["var"] for c in r["reduced_form"]], [c["est"] for c in r["reduced_form"]], [c["se"] for c in r["reduced_form"]],
                       dist="z", title=f"Reduced form: {r['outcome']} on the instruments")]
    rows = [test_row("First-stage F (robust)", r["first_stage"]["F"], r["first_stage"]["p"], f"partial R2 = {r['first_stage']['partial_r2']}; F < 10: weak"),
            test_row("Anderson-Rubin (weak-IV robust)", r["ar"]["F"], r["ar"]["p"], f"95% AR confidence set {r['ar']['ci']}" if r["ar"].get("bounded") else "unbounded confidence set"),
            test_row("Endogeneity (control function t)", r["endog"]["t"], r["endog"]["p"], "H0: treatment exogenous")]
    if r.get("overid"):
        rows.append(test_row(f"Hansen J ({r['overid']['df']} df)", r["overid"]["J"], r["overid"]["p"], "H0: instruments valid"))
    tabs.append(tests_table(rows))
    summary = [stat("Observations", r["n"]), stat("2SLS estimate", r["tsls"]["est"]), stat("OLS estimate", r["ols"]["est"]),
               stat("First-stage F", r["first_stage"]["F"]), stat("Weak instrument", "yes" if r["weak"] else "no")]
    return result("iv", f"Instrumental variables: effect of {r['treat']} on {r['outcome']}", r["n"], summary, tabs)


# ------------------------------------------------------------------------------------------------ designs over time
def _panel(df, spec, need_unit=True):
    keys = ("outcome", "treat", "time") + (("unit",) if need_unit else ())
    roles = _roles(spec, *keys)
    _check(df, roles)
    P, err = C._prep_panel(df, roles)
    if err:
        raise OESError(err)
    G, switch_off = C._adoption(P["d"])
    treated = [u for u in G.index if np.isfinite(G[u])]
    if not treated:
        raise OESError("No observation has the treatment switched on. The treatment column should be 1 for treated unit-periods and 0 otherwise.")
    return P, G, switch_off, treated


def _event_chart(ev, title):
    x = [e["e"] for e in ev]
    y = [e["est"] for e in ev]
    se = [e["se"] or 0 for e in ev]
    return chart(title, x, [("effect", y)], "periods relative to adoption", "estimate",
                 band=([a - 1.96 * s for a, s in zip(y, se)], [a + 1.96 * s for a, s in zip(y, se)]))


def _did_common(df, spec, staggered):
    _boot(spec)
    P, G, switch_off, treated = _panel(df, spec)
    cohorts = sorted(set(int(G[u]) for u in treated))
    never = len(G) - len(treated)
    if never == 0:
        raise OESError("No never-treated comparison units were found.")
    r = C._did(P, G, staggered)
    if "error" in r:
        raise OESError(r["error"])
    att, tw = r["att"], r["twfe"]
    label = "Callaway-Sant'Anna ATT (never-treated controls)" if staggered else "Difference-in-differences ATT"
    tabs = [coef_table([label, "Two-way fixed effects"], [att["est"], tw["est"]], [att["se"], tw["se"]], dist="z", p=[att["p"], tw["p"]],
                       title="Average effect on the treated",
                       note="ATT: cluster-bootstrap standard errors over units. TWFE: clustered by unit. With staggered adoption TWFE can be biased."),
            table("Event-study estimates (base period -1)", ["Event time", "Estimate", "Std. err."], [[e["e"], e["est"], e["se"]] for e in r["event"]])]
    rows = []
    if r.get("pretrend"):
        pt = r["pretrend"]
        rows.append(test_row(f"Joint pre-trend test, leads -5..-2 ({pt['df']} df)", pt["stat"], pt["p"], "H0: parallel pre-trends"))
    tabs.append(tests_table(rows or [["Pre-trend test", None, None, "not enough leads"]], "Identification checks"))
    notes = []
    if not staggered and len(cohorts) > 1:
        notes.append(f"Treatment starts at {len(cohorts)} different dates: the staggered DiD model is the appropriate one.")
    if switch_off:
        notes.append("Some units switch the treatment off again; the estimators assume adoption is permanent.")
    if r.get("pretrend") and r["pretrend"]["p"] is not None and r["pretrend"]["p"] < 0.05:
        notes.append("The pre-trend test rejects: consider the generalized synthetic control model.")
    summary = [stat("Units", len(G)), stat("Treated units", len(treated)), stat("Never treated", never), stat("Adoption dates", len(cohorts)),
               stat("ATT", att["est"]), stat("p-value", att["p"])]
    mid = "staggered" if staggered else "did"
    return result(mid, ("Staggered DiD" if staggered else "Difference-in-differences") + f": {P['y']}", len(P["d"]), summary, tabs,
                  [_event_chart(r["event"], "Event study")], notes)


def did(df, spec):
    return _did_common(df, spec, False)


def staggered(df, spec):
    return _did_common(df, spec, True)


def its(df, spec):
    P, G, switch_off, treated = _panel(df, spec, need_unit=False)
    r = C._its(P, G)
    if "error" in r:
        raise OESError(r["error"])
    co = r["coefs"]
    tabs = [coef_table([c["var"] for c in co], [c["est"] for c in co], [c["se"] for c in co], dist="z", p=[c["p"] for c in co],
                       title="Segmented regression", note=f"Newey-West standard errors with {r['nw_lags']} lags." + (" Seasonal dummies included." if r["seasonal"] else "")),
            tests_table([["Durbin-Watson", r["dw"], None, "about 2 means no autocorrelation"]] +
                        [test_row(f"Placebo break at {p['at']}", p["est"] / p["se"] if p["se"] else None, p["p"], f"level change {p['est']}; H0: none") for p in r["placebo"]],
                        "Checks")]
    summary = [stat("Periods", r["T"]), stat("Intervention", r["g_label"]), stat("Pre-period mean", r["mean_pre"]), stat("Average effect", r["avg_effect"])]
    charts = [chart("Observed series and counterfactual", r["labels"], [("observed", r["series"]), ("counterfactual", r["counterfactual"])])]
    notes = ["With several units, the treated units' average is analysed as one series."] if P["d"]["_u"].nunique() > 1 else []
    return result("its", f"Interrupted time series: {P['y']}", r["T"], summary, tabs, charts, notes)


def scm(df, spec):
    P, G, switch_off, treated = _panel(df, spec)
    r = C._scm(P, G)
    if r is None:
        raise OESError("Synthetic control needs a balanced panel for the treated and comparison units and at least 3 pre-treatment periods.")
    tabs, charts = [], []
    for u in r["units"]:
        tabs.append(table(f"Synthetic {u['unit']}: donor weights", ["Donor", "Weight"], [[w["unit"], w["w"]] for w in u["weights"]]))
        tabs.append(tests_table([["Pre-treatment RMSPE", u["pre_rmspe"], None, "fit before adoption"],
                                 ["Post/pre RMSPE ratio", u["ratio"], u["perm_p"], f"permutation p-value from {u['n_placebo']} placebo units"],
                                 ["Average post-treatment gap", u["att"], None, f"pre-treatment mean {u['mean_pre']}"]], f"Synthetic {u['unit']}: inference"))
        charts.append(chart(f"{u['unit']} and its synthetic control", P["labels"], [(u["unit"], u["actual"]), ("synthetic", u["synth"])]))
        charts.append(chart(f"Gap: {u['unit']} minus synthetic", P["labels"], [("gap", u["gap"])]))
    u0 = r["units"][0]
    summary = [stat("Treated units", len(r["units"])), stat("Donors", r["donors"]), stat("Adoption", u0["g_label"]),
               stat("Average gap", u0["att"]), stat("Permutation p", u0["perm_p"])]
    return result("scm", f"Synthetic control: {P['y']}", len(P["d"]), summary, tabs, charts)


def gsc(df, spec):
    _boot(spec)
    P, G, switch_off, treated = _panel(df, spec)
    if len(G) - len(treated) < 5:
        raise OESError("Generalized synthetic control needs at least 5 never-treated units.")
    r = C._gsc(P, G)
    if r is None:
        raise OESError("Could not estimate the factor model: check that the panel has enough pre-treatment periods.")
    att = r["att"]
    tabs = [coef_table(["ATT (interactive fixed effects)"], [att["est"]], [att["se"]], dist="z", p=[att["p"]], title="Average effect on the treated",
                       note=f"{r['r']} latent factor(s) chosen by cross-validation; bootstrap standard errors."),
            table("Cross-validation of the number of factors", ["Factors", "MSPE"], [[k, v] for k, v in r["cv"].items()]),
            table("Event-time effects", ["Event time", "Estimate", "Std. err."], [[e["e"], e["est"], e["se"]] for e in r["event"]])]
    summary = [stat("Units", len(G)), stat("Treated", len(treated)), stat("Factors", r["r"]), stat("ATT", att["est"]),
               stat("p-value", att["p"]), stat("Pre-treatment RMSPE", r["pre_rmspe"])]
    return result("gsc", f"Generalized synthetic control: {P['y']}", len(P["d"]), summary, tabs, [_event_chart(r["event"], "Event-time effects")])
