"""Machine-learning models of the Machine Learning Research Guide.

Prediction: LASSO, ridge, elastic net, random forest and XGBoost, each with cross-validated tuning, a hold-out test
set and an OLS benchmark. Causal: double machine learning (partially linear, interactive and IV models) using the
guide's cross-fitting routine (_ml_core.crossfit), and a causal forest built as a doubly robust forest (DR-learner).
"""
import numpy as np
import pandas as pd

from . import _ml_core as M
from .core import OESError, as_list, chart, coef_table, distinct, f, float_option, int_option, need, numeric, result, stat, table, tests_table, test_row, time_order


def _data(df, spec, cols):
    tcol = spec.get("time")
    if tcol:
        df = time_order(df, tcol)
    d = numeric(df, cols).reset_index(drop=True)
    if len(d) < 30:
        raise OESError(f"Machine-learning models need at least 30 complete observations; {len(d)} were found.")
    return d, bool(tcol)


class XGBoostRegressor:
    """Minimal scikit-learn-compatible wrapper around xgboost.train (XGBRegressor is not compatible with the
    scikit-learn 1.6 shipped with Pyodide 0.27)."""

    def __init__(self, n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=1.0, reg_lambda=1.0, random_state=0):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.subsample, self.colsample_bytree, self.min_child_weight = subsample, colsample_bytree, min_child_weight
        self.reg_lambda, self.random_state = reg_lambda, random_state

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in ("n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree",
                                              "min_child_weight", "reg_lambda", "random_state")}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        import xgboost
        params = {"objective": "reg:squarederror", "eta": self.learning_rate, "max_depth": int(self.max_depth),
                  "subsample": self.subsample, "colsample_bytree": self.colsample_bytree, "min_child_weight": self.min_child_weight,
                  "lambda": self.reg_lambda, "seed": self.random_state, "nthread": 1, "tree_method": "hist"}
        self.booster_ = xgboost.train(params, xgboost.DMatrix(np.asarray(X, float), label=np.asarray(y, float)), int(self.n_estimators))
        return self

    def predict(self, X):
        import xgboost
        return self.booster_.predict(xgboost.DMatrix(np.asarray(X, float)))

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


def _learners(binary, trees):
    """Cross-fitting learners. Same specification as the guide's, with lighter settings so they run in the browser."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    rf = dict(min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1, n_estimators=trees)
    return {"lasso": dict(reg=make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=50, random_state=0, max_iter=5000, tol=1e-3)),
                          clf=make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", Cs=10, max_iter=1000))),
            "rf": dict(reg=RandomForestRegressor(**rf), clf=RandomForestClassifier(**rf))}


def _split(n, ts, share, seed=0):
    k = max(5, int(round(n * share)))
    if ts:
        return np.arange(n - k), np.arange(n - k, n)
    idx = np.random.default_rng(seed).permutation(n)
    return np.sort(idx[k:]), np.sort(idx[:k])


def _fit_stats(model, X, y, tr, te, ts):
    m = model.fit(X.iloc[tr], y.iloc[tr])
    pred = m.predict(X.iloc[te])
    resid = y.iloc[te].to_numpy() - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r2 = 1 - float(resid @ resid) / float(((y.iloc[te] - y.iloc[tr].mean()) ** 2).sum())
    return m, pred, rmse, r2


def _predict(df, spec, kind):
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    y = need(spec, "y", "outcome to predict")
    x = as_list(need(spec, "x", "predictors"))
    distinct([y], x)
    d, ts = _data(df, spec, [y] + x)
    share = float_option(spec, "test_share", 0.2, "Test share")
    if not 0.05 <= share <= 0.5:
        raise OESError("The test share must be between 0.05 and 0.5.")
    X, Y = d[x].astype(float), d[y].astype(float)
    tr, te = _split(len(d), ts, share)
    models = M.models()
    if kind == "xgb":
        models["xgb"] = XGBoostRegressor(n_estimators=int_option(spec, "n_estimators", 400, 50, 3000, "Number of trees"),
                                         learning_rate=float_option(spec, "learning_rate", 0.05, "Learning rate"),
                                         max_depth=int_option(spec, "max_depth", 4, 1, 12, "Maximum depth"))
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        models["rf"] = RandomForestRegressor(n_estimators=int_option(spec, "n_estimators", 300, 50, 2000, "Number of trees"),
                                             min_samples_leaf=int_option(spec, "min_leaf", 5, 1, 100, "Minimum leaf size"),
                                             max_features=0.5, random_state=0, n_jobs=1)
    main = models[kind]
    m, pred, rmse, r2 = _fit_stats(clone(main), X, Y, tr, te, ts)
    cv = M.cv_rmse(clone(main), X.iloc[tr], Y.iloc[tr], ts)
    bench = []
    for name, key in (("OLS", "ols"), ("LASSO", "lasso"), ("Random forest", "rf")):
        if key == kind:
            continue
        _, _, r_, q_ = _fit_stats(clone(models[key]), X, Y, tr, te, ts)
        bench.append([name, f(r_, 4), f(q_, 4)])
    label = {"lasso": "LASSO", "ridge": "Ridge", "enet": "Elastic net", "rf": "Random forest", "xgb": "XGBoost"}[kind]
    tabs = [table("Out-of-sample accuracy", ["Model", "Test RMSE", "Test R-squared"], [[label, f(rmse, 4), f(r2, 4)]] + bench,
                  ("Test set: the last " if ts else "Test set: a random ") + f"{int(round(share * 100))}% of the observations"
                  + (" (time order kept; validation is rolling-origin)." if ts else "."))]
    summary = [stat("Observations", len(d)), stat("Predictors", len(x)), stat("CV RMSE (training)", cv), stat("Test RMSE", rmse), stat("Test R-squared", r2)]
    notes = []
    if kind in ("lasso", "ridge", "enet"):
        est = m[-1]
        coef = est.coef_
        alpha = float(est.alpha_)
        rows = [[v, f(c), "kept" if abs(c) > 1e-10 else "dropped"] for v, c in sorted(zip(x, coef), key=lambda z: -abs(z[1]))]
        tabs.append(table(f"{label} coefficients (standardised predictors)", ["Predictor", "Coefficient", "Status"], rows,
                          "Coefficients are per standard deviation of each predictor. Penalised estimates have no standard errors."))
        summary.append(stat("Penalty (alpha)", alpha, 6))
        if kind == "enet":
            summary.append(stat("L1 ratio", float(est.l1_ratio_), 2))
        if kind == "lasso":
            keep, a1 = M.lasso_1se(X.iloc[tr], Y.iloc[tr], ts)
            summary.append(stat("Kept at min CV", int(sum(abs(c) > 1e-10 for c in coef))))
            summary.append(stat("Kept at 1-s.e. rule", len(keep)))
            notes.append("One-standard-error rule keeps: " + (", ".join(keep) if keep else "no predictor") + ".")
    pi = permutation_importance(m, X.iloc[te], Y.iloc[te], n_repeats=10, random_state=0, scoring="neg_root_mean_squared_error")
    order = np.argsort(-pi.importances_mean)
    tabs.append(table("Permutation importance on the test set", ["Predictor", "RMSE increase", "Std."],
                      [[x[i], f(pi.importances_mean[i]), f(pi.importances_std[i])] for i in order]))
    top = order[: min(15, len(order))]
    charts = [chart("Permutation importance", [x[i] for i in top], [("RMSE increase", pi.importances_mean[top])], "", "", kind="bar"),
              chart("Test set: actual and predicted", list(range(1, len(te) + 1)), [("actual", Y.iloc[te].values), ("predicted", pred)], "test observation", y)]
    return result(kind, f"{label}: predicting {y}", len(d), summary, tabs, charts, notes)


def lasso(df, spec):
    return _predict(df, spec, "lasso")


def ridge(df, spec):
    return _predict(df, spec, "ridge")


def enet(df, spec):
    return _predict(df, spec, "enet")


def rf(df, spec):
    return _predict(df, spec, "rf")


def xgb(df, spec):
    return _predict(df, spec, "xgb")


# ------------------------------------------------------------------------------------------------ double machine learning
def _causal_data(df, spec, with_instr=False):
    y = need(spec, "y", "outcome")
    t = need(spec, "treat", "treatment")
    x = as_list(need(spec, "x", "controls"))
    z = need(spec, "instrument", "instrument") if with_instr else None
    distinct([y], [t], x, [z] if z else [])
    d, _ = _data(df, {}, [y, t] + x + ([z] if z else []))
    return d, y, t, x, z


def _ols_bench(d, y, t, x):
    import statsmodels.api as sm
    o = sm.OLS(d[y].astype(float), sm.add_constant(d[[t] + x].astype(float))).fit(cov_type="HC1")
    return float(o.params[t]), float(o.bse[t])


def _dml(df, spec, kind):
    d, y, t, x, z = _causal_data(df, spec, kind == "PLIV")
    folds = int_option(spec, "folds", 5, 2, 10, "Number of folds")
    reps = int_option(spec, "reps", 2, 1, 10, "Repetitions")
    binary = bool(M.is_binary(d[t]))
    if kind == "IRM" and not binary:
        raise OESError("The interactive model needs a binary (0/1) treatment. Use the partially linear model for a continuous treatment.")
    trees = int_option(spec, "trees", 100, 20, 1000, "Trees in the random forests")
    L = _learners(binary, trees)
    which = spec.get("learner", "both")
    keys = ["lasso", "rf"] if which == "both" else [which]
    rows, est = [], {}
    for k in keys:
        th, se = M.crossfit(d[y], d[t], d[x], L[k], "IRM" if kind == "IRM" else "PLR", Z=d[z] if z else None, folds=folds, reps=reps)
        est[k] = (th, se)
        rows.append([{"lasso": "DML with LASSO", "rf": "DML with random forest"}[k], th, se])
    ob, os_ = _ols_bench(d, y, t, x)
    rows.append(["OLS with controls (HC1)", ob, os_])
    title = {"PLR": "partially linear model", "IRM": "interactive model (ATE)", "PLIV": "partially linear IV model"}[kind]
    tabs = [coef_table([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], dist="z", title=f"Effect of {t} on {y}: {title}",
                       note=f"Cross-fitting with {folds} folds, {reps} repetition(s), median aggregation (Chernozhukov et al. 2018); "
                            f"random forests with {trees} trees.")]
    checks = []
    if len(est) == 2:
        g = abs(est["lasso"][0] - est["rf"][0]) / max(est["lasso"][1], est["rf"][1])
        checks.append(["Learner gap (in standard errors)", f(g, 3), None, "above 1: the learners disagree; nonlinearity matters, prefer the forest"])
    if kind == "IRM":
        from sklearn.model_selection import cross_val_predict
        ps = cross_val_predict(L["rf"]["clf"], d[x], d[t].astype(int), cv=5, method="predict_proba")[:, 1]
        out = float(np.mean((ps < 0.05) | (ps > 0.95)))
        checks.append(["Propensity score range", f"{f(ps.min(), 3)} to {f(ps.max(), 3)}", None, f"share outside [0.05, 0.95]: {f(out, 4)}"])
    if kind == "PLIV":
        import statsmodels.api as sm
        fs = sm.OLS(d[t].astype(float), sm.add_constant(d[[z] + x].astype(float))).fit(cov_type="HC1")
        F = float((fs.params[z] / fs.bse[z]) ** 2)
        checks.append(test_row("First-stage F (robust, linear)", F, float(fs.pvalues[z]), "below 10: weak instrument"))
    if checks:
        tabs.append(tests_table(checks, "Checks"))
    main = est.get("rf") or est[keys[0]]
    summary = [stat("Observations", len(d)), stat("Controls", len(x)), stat("Estimate", main[0]), stat("Std. err.", main[1]),
               stat("OLS estimate", ob)]
    return result({"PLR": "dml_plr", "IRM": "dml_irm", "PLIV": "dml_iv"}[kind], f"Double machine learning: {title}", len(d), summary, tabs)


def dml_plr(df, spec):
    return _dml(df, spec, "PLR")


def dml_irm(df, spec):
    return _dml(df, spec, "IRM")


def dml_iv(df, spec):
    return _dml(df, spec, "PLIV")


def causal_forest(df, spec):
    """Doubly robust forest (DR-learner): cross-fitted AIPW scores, then a random forest of the scores on the controls."""
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    import statsmodels.api as sm
    d, y, t, x, _ = _causal_data(df, spec)
    if not M.is_binary(d[t]):
        raise OESError("The causal forest here needs a binary (0/1) treatment.")
    folds = int_option(spec, "folds", 5, 2, 10, "Number of folds")
    Y, D, X = d[y].to_numpy(float), d[t].to_numpy(float), d[x].to_numpy(float)
    L = _learners(True, int_option(spec, "trees", 100, 20, 1000, "Trees in the nuisance forests"))["rf"]
    g0, g1, m = np.zeros_like(Y), np.zeros_like(Y), np.zeros_like(Y)
    kf = KFold(folds, shuffle=True, random_state=0)
    for tr, te in kf.split(X):
        for v, arr in ((0, g0), (1, g1)):
            idx = tr[D[tr] == v]
            arr[te] = clone(L["reg"]).fit(X[idx], Y[idx]).predict(X[te])
        m[te] = clone(L["clf"]).fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
    m = np.clip(m, 0.01, 0.99)
    psi = g1 - g0 + D * (Y - g1) / m - (1 - D) * (Y - g0) / (1 - m)
    ate, se = float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(Y)))
    leaf = int_option(spec, "min_leaf", 20, 5, 500, "Minimum leaf size")
    forest = RandomForestRegressor(n_estimators=int_option(spec, "n_estimators", 300, 50, 2000, "Number of trees"), min_samples_leaf=leaf,
                                   max_features=0.5, oob_score=True, random_state=0, n_jobs=1).fit(X, psi)
    tau = forest.oob_prediction_
    # best linear predictor test (Chernozhukov, Demirer, Duflo and Fernandez-Val): psi on 1 and (tau - mean tau)
    blp = sm.OLS(psi, sm.add_constant(tau - tau.mean())).fit(cov_type="HC1")
    q = pd.qcut(pd.Series(tau).rank(method="first"), 4, labels=False)
    gates = []
    for g in range(4):
        s = psi[q.values == g]
        gates.append([f"Quartile {g + 1} of predicted effect", f(float(np.mean(tau[q.values == g]))), f(float(s.mean())), f(float(s.std(ddof=1) / np.sqrt(len(s)))), int(len(s))])
    imp = sorted(zip(x, forest.feature_importances_), key=lambda z: -z[1])
    tabs = [coef_table(["Average treatment effect (doubly robust)"], [ate], [se], dist="z", title=f"Effect of {t} on {y}",
                       note="AIPW scores from cross-fitted random forests."),
            coef_table(["Average effect (intercept)", "Heterogeneity loading (slope on predicted effect)"], blp.params, blp.bse, dist="z",
                       title="Best linear predictor test of heterogeneity",
                       note="A slope significantly above 0 (ideally near 1) means the forest finds real differences in the effect."),
            table("Group average treatment effects (GATES)", ["Group", "Mean predicted effect", "Estimated effect", "Std. err.", "Obs."], gates),
            table("Which controls drive the differences", ["Variable", "Importance"], [[v, f(i, 4)] for v, i in imp])]
    summary = [stat("Observations", len(d)), stat("ATE", ate), stat("Std. err.", se),
               stat("Predicted effects: 10th pct", float(np.percentile(tau, 10))), stat("Predicted effects: 90th pct", float(np.percentile(tau, 90)))]
    notes = ["This is a doubly robust (DR-learner) forest: honest cross-fitted scores with out-of-bag predictions. It targets the same "
             "conditional effects as EconML's CausalForestDML but is not the same algorithm; EconML is not available in the browser."]
    if len(d) < 2000:
        notes.append("Effects by subgroup are noisy with fewer than about 2,000 observations.")
    hist, edges = np.histogram(tau, bins=20)
    charts = [chart("Distribution of predicted effects", [f((a + b) / 2, 4) for a, b in zip(edges[:-1], edges[1:])], [("count", hist)], "predicted effect", "count", kind="bar")]
    return result("causal_forest", f"Causal forest: effect of {t} on {y}", len(d), summary, tabs, charts, notes)
