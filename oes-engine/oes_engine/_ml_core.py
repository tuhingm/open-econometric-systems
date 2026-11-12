"""Decision engine for the Machine Learning Guide (ml-guide/index.html).

Runs at build time on the example datasets and in the visitor's browser (Pyodide) on uploaded data, so it uses only
numpy, pandas, scipy, statsmodels and scikit-learn. analyze(csv_text, goal, treat, instr) returns a JSON string.

Route A (prediction)                                    Route B (causal effect of one treatment D on the outcome Y)
  A1  time series (a date column, T >= 50): ADF + KPSS,     B2  OLS is enough if p < 10 controls and neither Y nor D
      I(1) variables are differenced; validation is              depends on the controls nonlinearly
      rolling-origin, never shuffled                        B3  treatment type: instrument -> DML-IV (first-stage F);
  A2  N < 100: regularised linear models only                    binary -> DML interactive model (overlap check);
  A3  random forest vs LASSO, cross-validated RMSE:              continuous -> DML partially linear model
      forest >= 5 % lower -> nonlinear                      B4  DML with LASSO and with random-forest learners;
  A4  linear: max |r| > 0.8 -> elastic net; LASSO keeps          a gap above one standard error -> nonlinear,
      < 50 % of predictors -> LASSO; otherwise ridge              report the forest
  A5  nonlinear: N < 1,000 -> random forest; else XGBoost   B5  N >= 2,000 and effects by subgroup -> causal forest
"""
import io, json, re, warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

MIN_TS = 50          # A1: periods needed to treat the data as a time series
SMALL_N = 100        # A2: below this, only regularised linear models
GAIN = 0.05          # A3: random forest must cut the CV RMSE by at least 5 %
HIGH_R = 0.8         # A4: correlation that calls for the elastic net
SPARSE = 0.5         # A4: LASSO keeps fewer than half of the predictors -> sparse
BIG_N = 1000         # A5: XGBoost needs at least this many observations
FEW_CTRL = 10        # B2: fewer controls than this, OLS may be enough
OVERLAP = (0.05, 0.95)   # B3: propensity scores outside this range are a problem
WEAK_F = 10          # B3: first-stage F below this means a weak instrument
CF_N = 2000          # B5: causal forests need at least this many observations
MAX_ROWS = 4000      # computing budget in the browser: larger files are subsampled for the forest checks
TIME_NAMES = re.compile(r"^(year|date|time|period|quarter|month|week|day|t|yq|ym)$", re.I)


def _f(x, d=4):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def read(text):
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    tcol = next((c for c in df.columns if TIME_NAMES.match(c)), None)
    if tcol:
        df = df.sort_values(tcol)
    num = [c for c in df.columns if c != tcol and pd.api.types.is_numeric_dtype(df[c])]
    skipped = [c for c in df.columns if c != tcol and c not in num]
    df = df[([tcol] if tcol else []) + num].dropna().reset_index(drop=True)
    return df, tcol, num, skipped


def is_binary(s):
    u = np.unique(s.dropna())
    return len(u) == 2 and set(np.round(u, 8)) <= {0.0, 1.0}


def unit_root(x):
    """ADF (AIC lag choice) and KPSS on the level, then on the first difference if needed. Returns the order and tests."""
    from statsmodels.tsa.stattools import adfuller, kpss
    out, s = {}, pd.Series(x).dropna()
    for order, lab in ((0, "level"), (1, "diff")):
        a = adfuller(s, autolag="AIC")
        try:
            k = kpss(s, regression="c", nlags="auto")
            kp = (k[0], k[1])
        except Exception:
            kp = (None, None)
        out[lab] = dict(adf=_f(a[0], 3), adf_p=_f(a[1], 4), kpss=_f(kp[0], 3), kpss_p=_f(kp[1], 3))
        if a[1] < 0.05:
            out["order"] = order
            return out
        s = s.diff().dropna()
    out["order"] = 2
    return out


def splitter(n, ts, k=5):
    from sklearn.model_selection import KFold, TimeSeriesSplit
    return TimeSeriesSplit(n_splits=k) if ts else KFold(n_splits=k, shuffle=True, random_state=0)


def cv_rmse(model, X, y, ts):
    from sklearn.model_selection import cross_val_score
    s = cross_val_score(model, X, y, cv=splitter(len(y), ts), scoring="neg_mean_squared_error")
    return float(np.sqrt(-s.mean()))


def models():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    return dict(
        ols=make_pipeline(StandardScaler(), LinearRegression()),
        lasso=make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0, max_iter=20000)),
        ridge=make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40))),
        enet=make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5, random_state=0, max_iter=20000)),
        rf=RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1),
        gb=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0))


def lasso_1se(X, y, ts):
    """Variables kept by LASSO at the one-standard-error penalty (the largest penalty whose CV error is within one
    standard error of the minimum; glmnet's lambda.1se), the usual choice when the aim is to see which predictors matter."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, Lasso
    Xs = StandardScaler().fit_transform(X)
    cv = LassoCV(cv=splitter(len(y), ts), random_state=0, max_iter=20000).fit(Xs, y)
    m, s = cv.mse_path_.mean(1), cv.mse_path_.std(1) / np.sqrt(cv.mse_path_.shape[1])
    i = int(np.argmin(m))
    ok = np.where(m <= m[i] + s[i])[0]
    alpha = float(cv.alphas_[ok].max())
    coef = Lasso(alpha=alpha, max_iter=20000).fit(Xs, y).coef_
    return [x for x, b in zip(X.columns, coef) if abs(b) > 1e-10], alpha


def summarize(df, cols):
    return [dict(var=c, mean=_f(df[c].mean()), sd=_f(df[c].std()), min=_f(df[c].min()), max=_f(df[c].max()),
                 binary=bool(is_binary(df[c]))) for c in cols]


def corr_info(X):
    c = X.corr().to_numpy(copy=True)
    np.fill_diagonal(c, 0)
    i, j = np.unravel_index(np.nanargmax(np.abs(c)), c.shape) if c.size > 1 else (0, 0)
    return dict(max_r=_f(abs(c[i, j]) if c.size > 1 else 0, 3), pair=[X.columns[i], X.columns[j]] if c.size > 1 else [],
                n_high=int((np.abs(np.triu(c, 1)) > HIGH_R).sum()))


def sub(df, n=MAX_ROWS, ts=False):
    if len(df) <= n:
        return df, False
    return (df.tail(n) if ts else df.sample(n, random_state=0)), True


# ------------------------------------------------------------------ route A: prediction
def predict(df, tcol, cols):
    y_name, xs = cols[0], cols[1:]
    R = dict(goal="predict", dep=y_name, xs=xs, n=len(df), p=len(xs), time=tcol)
    R["summary"] = summarize(df, cols)
    R["corr"] = corr_info(df[xs]) if len(xs) > 1 else dict(max_r=0, pair=[], n_high=0)
    ts = bool(tcol) and len(df) >= MIN_TS
    R["ts"] = ts
    d = df[cols].copy()
    if ts:
        ur, diffed, base_order = [], [], {}
        lag = re.compile(r"^(.*?)_(?:l|lag)(\d+)$")
        for c in cols:                     # lags of one series share its order: test each series once
            m = lag.match(c)
            base = m.group(1) if m else c
            if base not in base_order:
                if is_binary(df[c]) or df[c].nunique() < 8:
                    base_order[base] = 0
                    ur.append(dict(var=base, order=0, skipped=True))
                else:
                    u = unit_root(df[c]); u["var"] = base; ur.append(u)
                    base_order[base] = u["order"]
            if base_order[base] >= 1:
                d[c] = d[c].diff(); diffed.append(c)
        d = d.dropna().reset_index(drop=True)
        R["unitroot"], R["diffed"] = ur, diffed
        R["first"], R["last"] = str(df[tcol].iloc[0]), str(df[tcol].iloc[-1])
    n = len(d)
    R["n_used"] = n
    X, y = d[xs], d[y_name]
    M = models()
    dd, R["subsampled"] = sub(d, ts=ts)
    Xs, ys = dd[xs], dd[y_name]
    R["rmse"] = {"ols": _f(cv_rmse(M["ols"], Xs, ys, ts)), "lasso": _f(cv_rmse(M["lasso"], Xs, ys, ts))}
    R["sd_y"] = _f(float(ys.std()))
    if n < SMALL_N:
        R["small"] = True
        R["nonlinear"] = None
    else:
        R["small"] = False
        R["rmse"]["rf"] = _f(cv_rmse(M["rf"], Xs, ys, ts))
        R["gain"] = _f(1 - R["rmse"]["rf"] / R["rmse"]["lasso"], 4)
        R["nonlinear"] = bool(R["gain"] >= GAIN)
    if R["nonlinear"]:
        if n >= BIG_N:
            R["rmse"]["gb"] = _f(cv_rmse(M["gb"], Xs, ys, ts))
        R["model"] = "XGB" if n >= BIG_N else "RF"
    else:
        kept, alpha = lasso_1se(X, y, ts)
        R["lasso_kept"], R["lasso_share"] = kept, _f(len(kept) / len(xs), 3)
        R["lasso_alpha"] = _f(alpha, 6)
        R["rmse"]["ridge"] = _f(cv_rmse(M["ridge"], Xs, ys, ts))
        R["rmse"]["enet"] = _f(cv_rmse(M["enet"], Xs, ys, ts))
        if R["corr"]["max_r"] > HIGH_R:
            R["model"] = "ENET"
        elif len(kept) / len(xs) < SPARSE:
            R["model"] = "LASSO"
        else:
            R["model"] = "RIDGE"
    return R


# ------------------------------------------------------------------ route B: causal effect
def crossfit(y, D, X, learner, kind, Z=None, folds=5, reps=2, seed=0):
    """Double machine learning by cross-fitting (Chernozhukov et al. 2018). Returns theta, se (median over repetitions)."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import RandomForestClassifier
    y, D, X = np.asarray(y, float), np.asarray(D, float), np.asarray(X, float)
    Z = None if Z is None else np.asarray(Z, float)
    ths, ses = [], []
    for r in range(reps):
        kf = KFold(folds, shuffle=True, random_state=seed + r)
        if kind == "IRM":
            g0, g1, m = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
            for tr, te in kf.split(X):
                for v, arr in ((0, g0), (1, g1)):
                    idx = tr[D[tr] == v]
                    arr[te] = clone(learner["reg"]).fit(X[idx], y[idx]).predict(X[te])
                clf = clone(learner["clf"]).fit(X[tr], D[tr])
                m[te] = clf.predict_proba(X[te])[:, 1]
            m = np.clip(m, 0.01, 0.99)
            psi = g1 - g0 + D * (y - g1) / m - (1 - D) * (y - g0) / (1 - m)
            th = psi.mean(); se = psi.std(ddof=1) / np.sqrt(len(y))
        else:
            ly, md, mz = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
            for tr, te in kf.split(X):
                ly[te] = clone(learner["reg"]).fit(X[tr], y[tr]).predict(X[te])
                md[te] = clone(learner["reg"]).fit(X[tr], D[tr]).predict(X[te])
                if Z is not None:
                    mz[te] = clone(learner["reg"]).fit(X[tr], Z[tr]).predict(X[te])
            uy, ud = y - ly, D - md
            v = ud if Z is None else Z - mz
            th = (v @ uy) / (v @ ud)
            e = uy - th * ud
            J = -(v @ ud) / len(y)
            se = np.sqrt(np.mean((v * e) ** 2) / J ** 2 / len(y))
        ths.append(th); ses.append(se)
    th = float(np.median(ths))
    se = float(np.median(np.sqrt(np.array(ses) ** 2 + (np.array(ths) - th) ** 2)))
    return th, se


def learners(binary=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    return {"lasso": dict(reg=make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0, max_iter=20000)),
                          clf=make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, penalty="l1", solver="saga", max_iter=5000, Cs=10))),
            "rf": dict(reg=RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1),
                       clf=RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1))}


def causal(df, cols, treat, instr):
    import statsmodels.api as sm
    y_name = cols[0]
    treat = treat if treat in cols and treat != y_name else cols[1]
    instr = instr if instr and instr in cols and instr not in (y_name, treat) else None
    xs = [c for c in cols if c not in (y_name, treat, instr)]
    R = dict(goal="causal", dep=y_name, treat=treat, instr=instr, xs=xs, n=len(df), p=len(xs))
    R["summary"] = summarize(df, [y_name, treat] + ([instr] if instr else []) + xs)
    R["corr"] = corr_info(df[xs]) if len(xs) > 1 else dict(max_r=0, pair=[], n_high=0)
    R["binary"] = bool(is_binary(df[treat]))
    dd, R["subsampled"] = sub(df)
    y, D, X = dd[y_name], dd[treat], dd[xs]
    # OLS benchmark with robust standard errors
    o = sm.OLS(df[y_name], sm.add_constant(df[[treat] + xs])).fit(cov_type="HC1")
    R["ols"] = dict(b=_f(o.params[treat]), se=_f(o.bse[treat]), p=_f(o.pvalues[treat], 4))
    # B2: do the controls enter nonlinearly? random forest vs LASSO for Y on X and for D on X
    M = models()
    nl = {}
    for lab, tgt in (("y", y), ("d", D)):
        l, r = cv_rmse(M["lasso"], X, tgt, False), cv_rmse(M["rf"], X, tgt, False)
        nl[lab] = dict(lasso=_f(l), rf=_f(r), gain=_f(1 - r / l, 4))
    R["nl"] = nl
    R["controls_nonlinear"] = bool(max(nl["y"]["gain"], nl["d"]["gain"]) >= GAIN)
    R["ols_enough"] = bool(len(xs) < FEW_CTRL and not R["controls_nonlinear"])
    # B3: treatment type, overlap, instrument strength
    if instr:
        fs = sm.OLS(df[treat], sm.add_constant(df[[instr] + xs])).fit(cov_type="HC1")
        R["first_stage"] = dict(b=_f(fs.params[instr]), se=_f(fs.bse[instr]), F=_f((fs.params[instr] / fs.bse[instr]) ** 2, 2))
        R["weak"] = bool(R["first_stage"]["F"] < WEAK_F)
        kind = "PLIV"
    elif R["binary"]:
        from sklearn.model_selection import cross_val_predict
        from sklearn.ensemble import RandomForestClassifier
        ps = cross_val_predict(RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=1),
                               X, D.astype(int), cv=5, method="predict_proba")[:, 1]
        lo, hi = OVERLAP
        R["overlap"] = dict(min=_f(ps.min(), 3), max=_f(ps.max(), 3), share_out=_f(np.mean((ps < lo) | (ps > hi)), 4),
                            hist_t=np.histogram(ps[D.to_numpy() == 1], bins=20, range=(0, 1))[0].tolist(),
                            hist_c=np.histogram(ps[D.to_numpy() == 0], bins=20, range=(0, 1))[0].tolist())
        R["treated_share"] = _f(D.mean(), 4)
        kind = "IRM"
    else:
        kind = "PLR"
    R["kind"] = kind
    # B4: DML with both learners
    L = learners(R["binary"])
    Z = dd[instr] if instr else None
    est = {}
    for k in ("lasso", "rf"):
        th, se = crossfit(y, D, X, L[k], kind, Z=Z)
        est[k] = dict(theta=_f(th), se=_f(se), lo=_f(th - 1.96 * se), hi=_f(th + 1.96 * se))
    R["dml"] = est
    gap = abs(est["lasso"]["theta"] - est["rf"]["theta"])
    R["gap_se"] = _f(gap / max(est["lasso"]["se"], est["rf"]["se"]), 3)
    R["learner"] = "rf" if R["gap_se"] > 1 else "both"
    R["cf_possible"] = bool(len(df) >= CF_N)
    R["model"] = "OLS" if R["ols_enough"] and not instr else kind
    return R


def analyze(text, goal="predict", treat=None, instr=None):
    try:
        df, tcol, num, skipped = read(text)
        if len(num) < 2:
            return json.dumps(dict(error="The file needs at least two numeric columns: the outcome and one predictor."))
        if len(df) < 30:
            return json.dumps(dict(error=f"Only {len(df)} complete rows. The checks need at least 30."))
        if goal == "causal":
            if len(num) < 3 and not treat:
                return json.dumps(dict(error="A causal analysis needs the outcome, the treatment and at least one control."))
            R = causal(df, num, treat, instr)
        else:
            R = predict(df, tcol, num)
        R["skipped"] = skipped
        R["cols"] = num
        return json.dumps(R)
    except Exception as e:  # report the problem to the page instead of failing silently
        return json.dumps(dict(error=f"{type(e).__name__}: {e}"))
