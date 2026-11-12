"""Model-selection analysis for the Panel-Data Research Guide.

analyze(csv_text) reads a long-format panel CSV: entity identifier, time identifier, then the dependent variable and the
regressors (one column per variable). It reports the panel structure, summary statistics (overall, between, within),
correlations and then one of two test sequences:

The route depends on both dimensions: the number of units N and the median number of periods per unit T.

* T below LONG_T (short panels): F test for unit effects, Breusch-Pagan LM test, Hausman and Mundlak tests,
  with pooled OLS, random-effects and fixed-effects estimates;
  A dynamic model is sent to GMM only when N >= GMM_N; with fewer units, to bias-corrected fixed effects (LSDVC).
* T of LONG_T or more and N >= FEW_N (long panels): Pesaran CD tests, the Pesaran-Yamagata slope-homogeneity test, panel unit-root tests (Fisher-ADF and
  CIPS) on levels and differences, and, when every variable is I(1), Kao and Fisher-Engle-Granger cointegration tests.
* T of LONG_T or more but N below FEW_N: too few units for panel asymptotics; each unit is modelled as a time series.

Returns a JSON string. The same file runs in the browser (Pyodide) for uploaded data and offline for the examples.
"""
import io, json, math, warnings
import numpy as np
import pandas as pd

LONG_T = 15     # periods: below this the panel is short
GMM_N = 30      # units: difference / system GMM needs many units
FEW_N = 10      # units: CD, CIPS, CCE and CS-ARDL need at least this many
CIPS_Q = {"p":1,"N":[5,10,15,20,30,50,100,200],"T":[10,15,20,25,30,40,50,75,100],"probs":[0.001,0.005,0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99],"q":[[[-17.0544,-10.8133,-8.3805,-6.0689,-4.664,-3.6353,-2.7288,-2.245,-1.9329,-1.6544,-1.4076,-1.1434,-0.8754,-0.4853,-0.1938,0.6766],[-4.0929,-3.422,-3.1978,-2.9654,-2.7229,-2.4617,-2.1603,-1.9448,-1.7494,-1.5869,-1.4356,-1.2758,-1.0999,-0.8379,-0.6369,-0.2343],[-3.442,-3.1381,-2.9779,-2.7353,-2.5629,-2.3649,-2.1206,-1.9551,-1.8065,-1.668,-1.5296,-1.3809,-1.2042,-0.9416,-0.7127,-0.3599],[-3.4224,-3.1389,-3.0014,-2.7549,-2.5733,-2.3648,-2.1452,-1.9797,-1.8271,-1.7025,-1.5734,-1.4223,-1.2583,-1.0364,-0.8285,-0.3891],[-3.2526,-3.0311,-2.9058,-2.743,-2.5838,-2.3837,-2.1593,-1.9872,-1.8362,-1.7053,-1.5719,-1.4309,-1.2778,-1.0278,-0.8362,-0.474],[-3.142,-3.0126,-2.8966,-2.7246,-2.5614,-2.3674,-2.1593,-2.0009,-1.8619,-1.7322,-1.6059,-1.4668,-1.3001,-1.096,-0.9113,-0.496],[-3.2401,-3.0193,-2.9121,-2.7176,-2.5543,-2.3692,-2.1557,-2.0041,-1.8798,-1.7503,-1.6274,-1.4978,-1.3394,-1.1089,-0.9341,-0.5657],[-3.1784,-3.0,-2.8952,-2.7091,-2.5685,-2.3934,-2.1777,-2.0225,-1.8995,-1.7836,-1.6636,-1.5382,-1.3694,-1.1523,-0.9624,-0.6248],[-3.2549,-2.9794,-2.811,-2.6703,-2.5451,-2.3892,-2.1766,-2.028,-1.9108,-1.7856,-1.6723,-1.5536,-1.4076,-1.1802,-0.9871,-0.6391]],[[-22.5906,-7.895,-6.484,-4.7427,-3.9443,-3.1199,-2.5263,-2.1809,-1.9094,-1.6905,-1.487,-1.2852,-1.0733,-0.7775,-0.5291,-0.036],[-3.2373,-2.8668,-2.7572,-2.5845,-2.4116,-2.2278,-2.0263,-1.8703,-1.7396,-1.621,-1.4993,-1.3699,-1.2262,-1.0275,-0.871,-0.5717],[-2.8719,-2.7456,-2.649,-2.4952,-2.3708,-2.1975,-2.0108,-1.8781,-1.7546,-1.6446,-1.5442,-1.4333,-1.3145,-1.1304,-0.9735,-0.7148],[-2.9173,-2.7077,-2.6093,-2.46,-2.3501,-2.2046,-2.0275,-1.9056,-1.7977,-1.7009,-1.5993,-1.4941,-1.3718,-1.2004,-1.0732,-0.8092],[-2.8747,-2.67,-2.5727,-2.4494,-2.3321,-2.1912,-2.0317,-1.9106,-1.801,-1.7118,-1.6211,-1.526,-1.4045,-1.2211,-1.0848,-0.8044],[-2.7626,-2.6151,-2.5519,-2.4454,-2.3102,-2.1944,-2.039,-1.925,-1.8311,-1.7382,-1.643,-1.5424,-1.4346,-1.2726,-1.144,-0.8867],[-2.7844,-2.5834,-2.5079,-2.4084,-2.3098,-2.1887,-2.0492,-1.9345,-1.8453,-1.757,-1.6693,-1.5769,-1.4708,-1.3159,-1.177,-0.9308],[-2.8478,-2.6676,-2.5307,-2.4171,-2.3253,-2.1961,-2.0425,-1.9445,-1.8524,-1.7624,-1.6811,-1.599,-1.4902,-1.3398,-1.2203,-0.97],[-2.7869,-2.6062,-2.5466,-2.4278,-2.3174,-2.2074,-2.0646,-1.9633,-1.8706,-1.7809,-1.6975,-1.6046,-1.494,-1.3371,-1.1968,-0.97]],[[-12.8944,-7.722,-5.7105,-4.3254,-3.6372,-3.0768,-2.5065,-2.1969,-1.9465,-1.7585,-1.5678,-1.3884,-1.1947,-0.9154,-0.7032,-0.3059],[-3.0812,-2.7438,-2.5966,-2.419,-2.2927,-2.1338,-1.9458,-1.8189,-1.7177,-1.6145,-1.5157,-1.4062,-1.2845,-1.119,-0.9843,-0.7628],[-2.6864,-2.5549,-2.5018,-2.3407,-2.2372,-2.1243,-1.9767,-1.8545,-1.7587,-1.6725,-1.5795,-1.478,-1.3732,-1.2232,-1.1162,-0.9259],[-2.7387,-2.5312,-2.4752,-2.3672,-2.2631,-2.1306,-1.9868,-1.8818,-1.7931,-1.7105,-1.6188,-1.5244,-1.4336,-1.2895,-1.1802,-0.9564],[-2.6307,-2.522,-2.4351,-2.3335,-2.2447,-2.1301,-1.9918,-1.8904,-1.8014,-1.7162,-1.6304,-1.544,-1.4425,-1.2998,-1.1828,-0.9782],[-2.7402,-2.5416,-2.4392,-2.3266,-2.2359,-2.1329,-2.0058,-1.9031,-1.8168,-1.7384,-1.6571,-1.5713,-1.467,-1.3177,-1.2036,-1.0204],[-2.629,-2.4929,-2.4147,-2.3163,-2.2471,-2.1361,-2.0083,-1.9063,-1.8248,-1.7448,-1.6736,-1.5935,-1.4893,-1.3562,-1.2506,-1.0552],[-2.5898,-2.4905,-2.4193,-2.3216,-2.2437,-2.1362,-2.0076,-1.9217,-1.8485,-1.765,-1.6896,-1.6099,-1.5106,-1.384,-1.2691,-1.0807],[-2.6136,-2.4863,-2.4356,-2.3332,-2.242,-2.1426,-2.0186,-1.9277,-1.847,-1.7696,-1.6939,-1.6168,-1.5286,-1.4016,-1.3162,-1.1167]],[[-10.4906,-7.1271,-5.3017,-4.1929,-3.5006,-2.9679,-2.4601,-2.1653,-1.9311,-1.7561,-1.5941,-1.4352,-1.2443,-1.0103,-0.8056,-0.3262],[-2.7642,-2.6009,-2.5123,-2.3583,-2.253,-2.1137,-1.9504,-1.8294,-1.724,-1.6228,-1.5266,-1.4299,-1.312,-1.1641,-1.0503,-0.8109],[-2.561,-2.4605,-2.3724,-2.2726,-2.1876,-2.0746,-1.9327,-1.8317,-1.7467,-1.6642,-1.5875,-1.5061,-1.3992,-1.2546,-1.1421,-0.9492],[-2.6216,-2.4372,-2.3886,-2.2873,-2.1932,-2.089,-1.9548,-1.8553,-1.7762,-1.6963,-1.6162,-1.5351,-1.4385,-1.3104,-1.2077,-1.0128],[-2.667,-2.4527,-2.393,-2.2646,-2.1797,-2.0886,-1.955,-1.8627,-1.7861,-1.7079,-1.6291,-1.5515,-1.457,-1.346,-1.2395,-1.0372],[-2.5204,-2.4036,-2.3532,-2.2702,-2.1875,-2.0869,-1.9765,-1.8865,-1.8029,-1.7262,-1.6565,-1.582,-1.494,-1.3658,-1.2619,-1.0882],[-2.5155,-2.4097,-2.3648,-2.2629,-2.188,-2.0903,-1.977,-1.8893,-1.8157,-1.7439,-1.6768,-1.5977,-1.5107,-1.3962,-1.2979,-1.1219],[-2.5746,-2.3921,-2.3505,-2.2614,-2.1991,-2.1032,-1.9924,-1.9123,-1.8404,-1.7693,-1.7039,-1.6324,-1.5498,-1.429,-1.3423,-1.1903],[-2.4782,-2.3729,-2.3279,-2.2633,-2.1955,-2.1067,-1.9985,-1.9128,-1.8393,-1.7732,-1.7001,-1.6243,-1.5376,-1.42,-1.3297,-1.1788]],[[-8.665,-5.9596,-4.6128,-3.9195,-3.3325,-2.8536,-2.421,-2.1663,-1.9699,-1.7949,-1.6333,-1.4847,-1.3114,-1.0761,-0.9108,-0.5893],[-2.6115,-2.4627,-2.3739,-2.2611,-2.1574,-2.0305,-1.8909,-1.7842,-1.6934,-1.6061,-1.525,-1.4309,-1.337,-1.2048,-1.1029,-0.9327],[-2.5001,-2.3493,-2.2913,-2.2084,-2.1308,-2.035,-1.9083,-1.8215,-1.7434,-1.6599,-1.5907,-1.5092,-1.4141,-1.2901,-1.1989,-1.0248],[-2.4762,-2.3635,-2.3163,-2.212,-2.1295,-2.0433,-1.9194,-1.8361,-1.7622,-1.6911,-1.6138,-1.5393,-1.4575,-1.3454,-1.2596,-1.1095],[-2.4464,-2.3399,-2.2693,-2.2054,-2.1379,-2.0452,-1.9362,-1.848,-1.7768,-1.7131,-1.6495,-1.579,-1.4876,-1.3819,-1.2907,-1.1459],[-2.4234,-2.3444,-2.2885,-2.2151,-2.1413,-2.0578,-1.9485,-1.8686,-1.8023,-1.7423,-1.6701,-1.6021,-1.5227,-1.4125,-1.3401,-1.1648],[-2.4084,-2.324,-2.2788,-2.2032,-2.1373,-2.0589,-1.9548,-1.8757,-1.8076,-1.7462,-1.6837,-1.6145,-1.5388,-1.4258,-1.3419,-1.1964],[-2.399,-2.3374,-2.29,-2.2216,-2.1525,-2.0767,-1.9752,-1.9014,-1.8317,-1.7724,-1.7038,-1.6355,-1.56,-1.4632,-1.3839,-1.2449],[-2.3854,-2.3128,-2.2745,-2.2005,-2.1478,-2.0739,-1.9668,-1.8897,-1.8307,-1.7699,-1.7083,-1.647,-1.576,-1.4766,-1.4041,-1.2723]],[[-7.7857,-5.2741,-4.3686,-3.6088,-3.138,-2.7622,-2.3988,-2.1497,-1.9915,-1.8353,-1.6964,-1.5547,-1.4012,-1.1957,-1.0279,-0.7105],[-2.5678,-2.3961,-2.3031,-2.1989,-2.1026,-1.9976,-1.8683,-1.7733,-1.6933,-1.6259,-1.5451,-1.46,-1.3659,-1.2518,-1.167,-1.0085],[-2.3878,-2.3014,-2.2439,-2.1524,-2.0814,-1.9995,-1.8861,-1.8116,-1.7316,-1.6602,-1.5938,-1.5175,-1.4408,-1.3366,-1.2532,-1.1109],[-2.4055,-2.3024,-2.25,-2.1618,-2.0872,-1.9995,-1.8985,-1.8172,-1.7448,-1.6803,-1.6203,-1.552,-1.478,-1.3807,-1.2952,-1.163],[-2.405,-2.3086,-2.2486,-2.1797,-2.1021,-2.0202,-1.9154,-1.843,-1.7788,-1.7175,-1.6534,-1.5829,-1.5101,-1.4198,-1.3422,-1.2218],[-2.3345,-2.2614,-2.2192,-2.1564,-2.097,-2.0257,-1.9269,-1.8564,-1.7939,-1.7324,-1.6749,-1.6157,-1.5484,-1.4508,-1.3815,-1.2415],[-2.3407,-2.2591,-2.2301,-2.1596,-2.1051,-2.0394,-1.9401,-1.8663,-1.8045,-1.7436,-1.6867,-1.6325,-1.5637,-1.4792,-1.4113,-1.2923],[-2.3269,-2.2629,-2.2314,-2.1632,-2.105,-2.0346,-1.9463,-1.8766,-1.8183,-1.7598,-1.7039,-1.6489,-1.5885,-1.5063,-1.4414,-1.3287],[-2.3717,-2.2559,-2.2284,-2.1642,-2.1089,-2.0461,-1.9568,-1.8911,-1.8322,-1.7726,-1.7173,-1.657,-1.5908,-1.5137,-1.4511,-1.3431]],[[-9.1091,-3.9298,-3.6372,-3.2191,-2.8902,-2.6172,-2.2914,-2.1365,-1.998,-1.8688,-1.7432,-1.6186,-1.4665,-1.2844,-1.1448,-0.8966],[-2.4678,-2.2988,-2.223,-2.1406,-2.0465,-1.9599,-1.8464,-1.7605,-1.6873,-1.6177,-1.5475,-1.4775,-1.3979,-1.2892,-1.2134,-1.0792],[-2.3442,-2.2446,-2.1834,-2.1025,-2.0392,-1.9624,-1.8791,-1.8006,-1.7349,-1.6741,-1.6103,-1.5404,-1.4645,-1.3736,-1.3074,-1.1923],[-2.2931,-2.2217,-2.1619,-2.1054,-2.0496,-1.9802,-1.889,-1.8211,-1.756,-1.6917,-1.6312,-1.5728,-1.5075,-1.4227,-1.3581,-1.2545],[-2.2888,-2.2188,-2.1705,-2.1114,-2.0575,-1.9938,-1.9006,-1.833,-1.7727,-1.7076,-1.649,-1.5908,-1.5218,-1.4393,-1.3851,-1.2935],[-2.2564,-2.1924,-2.1713,-2.1159,-2.0653,-2.0054,-1.9221,-1.8454,-1.78,-1.724,-1.6713,-1.6194,-1.5651,-1.4927,-1.4372,-1.3427],[-2.2643,-2.2015,-2.1746,-2.121,-2.0758,-2.0074,-1.9261,-1.8567,-1.8019,-1.7465,-1.6942,-1.6412,-1.5888,-1.5182,-1.4606,-1.3605],[-2.2357,-2.1959,-2.1694,-2.1217,-2.0821,-2.0198,-1.9381,-1.8741,-1.8143,-1.7613,-1.7087,-1.6598,-1.6012,-1.533,-1.4848,-1.392],[-2.271,-2.2012,-2.1789,-2.1249,-2.0748,-2.0197,-1.94,-1.8714,-1.8142,-1.7603,-1.7108,-1.6635,-1.6142,-1.5523,-1.4968,-1.4016]],[[-5.4223,-3.7569,-3.3479,-2.9736,-2.7367,-2.5031,-2.2655,-2.11,-1.9855,-1.8667,-1.7604,-1.6509,-1.5157,-1.349,-1.2265,-1.0292],[-2.3742,-2.2418,-2.1885,-2.0995,-2.029,-1.9466,-1.8429,-1.7568,-1.6893,-1.6175,-1.553,-1.4828,-1.4012,-1.3081,-1.238,-1.1299],[-2.2673,-2.1799,-2.1444,-2.0834,-2.0256,-1.9561,-1.8637,-1.7894,-1.7307,-1.6642,-1.599,-1.5371,-1.4675,-1.3881,-1.3324,-1.2413],[-2.2361,-2.1749,-2.1316,-2.084,-2.0317,-1.9671,-1.8782,-1.8022,-1.738,-1.6809,-1.6239,-1.5654,-1.5024,-1.4301,-1.3768,-1.2925],[-2.2613,-2.1766,-2.1382,-2.086,-2.0332,-1.9781,-1.8991,-1.8265,-1.759,-1.703,-1.6468,-1.5951,-1.5348,-1.4615,-1.4102,-1.3351],[-2.2244,-2.1688,-2.1263,-2.0819,-2.0447,-1.9882,-1.9055,-1.8435,-1.7881,-1.732,-1.6786,-1.6263,-1.5714,-1.5078,-1.46,-1.3872],[-2.2143,-2.1641,-2.1414,-2.0944,-2.0531,-1.9994,-1.9172,-1.8452,-1.789,-1.7328,-1.6819,-1.6326,-1.5838,-1.523,-1.4768,-1.4127],[-2.2013,-2.1543,-2.1297,-2.0912,-2.0501,-2.0002,-1.9241,-1.8659,-1.8063,-1.7526,-1.7033,-1.6592,-1.6098,-1.5531,-1.5092,-1.4354],[-2.2154,-2.1753,-2.1469,-2.0965,-2.0545,-2.0066,-1.9256,-1.8667,-1.8131,-1.7597,-1.7123,-1.6664,-1.618,-1.5586,-1.5148,-1.4497]]]}  # filled in by the page builder: simulated quantiles of the CIPS statistic (see simulate_cips.py)
ID_NAMES = ("id", "entity", "unit", "country", "code", "iso", "iso3", "state", "province", "region", "county", "city",
            "firm", "company", "bank", "industry", "sector", "nr", "person", "pid", "hhid", "household", "individual", "panel")
TIME_NAMES = ("year", "time", "t", "period", "date", "month", "quarter", "yr", "wave")


def _f(x, nd=4):
    try:
        x = float(x)
    except Exception:
        return None
    return None if not np.isfinite(x) else round(x, nd)


def _ncdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _ninv(p):
    from scipy.stats import norm
    return float(norm.ppf(p))


def _chi2_sf(x, df):
    from scipy.stats import chi2
    return float(chi2.sf(x, df))


def _f_sf(x, d1, d2):
    from scipy.stats import f
    return float(f.sf(x, d1, d2))


def _ols(y, X):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    return b, e, XtX_inv


def _cluster_vcov(X, e, XtX_inv, groups, k_used):
    n = len(e)
    G = len(np.unique(groups))
    order = np.argsort(groups, kind="stable")
    g, Xs, es = groups[order], X[order], e[order]
    cut = np.flatnonzero(np.diff(g)) + 1
    meat = np.zeros((X.shape[1], X.shape[1]))
    for Xg, eg in zip(np.split(Xs, cut), np.split(es, cut)):
        s = Xg.T @ eg
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / max(n - k_used, 1)) if G > 1 else 1.0
    return c * XtX_inv @ meat @ XtX_inv


def _demean(a, codes, ng):
    a = np.asarray(a, float)
    if a.ndim == 1:
        m = np.bincount(codes, a, ng) / np.bincount(codes, minlength=ng)
        return a - m[codes]
    out = np.empty_like(a)
    cnt = np.bincount(codes, minlength=ng)
    for j in range(a.shape[1]):
        out[:, j] = a[:, j] - (np.bincount(codes, a[:, j], ng) / cnt)[codes]
    return out


def _gmeans(a, codes, ng):
    cnt = np.bincount(codes, minlength=ng)
    if a.ndim == 1:
        return (np.bincount(codes, a, ng) / cnt)
    return np.column_stack([np.bincount(codes, a[:, j], ng) / cnt for j in range(a.shape[1])])


def _independent_cols(X, tol=1e-8):
    """Indices of a maximal set of linearly independent columns (keeps the earlier column when two are collinear)."""
    keep = []
    for j in range(X.shape[1]):
        cand = keep + [j]
        sub = X[:, cand]
        if np.linalg.matrix_rank(sub, tol=tol * max(1.0, np.abs(sub).max())) == len(cand):
            keep.append(j)
    return keep


# ---------------------------------------------------------------- reading and structure

def _read(csv_text, max_vars):
    try:
        raw = pd.read_csv(io.StringIO(csv_text), sep=None, engine="python")
    except Exception as e:
        return None, f"Could not read the file as CSV ({e})."
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.loc[:, [c for c in raw.columns if not c.lower().startswith("unnamed")]]
    if raw.shape[1] < 4:
        return None, "The file needs at least four columns: entity, time, the dependent variable and one regressor."
    cols = list(raw.columns)
    low = [c.lower() for c in cols]
    ent = next((cols[i] for i, c in enumerate(low) if c in ID_NAMES), None)
    tim = next((cols[i] for i, c in enumerate(low) if c in TIME_NAMES and cols[i] != ent), None)
    if ent is None or tim is None:
        ent, tim = cols[0], cols[1]
    num = raw.drop(columns=[ent, tim]).apply(pd.to_numeric, errors="coerce")
    keep = [c for c in num.columns if num[c].notna().mean() > 0.8 and num[c].nunique() >= 2]
    skipped = [c for c in num.columns if c not in keep]
    if keep and num[keep[0]].nunique() <= 2:
        return None, f"The first variable after the identifiers ({keep[0]}) is treated as the dependent variable, but it takes only two values. This guide covers continuous outcomes; see the binary and count models section below."
    keep = keep[:max_vars]
    if len(keep) < 2:
        return None, "The file needs a dependent variable and at least one numeric regressor after the entity and time columns."
    df = pd.concat([raw[[ent, tim]], num[keep]], axis=1).dropna().copy()
    df[ent] = df[ent].astype(str).str.strip()
    tnum = pd.to_numeric(df[tim], errors="coerce")
    df["_t"] = tnum if tnum.notna().all() else df[tim].astype(str).str.strip()
    if df.duplicated([ent, "_t"]).any():
        d = df[df.duplicated([ent, "_t"], keep=False)].iloc[0]
        return None, f"Entity “{d[ent]}” appears more than once in period {d[tim]}. Each entity–period pair must be a single row (long format)."
    df = df.sort_values([ent, "_t"]).reset_index(drop=True)
    periods = sorted(df["_t"].unique())
    tmap = {p: i for i, p in enumerate(periods)}
    df["_ti"] = df["_t"].map(tmap).astype(int)
    cnt = df.groupby(ent)[tim].transform("size")
    df = df[cnt >= 2].reset_index(drop=True)
    return dict(df=df, ent=ent, tim=tim, vars=keep, skipped=skipped, periods=periods), None


# ---------------------------------------------------------------- short-panel (static) tests

def _short_panel(df, ent, dep, xs):
    codes, uniq = pd.factorize(df[ent])
    ng = len(uniq)
    y = df[dep].to_numpy(float)
    Xr = df[xs].to_numpy(float)
    n = len(y)
    tcodes = df["_ti"].to_numpy()
    tlev = np.unique(tcodes)
    D = (tcodes[:, None] == tlev[None, 1:]).astype(float)           # time dummies (first period omitted)
    out = {"time_effects": True}

    # regressors usable in FE: varying within units and not collinear with unit and time effects
    Xw, Dw = _demean(Xr, codes, ng), _demean(D, codes, ng)
    within_sd = Xw.std(axis=0)
    tinv = [x for x, s in zip(xs, within_sd) if s < 1e-10]
    cand = [j for j, s in enumerate(within_sd) if s >= 1e-10]
    if cand:
        Z = np.column_stack([Dw, Xw[:, cand]]) if D.shape[1] else Xw[:, cand]
        ind = _independent_cols(Z)
        ok = [cand[i - D.shape[1]] for i in ind if i >= D.shape[1]]
    else:
        ok = []
    coll = [xs[j] for j in cand if j not in ok]
    fx = [xs[j] for j in ok]
    out["fe_dropped"] = dict(time_invariant=tinv, collinear=coll)
    if not fx:
        return {"error": "No regressor varies within units once unit and time effects are removed, so the fixed-effects and Hausman comparisons cannot be run."}

    # pooled OLS with time dummies (all regressors that are not collinear with the time dummies)
    Xp_full = np.column_stack([np.ones(n), Xr, D])
    pk = _independent_cols(Xp_full)
    px = [xs[j - 1] for j in pk if 1 <= j <= len(xs)]
    Xp = Xp_full[:, pk]
    bp, ep, Ip = _ols(y, Xp)
    Vp = _cluster_vcov(Xp, ep, Ip, codes, Xp.shape[1])
    ssr_p = float(ep @ ep)

    # fixed effects (two-way: unit effects by demeaning, time effects by dummies)
    Xf = np.column_stack([Xw[:, ok], Dw]) if D.shape[1] else Xw[:, ok]
    yw = _demean(y, codes, ng)
    bf, ef, If = _ols(yw, Xf)
    kf = Xf.shape[1]
    dof_f = n - ng - kf
    s2e = float(ef @ ef) / dof_f
    Vf_conv = s2e * If
    Vf = _cluster_vcov(Xf, ef, If, codes, kf)
    ssr_f = float(ef @ ef)

    # F test for unit effects: FE (with the same regressors) against pooled OLS
    Xp2 = np.column_stack([np.ones(n), Xr[:, ok], D])
    _, ep2, _ = _ols(y, Xp2)
    ssr_p2 = float(ep2 @ ep2)
    F = ((ssr_p2 - ssr_f) / (ng - 1)) / (ssr_f / dof_f)
    out["ftest"] = dict(stat=_f(F, 2), df1=int(ng - 1), df2=int(dof_f), p=_f(_f_sf(F, ng - 1, dof_f), 4))

    # Breusch-Pagan LM test (Baltagi-Li version, valid for unbalanced panels), from pooled OLS residuals
    Ti = np.bincount(codes, minlength=ng).astype(float)
    se_i = np.bincount(codes, ep, ng)
    lm = (n ** 2 / (2 * float((Ti * (Ti - 1)).sum()))) * ((se_i ** 2).sum() / ssr_p - 1) ** 2
    out["lm"] = dict(stat=_f(lm, 2), p=_f(_chi2_sf(lm, 1), 4))

    # random effects (Swamy-Arora variance components; unbalanced: harmonic mean of T_i)
    Xb = np.column_stack([np.ones(ng), _gmeans(Xr[:, ok], codes, ng), _gmeans(D, codes, ng)])
    yb = _gmeans(y, codes, ng)
    kb = _independent_cols(Xb)
    _, eb, _ = _ols(yb, Xb[:, kb])
    s2b = float(eb @ eb) / max(ng - len(kb), 1)
    Th = ng / float((1 / Ti).sum())
    s2u = max(s2b - s2e / Th, 0.0)
    theta = 1 - np.sqrt(s2e / (Ti * s2u + s2e))
    th = theta[codes]
    Xre_raw = np.column_stack([np.ones(n), Xr[:, ok], D])
    mX = _gmeans(Xre_raw, codes, ng)[codes]
    Xre = Xre_raw - th[:, None] * mX
    yre = y - th * _gmeans(y, codes, ng)[codes]
    br, er, Ir = _ols(yre, Xre)
    s2re = float(er @ er) / (n - Xre.shape[1])
    Vr_conv = s2re * Ir
    Vr = _cluster_vcov(Xre, er, Ir, codes, Xre.shape[1])
    out["re"] = dict(sigma_u=_f(math.sqrt(s2u), 4), sigma_e=_f(math.sqrt(s2e), 4),
                     rho=_f(s2u / (s2u + s2e), 3), theta_mean=_f(float(theta.mean()), 3))

    # conventional Hausman test on the regressors common to FE and RE
    kk = len(ok)
    d = bf[:kk] - br[1:1 + kk]
    Vd = Vf_conv[:kk, :kk] - Vr_conv[1:1 + kk, 1:1 + kk]
    try:
        w, U = np.linalg.eigh((Vd + Vd.T) / 2)
        pos = w > 1e-12 * max(1, abs(w).max())
        H = float(d @ (U[:, pos] @ np.diag(1 / w[pos]) @ U[:, pos].T) @ d)
        hdf = int(pos.sum())
        out["hausman"] = dict(stat=_f(H, 2), df=hdf, p=_f(_chi2_sf(H, hdf), 4), pd=bool(pos.all()))
    except Exception:
        out["hausman"] = None

    # Mundlak / correlated random effects: pooled OLS on X, unit means of X and time dummies, cluster-robust Wald test
    Xm = np.column_stack([np.ones(n), Xr[:, ok], _gmeans(Xr[:, ok], codes, ng)[codes], D])
    bm, em, Im = _ols(y, Xm)
    Vm = _cluster_vcov(Xm, em, Im, codes, Xm.shape[1])
    sl = slice(1 + kk, 1 + 2 * kk)
    gm = bm[sl]
    W = float(gm @ np.linalg.pinv(Vm[sl, sl]) @ gm)
    out["mundlak"] = dict(stat=_f(W, 2), df=kk, p=_f(_chi2_sf(W, kk), 4))

    # coefficient table (cluster-robust standard errors by entity)
    rows = []
    for j, x in enumerate(xs):
        r = dict(var=x)
        if x in px:
            i = 1 + px.index(x)
            r["pols"], r["pols_se"] = _f(bp[i]), _f(math.sqrt(max(Vp[i, i], 0)))
        if x in fx:
            i = fx.index(x)
            r["re"], r["re_se"] = _f(br[1 + i]), _f(math.sqrt(max(Vr[1 + i, 1 + i], 0)))
            r["fe"], r["fe_se"] = _f(bf[i]), _f(math.sqrt(max(Vf[i, i], 0)))
        rows.append(r)
    out["coefs"] = rows
    out["within_r2"] = _f(1 - ssr_f / float(yw @ yw), 3)

    fp, lp, mp = out["ftest"]["p"], out["lm"]["p"], out["mundlak"]["p"]
    if fp >= 0.05 and lp >= 0.05:
        model = "POLS"
    elif mp < 0.05:
        model = "FE"
    elif lp < 0.05:
        model = "RE"
    else:
        model = "FE"
    out["static_model"] = model
    return out


# ---------------------------------------------------------------- long-panel tests

def _wide(df, ent, v):
    return df.pivot(index="_ti", columns=ent, values=v).sort_index()


def _cd(W):
    """Pesaran (2004/2015) CD test on a T x N wide array (NaN allowed). Returns CD, p, mean |rho|."""
    A = np.asarray(W, float)
    N = A.shape[1]
    s, sabs, npairs = 0.0, 0.0, 0
    M = ~np.isnan(A)
    for i in range(N - 1):
        for j in range(i + 1, N):
            m = M[:, i] & M[:, j]
            Tij = int(m.sum())
            if Tij < 3:
                continue
            a, b = A[m, i], A[m, j]
            a, b = a - a.mean(), b - b.mean()
            den = math.sqrt(float(a @ a) * float(b @ b))
            if den == 0:
                continue
            r = float(a @ b) / den
            s += math.sqrt(Tij) * r
            sabs += abs(r)
            npairs += 1
    if npairs == 0:
        return None, None, None
    cd = math.sqrt(2.0 / (N * (N - 1))) * s
    return cd, 2 * (1 - _ncdf(abs(cd))), sabs / npairs


def _fisher_adf(W, maxlag):
    from statsmodels.tsa.stattools import adfuller
    ps = []
    for c in W.columns:
        x = W[c].dropna().to_numpy(float)
        if len(x) < 8 or np.ptp(x) == 0:
            continue
        ml = max(0, min(maxlag, len(x) // 3 - 2))
        try:
            ps.append(adfuller(x, maxlag=ml, autolag="AIC" if ml > 0 else None, regression="c")[1])
        except Exception:
            pass
    if len(ps) < 2:
        return None
    ps = np.clip(np.array(ps), 1e-12, 1 - 1e-12)
    N = len(ps)
    P = -2 * np.log(ps).sum()
    Z = sum(_ninv(p) for p in ps) / math.sqrt(N)
    return dict(P=_f(P, 2), P_p=_f(_chi2_sf(P, 2 * N), 4), Z=_f(Z, 3), Z_p=_f(_ncdf(Z), 4), n=int(N))


def _cips(W, p=1):
    """Pesaran (2007) CIPS with an intercept and p lags on a T x N wide array (NaN allowed)."""
    A = np.asarray(W, float)
    yb = np.nanmean(A, axis=1)
    dA, dyb = np.diff(A, axis=0), np.diff(yb)
    T = A.shape[0]
    ts = []
    for i in range(A.shape[1]):
        rows_y, rows_X = [], []
        for t in range(p + 1, T):             # dy at time t uses index t-1 in the diff arrays
            k = t - 1
            vals = [dA[k, i], A[t - 1, i], yb[t - 1], dyb[k]] + [dyb[k - j] for j in range(1, p + 1)] + [dA[k - j, i] for j in range(1, p + 1)]
            if np.all(np.isfinite(vals)):
                rows_y.append(vals[0]); rows_X.append([1.0] + vals[1:])
        L, kx = len(rows_y), 4 + 2 * p
        if L < kx + 5:
            continue
        X, yv = np.array(rows_X), np.array(rows_y)
        b, e, I = _ols(yv, X)
        s2 = float(e @ e) / (L - kx)
        se = math.sqrt(s2 * I[1, 1])
        if se > 0:
            ts.append(b[1] / se)
    if len(ts) < 2:
        return None
    stat = float(np.mean(ts))
    Nn, Tn = len(ts), int(np.isfinite(A).any(axis=1).sum())
    return dict(stat=_f(stat, 3), p=_f(_cips_p(stat, Nn, Tn), 4), n=Nn, T=Tn)


def _cips_p(stat, N, T):
    Q = CIPS_Q
    if not Q:
        return None
    Ns, Ts, probs, q = np.array(Q["N"], float), np.array(Q["T"], float), np.array(Q["probs"]), np.array(Q["q"])
    def interp_axis(vals, x):
        x = min(max(x, vals[0]), vals[-1])
        j = int(np.searchsorted(vals, x))
        j = min(max(j, 1), len(vals) - 1)
        w = (math.log(x) - math.log(vals[j - 1])) / (math.log(vals[j]) - math.log(vals[j - 1]))
        return j, w
    i, wi = interp_axis(Ns, N)
    j, wj = interp_axis(Ts, T)
    qq = ((1 - wi) * ((1 - wj) * q[i - 1, j - 1] + wj * q[i - 1, j]) + wi * ((1 - wj) * q[i, j - 1] + wj * q[i, j]))
    if stat <= qq[0]:
        return float(probs[0]) / 2
    if stat >= qq[-1]:
        return min(1.0, float(probs[-1]) + (1 - probs[-1]) / 2)
    return float(np.interp(stat, qq, probs))


def _py_delta(df, ent, dep, xs):
    """Pesaran-Yamagata (2008) slope-homogeneity test, model y_it = a_i + x_it' b_i + e_it."""
    k = len(xs)
    units = []
    for _, g in df.groupby(ent, sort=False):
        if len(g) <= k + 2:
            continue
        y = g[dep].to_numpy(float); X = g[xs].to_numpy(float)
        y = y - y.mean(); X = X - X.mean(axis=0)
        if np.linalg.matrix_rank(X) < k:
            continue
        units.append((y, X))
    N = len(units)
    if N < 5:
        return None
    XX = sum(X.T @ X for _, X in units); Xy = sum(X.T @ y for y, X in units)
    bfe = np.linalg.solve(XX, Xy)
    s2 = [float((y - X @ bfe) @ (y - X @ bfe)) / (len(y) - 1) for y, X in units]
    A = sum(X.T @ X / s for (y, X), s in zip(units, s2)); c = sum(X.T @ y / s for (y, X), s in zip(units, s2))
    bw = np.linalg.solve(A, c)
    S = 0.0
    for (y, X), s in zip(units, s2):
        bi = np.linalg.solve(X.T @ X, X.T @ y)
        d = bi - bw
        S += float(d @ (X.T @ X) @ d) / s
    T = float(np.mean([len(y) for y, _ in units]))
    dl = math.sqrt(N) * (S / N - k) / math.sqrt(2 * k)
    da = math.sqrt(N) * (S / N - k) / math.sqrt(2 * k * (T - k - 1) / (T + 1))
    return dict(delta=_f(dl, 3), delta_p=_f(1 - _ncdf(dl), 4), delta_adj=_f(da, 3), delta_adj_p=_f(1 - _ncdf(da), 4), n=N)


def _lrcov(W, bw):
    T = W.shape[0]
    G = W.T @ W / T
    for j in range(1, bw + 1):
        Gj = W[j:].T @ W[:-j] / T
        G = G + (1 - j / (bw + 1)) * (Gj + Gj.T)
    return G


def _kao(df, ent, dep, xs, p=1):
    """Kao (1999) residual-based ADF test of no cointegration (homogeneous slopes, unit intercepts)."""
    codes, uniq = pd.factorize(df[ent]); ng = len(uniq)
    y = df[dep].to_numpy(float); X = df[xs].to_numpy(float)
    b, _, _ = _ols(_demean(y, codes, ng), _demean(X, codes, ng))
    df = df.assign(_e=_demean(y - X @ b, codes, ng))
    Sig, Om, rows_y, rows_X = [], [], [], []
    for _, g in df.groupby(ent, sort=False):
        e = g["_e"].to_numpy(); Xi = g[xs].to_numpy(float)
        if len(e) < p + 5:
            continue
        de = np.diff(e)
        for t in range(p + 1, len(e)):
            rows_y.append(e[t]); rows_X.append([e[t - 1]] + [de[t - 1 - j] for j in range(1, p + 1)])
        Wm = np.column_stack([e[1:], np.diff(Xi, axis=0)])
        Wm = Wm - Wm.mean(axis=0)
        Sig.append(Wm.T @ Wm / len(Wm)); Om.append(_lrcov(Wm, max(1, int(4 * (len(Wm) / 100) ** (2 / 9)))))
    if len(Sig) < 5:
        return None
    N = len(Sig)
    Sg, O = np.mean(Sig, axis=0), np.mean(Om, axis=0)
    sv2 = Sg[0, 0] - Sg[0, 1:] @ np.linalg.pinv(Sg[1:, 1:]) @ Sg[1:, 0]
    s0v2 = O[0, 0] - O[0, 1:] @ np.linalg.pinv(O[1:, 1:]) @ O[1:, 0]
    if sv2 <= 0 or s0v2 <= 0:
        return None
    Xa, ya = np.array(rows_X), np.array(rows_y)
    bb, ee, I = _ols(ya, Xa)
    s2 = float(ee @ ee) / (len(ya) - Xa.shape[1])
    t_adf = (bb[0] - 1) / math.sqrt(s2 * I[0, 0])
    sv, s0v = math.sqrt(sv2), math.sqrt(s0v2)
    stat = (t_adf + math.sqrt(6 * N) * sv / (2 * s0v)) / math.sqrt(s0v2 / (2 * sv2) + 3 * sv2 / (10 * s0v2))
    return dict(stat=_f(stat, 3), p=_f(_ncdf(stat), 4), n=N)


def _fisher_eg(df, ent, dep, xs):
    """Maddala-Wu (1999) Fisher combination of unit-by-unit Engle-Granger tests (heterogeneous slopes)."""
    from statsmodels.tsa.stattools import coint
    ps = []
    for _, g in df.groupby(ent, sort=False):
        if len(g) < 12 + 2 * len(xs):
            continue
        try:
            ps.append(coint(g[dep].to_numpy(float), g[xs].to_numpy(float), trend="c", maxlag=1, autolag=None)[1])
        except Exception:
            pass
    if len(ps) < 5:
        return None
    ps = np.clip(np.array(ps), 1e-12, 1 - 1e-12)
    N = len(ps)
    P = -2 * np.log(ps).sum(); Z = sum(_ninv(p) for p in ps) / math.sqrt(N)
    return dict(P=_f(P, 2), P_p=_f(_chi2_sf(P, 2 * N), 4), Z=_f(Z, 3), Z_p=_f(_ncdf(Z), 4), n=int(N))


def _long_panel(df, ent, dep, xs, T):
    out = {}
    V = [dep] + xs
    maxlag = max(1, min(4, int(12 * (T / 100) ** 0.25)))
    # cross-sectional dependence of each variable and of the fixed-effects residuals
    cds = []
    for v in V:
        cd, p, ar = _cd(_wide(df, ent, v).to_numpy())
        cds.append(dict(var=v, cd=_f(cd, 2), p=_f(p, 4), absr=_f(ar, 3)))
    out["cd"] = cds
    codes, uniq = pd.factorize(df[ent]); ng = len(uniq)
    yw = _demean(df[dep].to_numpy(float), codes, ng); Xw = _demean(df[xs].to_numpy(float), codes, ng)
    b, e, I = _ols(yw, Xw)
    cd, p, ar = _cd(df.assign(_r=e).pivot(index="_ti", columns=ent, values="_r").to_numpy())
    out["cd_resid"] = dict(cd=_f(cd, 2), p=_f(p, 4), absr=_f(ar, 3))
    out["slope"] = _py_delta(df, ent, dep, xs)

    # panel unit-root tests on levels, first and (if needed) second differences
    ur, orders = [], []
    for v, c in zip(V, cds):
        W = _wide(df, ent, v)
        use = "CIPS" if (c["p"] is not None and c["p"] < 0.05) else "Fisher"
        res = {"var": v, "test": use}
        order = None
        for d, lab in ((0, "lev"), (1, "d1"), (2, "d2")):
            Wd = W if d == 0 else W.diff(d) if d == 1 else W.diff().diff()
            fi, ci = _fisher_adf(Wd, maxlag), _cips(Wd.to_numpy())
            res[lab] = dict(fisher=fi, cips=ci)
            pv = (ci or {}).get("p") if use == "CIPS" else (fi or {}).get("Z_p")
            if pv is None:
                pv = (fi or {}).get("Z_p") if use == "CIPS" else (ci or {}).get("p")
            res[lab]["p"] = pv
            if pv is not None and pv < 0.05:
                order = d
                break
        res["order"] = 3 if order is None else order
        orders.append(res["order"])
        ur.append(res)
    out["unitroot"] = ur
    out["orders"] = orders

    cdr = out["cd_resid"]["p"] is not None and out["cd_resid"]["p"] < 0.05
    het = out["slope"] is not None and out["slope"]["delta_adj_p"] is not None and out["slope"]["delta_adj_p"] < 0.05
    out["cd_flag"], out["het_flag"] = bool(cdr), bool(het)
    if max(orders) >= 2:
        model = "I2"
    elif all(o == 0 for o in orders):
        model = "MG" if het else "FEDK"
    else:
        if all(o == 1 for o in orders):
            k1, f1 = _kao(df, ent, dep, xs), _fisher_eg(df, ent, dep, xs)
            used = "fisher" if het else "kao"
            pv = (f1 or {}).get("Z_p") if used == "fisher" else (k1 or {}).get("p")
            if pv is None:
                used = "kao" if used == "fisher" else "fisher"
                pv = (f1 or {}).get("Z_p") if used == "fisher" else (k1 or {}).get("p")
            coint = pv is not None and pv < 0.05
            out["coint"] = dict(kao=k1, fisher=f1, used=used, p=pv, coint=bool(coint))
            if not coint:
                model = "DIFF"
            else:
                model = "CSARDL" if cdr else "PMG"
        else:
            model = "CSARDL" if cdr else "PMG"
    out["model"] = model
    return out


# ---------------------------------------------------------------- main entry point

def analyze(csv_text, max_vars=6):
    warnings.simplefilter("ignore")
    rd, err = _read(csv_text, max_vars)
    if err:
        return json.dumps({"error": err})
    df, ent, tim, V, periods = rd["df"], rd["ent"], rd["tim"], rd["vars"], rd["periods"]
    dep, xs = V[0], V[1:]
    Ti = df.groupby(ent).size()
    N, Tn, n = int(len(Ti)), int(len(periods)), int(len(df))
    if N < 5:
        return json.dumps({"error": f"Only {N} entities with at least two periods were found. A panel needs several entities; with one or two, use the Time-Series Guide."})
    if n < 30:
        return json.dumps({"error": f"Only {n} complete entity–period rows were found. These tests need at least 30."})
    Tmed = float(Ti.median())
    out = dict(n=n, N=N, T=Tn, Tmin=int(Ti.min()), Tmax=int(Ti.max()), Tmed=_f(Tmed, 1), balanced=bool(Ti.min() == Ti.max() == Tn),
               entity=ent, time=tim, first=str(periods[0]), last=str(periods[-1]), vars=V, dep=dep, skipped=rd["skipped"],
               branch="short" if Tmed < LONG_T else ("long" if N >= FEW_N else "few"), long_t=LONG_T, gmm_n=GMM_N, few_n=FEW_N,
               n_large=bool(N >= GMM_N))

    # charts: cross-sectional mean over time with the 10th-90th percentile band, and the same for first differences
    g = df.groupby("_ti")
    def band(s):
        q = s.groupby(df["_ti"]).quantile([0.1, 0.9]).unstack()
        return dict(mean=[_f(v, 5) for v in s.groupby(df["_ti"]).mean().values], lo=[_f(v, 5) for v in q[0.1].values], hi=[_f(v, 5) for v in q[0.9].values])
    out["series"] = {v: band(df[v]) for v in V}
    dfd = df.copy()
    for v in V:
        dfd[v] = df.groupby(ent)[v].diff()
    out["dseries"] = {v: [_f(x, 5) for x in dfd.groupby("_ti")[v].mean().values[1:]] for v in V}
    # a few individual units for the spaghetti chart
    top = Ti.sort_values(ascending=False).index[:6]
    out["units"] = {str(u): [_f(x, 5) for x in df[df[ent] == u].set_index("_ti")[dep].reindex(range(Tn)).values] for u in top}

    summ = []
    codes, uniq = pd.factorize(df[ent])
    for v in V:
        x = df[v].to_numpy(float)
        m = _gmeans(x, codes, len(uniq))
        w = x - m[codes] + x.mean()
        summ.append(dict(var=v, mean=_f(x.mean()), sd=_f(x.std(ddof=1)), between=_f(m.std(ddof=1)), within=_f(w.std(ddof=1)), min=_f(x.min()), max=_f(x.max())))
    out["summary"] = summ
    out["corr"] = [[_f(c, 3) for c in row] for row in df[V].corr().values]

    if out["branch"] == "short":
        r = _short_panel(df, ent, dep, xs)
        if "error" in r:
            return json.dumps({"error": r["error"]})
        out["static"] = r
        out["model"] = r["static_model"]
    elif out["branch"] == "long":
        out.update(_long_panel(df, ent, dep, xs, Tn))
    else:
        out["model"] = "TS"
    return json.dumps(out)


# ---------------------------------------------------------------- public functions for use in your own scripts
# Long-format data must be sorted by entity and time. A "wide" frame has one row per period and one column per unit,
# for example wide = df.pivot(index="year", columns="country", values="log_gdp").

def cd_test(wide):
    """Pesaran CD test. Returns (CD statistic, p-value, mean absolute correlation)."""
    return _cd(wide.to_numpy(float))


def cips_test(wide, p=1):
    """Pesaran (2007) CIPS unit-root test with an intercept and p lags (p-value from simulated quantiles, p=1)."""
    return _cips(wide.to_numpy(float), p)


def fisher_adf_test(wide, maxlag=4):
    """Maddala-Wu / Choi Fisher-type ADF unit-root test (combines unit-by-unit ADF p-values)."""
    return _fisher_adf(wide, maxlag)


def slope_test(df, entity, y, xs):
    """Pesaran-Yamagata (2008) slope-homogeneity test (delta and adjusted delta, upper-tail p-values)."""
    return _py_delta(df.sort_values(entity, kind="stable"), entity, y, list(xs))


def kao_test(df, entity, y, xs):
    """Kao (1999) ADF test of no cointegration."""
    return _kao(df, entity, y, list(xs))


def fisher_eg_test(df, entity, y, xs):
    """Maddala-Wu Fisher combination of unit-by-unit Engle-Granger cointegration tests."""
    return _fisher_eg(df, entity, y, list(xs))
