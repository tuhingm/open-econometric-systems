"""Shared helpers: reading data, validating column roles and building the result object.

Every estimator returns a plain dict that the browser renders without knowing the model:

    {"model": id, "title": str, "nobs": int,
     "summary": [{"label", "value"}],                  key statistics shown as tiles
     "tables":  [{"title", "columns", "rows", "note"}],  coefficient and test tables
     "charts":  [{"title", "x", "series": [{"name", "y"}], "xlabel", "ylabel", "kind"}],
     "notes":   [str]}                                 assumptions and warnings
"""
import io
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats


class OESError(Exception):
    """An error caused by the data or the specification, reported to the user as is."""


def f(x, nd=6):
    """Round for display; None for missing, NaN or infinite values."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return round(x, nd)


def read_csv(text):
    if not isinstance(text, str) or not text.strip():
        raise OESError("The dataset is empty.")
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as e:  # pragma: no cover - parser messages vary
        raise OESError(f"Could not read the file as CSV ({e}).")
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] < 2:
        raise OESError("The file needs at least two columns.")
    return df


def need(spec, key, label=None):
    v = spec.get(key)
    if v in (None, "", []):
        raise OESError(f"Choose the {label or key}.")
    return v


def as_list(v):
    if v in (None, ""):
        return []
    return [v] if isinstance(v, str) else [c for c in v if c not in (None, "")]


def check_columns(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise OESError("Column(s) not found in the data: " + ", ".join(miss))


def numeric(df, cols, keep=()):
    """Numeric copy of the columns in cols (plus untouched columns in keep), rows with missing values dropped."""
    cols = list(dict.fromkeys(cols))
    check_columns(df, cols + list(keep))
    d = df[list(keep) + [c for c in cols if c not in keep]].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=cols)
    if len(d) == 0:
        raise OESError("No complete numeric rows remain for the selected variables.")
    return d


def distinct(*groups):
    seen = []
    for g in groups:
        for c in as_list(g):
            if c in seen:
                raise OESError(f"The variable {c} is used in more than one role.")
            seen.append(c)


def pval(stat, dist="t", df=None):
    stat = abs(float(stat))
    if dist == "t" and df:
        return 2 * stats.t.sf(stat, df)
    return 2 * stats.norm.sf(stat)


def coef_table(names, b, se, title="Coefficient estimates", dist="t", df=None, level=0.95, note=None, p=None, stat=None):
    b, se = np.asarray(b, float), np.asarray(se, float)
    z = b / se if stat is None else np.asarray(stat, float)
    crit = stats.t.ppf(0.5 + level / 2, df) if (dist == "t" and df) else stats.norm.ppf(0.5 + level / 2)
    p = [pval(v, dist, df) for v in z] if p is None else list(p)
    lab = "t" if dist == "t" else "z"
    pct = int(round(level * 100))
    rows = [[str(n), f(bi), f(si), f(zi, 4), f(pi, 4), f(bi - crit * si), f(bi + crit * si)]
            for n, bi, si, zi, pi in zip(names, b, se, z, p)]
    return table(title, ["Variable", "Coef.", "Std. err.", lab, "p-value", f"{pct}% CI low", f"{pct}% CI high"], rows, note)


def table(title, columns, rows, note=None):
    clean = [[(f(v) if isinstance(v, (float, np.floating, np.integer)) and not isinstance(v, bool) else v) for v in r] for r in rows]
    t = {"title": title, "columns": list(columns), "rows": clean}
    if note:
        t["note"] = note
    return t


def test_row(name, statistic, p, extra="", nd=4):
    return [name, f(statistic, nd), f(p, 4), extra]


def tests_table(rows, title="Specification and diagnostic tests", note=None):
    return table(title, ["Test", "Statistic", "p-value", "Detail"], rows, note)


def stat(label, value, nd=4):
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return {"label": label, "value": int(value)}
    if isinstance(value, (float, np.floating)):
        return {"label": label, "value": f(value, nd)}
    return {"label": label, "value": value}


def chart(title, x, series, xlabel="", ylabel="", kind="line", band=None):
    out = {"title": title, "x": [v if isinstance(v, str) else f(v) for v in x],
           "series": [{"name": n, "y": [f(v) for v in y]} for n, y in series], "xlabel": xlabel, "ylabel": ylabel, "kind": kind}
    if band is not None:
        out["band"] = {"lo": [f(v) for v in band[0]], "hi": [f(v) for v in band[1]]}
    return out


def result(model, title, nobs, summary=(), tables=(), charts=(), notes=()):
    return {"model": model, "title": title, "nobs": int(nobs), "summary": list(summary), "tables": list(tables),
            "charts": list(charts), "notes": list(notes)}


def time_order(df, col):
    """Sort a data frame by a time column that may be numeric, a date or text."""
    t = pd.to_numeric(df[col], errors="coerce")
    if t.notna().all():
        return df.assign(_order=t).sort_values("_order").drop(columns="_order")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(df[col], errors="coerce")
    if d.notna().all():
        return df.assign(_order=d).sort_values("_order").drop(columns="_order")
    return df.sort_values(col)


def int_option(spec, key, default, lo, hi, label):
    v = spec.get(key, default)
    if v in (None, ""):
        v = default
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise OESError(f"{label} must be a whole number.")
    if not lo <= v <= hi:
        raise OESError(f"{label} must be between {lo} and {hi}.")
    return v


def float_option(spec, key, default, label):
    v = spec.get(key, default)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        raise OESError(f"{label} must be a number.")
