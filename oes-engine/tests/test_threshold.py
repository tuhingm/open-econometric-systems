"""Threshold (TAR / SETAR) and smooth-transition (LSTAR / ESTAR) regressions."""
import numpy as np
import pandas as pd
import pytest

from oes_engine import smooth, tar


def setar(T, seed, thr=0.5, jump=1.0):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = (0.8 + 0.6 * y[t - 1] if y[t - 1] <= thr else 0.8 - jump - 0.3 * y[t - 1]) + 0.7 * rng.normal()
    return pd.DataFrame({"y": y})


def table(res, title):
    return next(t for t in res["tables"] if t["title"].startswith(title))


def test_scan_matches_brute_force():
    rng = np.random.default_rng(0)
    T = 80
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    q = rng.normal(size=T)
    y = X @ [1.0, 2.0] + np.where(q > 0.3, 1.5, 0.0) + rng.normal(size=T)
    order = np.argsort(q)
    sq = q[order]
    h = 12
    ks = tar._valid_splits(sq, h)
    S, _ = tar._scan(y, X, order, sq, ks)
    for k, s in zip(ks[::7], S[::7]):
        lo, hi = order[:k], order[k:]
        brute = 0.0
        for idx in (lo, hi):
            e = y[idx] - X[idx] @ np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
            brute += float(e @ e)
        assert s == pytest.approx(brute, rel=1e-8)


def test_setar_recovers_threshold_and_delay():
    r = tar.tar(setar(400, 5), dict(y="y", ar_lags=2, search_delay="yes", max_delay=3, boot=99))
    est = table(r, "Estimated thresholds")["rows"][0]
    assert abs(est[1] - 0.5) < 0.25
    summ = {s["label"]: s["value"] for s in r["summary"]}
    assert summ["Delay"] == 1
    assert summ["Bootstrap p (linearity)"] <= 0.02


def test_regime_coefficients_equal_ols_on_each_side():
    df = setar(300, 7)
    r = tar.tar(df, dict(y="y", ar_lags=1, delay=1, boot=0, max_thresholds=1))
    g = r["summary"][-1]["value"]
    y = df["y"].to_numpy()
    yy, ylag = y[1:], y[:-1]
    for side, col in ((ylag <= g, 1), (ylag > g, 2)):
        b = np.linalg.lstsq(np.column_stack([np.ones(side.sum()), ylag[side]]), yy[side], rcond=None)[0]
        row_c, row_a = table(r, "Coefficients")["rows"][0], table(r, "Coefficients")["rows"][1]
        assert float(row_c[col].split()[0]) == pytest.approx(b[0], abs=1e-3)
        assert float(row_a[col].split()[0]) == pytest.approx(b[1], abs=1e-3)


def test_linear_series_rarely_rejected():
    rej = 0
    for s in range(20):
        rng = np.random.default_rng(100 + s)
        y = np.zeros(200)
        for t in range(1, 200):
            y[t] = 0.5 + 0.6 * y[t - 1] + rng.normal()
        r = tar.tar(pd.DataFrame({"y": y}), dict(y="y", ar_lags=1, delay=1, boot=99, max_thresholds=1))
        p = {s_["label"]: s_["value"] for s_ in r["summary"]}["Bootstrap p (linearity)"]
        rej += p < 0.05
    assert rej <= 4


def test_second_threshold_found_with_two_jumps():
    rng = np.random.default_rng(2)
    T = 500
    q = rng.normal(size=T)
    x = rng.normal(size=T)
    y = np.where(q < -0.5, 1.0 * x, np.where(q < 0.6, -0.5 * x, 1.5 * x)) + 0.5 * rng.normal(size=T)
    d = pd.DataFrame({"y": y, "x": x, "q": q})
    r = tar.tar(d, dict(y="y", x=["x"], ar_lags=0, threshold="q", delay=0, boot=99))
    summ = {s["label"]: s["value"] for s in r["summary"]}
    assert summ["Thresholds found"] == 2
    assert abs(summ["Threshold 1"] + 0.5) < 0.3 and abs(summ["Threshold 2"] - 0.6) < 0.3


def test_transition_function_derivatives():
    s = np.linspace(-2, 3, 25)
    for kind in ("logistic", "exponential"):
        g, c, sd = 2.3, 0.4, 1.1
        dg, dc = smooth._dG(s, g, c, sd, kind)
        h = 1e-6
        assert np.allclose(dg, (smooth._G(s, g + h, c, sd, kind) - smooth._G(s, g - h, c, sd, kind)) / (2 * h), atol=1e-6)
        assert np.allclose(dc, (smooth._G(s, g, c + h, sd, kind) - smooth._G(s, g, c - h, sd, kind)) / (2 * h), atol=1e-6)


def lstar(T, seed, gamma=4.0, c=0.2, kind="logistic"):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=T)
    y = np.zeros(T)
    for t in range(1, T):
        G = smooth._G(s[t - 1], gamma, c, 1.0, kind)
        y[t] = (0.5 + 0.2 * y[t - 1]) * (1 - G) + (-0.5 + 0.7 * y[t - 1]) * G + 0.5 * rng.normal()
    return pd.DataFrame({"y": y, "s": s})


def test_lstar_recovered():
    r = smooth.smooth(lstar(500, 3), dict(y="y", ar_lags=1, transition="s", delay=1))
    summ = {s["label"]: s["value"] for s in r["summary"]}
    assert summ["Transition function"] == "logistic"
    assert 2 < summ["gamma"] < 9 and abs(summ["c"] - 0.2) < 0.25
    assert summ["Linearity p"] < 0.001
    imp = table(r, "Coefficients in the two extreme regimes")["rows"]
    assert float(imp[0][2].split()[0]) < 0 < float(imp[0][1].split()[0])


def test_estar_chosen_for_exponential_dgp():
    r = smooth.smooth(lstar(600, 4, gamma=3.0, c=0.0, kind="exponential"), dict(y="y", ar_lags=1, transition="s", delay=1))
    assert {s["label"]: s["value"] for s in r["summary"]}["Transition function"] == "exponential"


def test_linearity_not_rejected_for_linear_dgp():
    rej = 0
    for s in range(20):
        rng = np.random.default_rng(300 + s)
        T = 250
        z = rng.normal(size=T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.3 + 0.5 * y[t - 1] + 0.4 * z[t] + rng.normal()
        r = smooth.smooth(pd.DataFrame({"y": y, "z": z}), dict(y="y", x=["z"], ar_lags=1, transition="z", delay=1))
        rej += {s_["label"]: s_["value"] for s_ in r["summary"]}["Linearity p"] < 0.05
    assert rej <= 4
