#!/usr/bin/env python3
"""Simulate the critical values used by oes_engine/breaks.py and write oes_engine/_bp_cv.py.

For every trimming fraction eps and number q of changing coefficients (the constant plus q - 1 regressors) we draw
T = 150 observations of pure noise, find the optimal partitions with 1..M breaks, and record sup-F(l) for l = 1..M,
UDmax = max sup-F(l), and the sup-F(1) statistic that the sequential test uses. The 5% critical values of the sequential
test sup-F(l+1 | l) are the (0.95)**(1/(l+1)) quantiles of the sup-F(1) distribution (Bai and Perron 1998: the test is the
maximum of l + 1 asymptotically independent sup-F(1) tests, one per regime).
Also simulates the quantiles of the argmax of W(s) - |s|/2 (two-sided Brownian motion), used for break-date intervals.
Usage: python tools/simulate_bp_cv.py [reps]   (about 20-40 minutes)
"""
import itertools, pathlib, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import importlib.util
# breaks.py imports _bp_cv: provide an empty stub so the module can be imported while the table is being built
stub = pathlib.Path(__file__).resolve().parents[1] / "oes_engine" / "_bp_cv.py"
if not stub.exists():
    stub.write_text("QMAX = 5\nCV = {}\nARGMAX = {}\n")
from oes_engine.breaks import ssr_table, partitions, sup_f, TRIMS

T = 150
Q = [1, 2, 3, 4, 5]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 700


def sim(args):
    q, eps, seed = args
    rng = np.random.default_rng(seed)
    h = int(np.floor(eps * T))
    M = TRIMS[eps]
    F, seq1 = [], []
    for _ in range(REPS):
        X = np.column_stack([np.ones(T)] + [rng.standard_normal(T) for _ in range(q - 1)])
        y = rng.standard_normal(T)
        S = ssr_table(y, X, h)
        ssr, _ = partitions(S, T, h, M)
        F.append(sup_f(ssr, T, q, M))
        seq1.append((ssr[0] - ssr[1]) / (ssr[1] / (T - 2 * q)))
    F, seq1 = np.array(F), np.array(seq1)
    supf = [float(np.quantile(F[:, l], 0.95)) for l in range(M)]
    ud = float(np.quantile(F.max(axis=1), 0.95))
    seq = [float(np.quantile(seq1, 0.95 ** (1 / (l + 1)))) for l in range(M)]
    return (q, eps), dict(supf=supf, udmax=ud, seq=seq)


def argmax_quantiles(reps=40000, half=60.0, dt=0.05, seed=1):
    rng = np.random.default_rng(seed)
    n = int(half / dt)
    out = []
    for _ in range(reps // 2000):
        right = np.cumsum(rng.standard_normal((2000, n)) * np.sqrt(dt) - 0.5 * dt, axis=1)
        left = np.cumsum(rng.standard_normal((2000, n)) * np.sqrt(dt) - 0.5 * dt, axis=1)
        m_r, m_l = right.max(axis=1), left.max(axis=1)
        a_r, a_l = (right.argmax(axis=1) + 1) * dt, (left.argmax(axis=1) + 1) * dt
        m_r, m_l = np.maximum(m_r, 0), np.maximum(m_l, 0)
        pos = np.where(m_r >= m_l, a_r, -a_l)
        pos = np.where((m_r == 0) & (m_l == 0), 0.0, pos)
        out.append(pos)
    a = np.abs(np.concatenate(out))
    return {0.90: float(np.quantile(a, 0.90)), 0.95: float(np.quantile(a, 0.95))}


if __name__ == "__main__":
    jobs = [(q, eps, 1000 * q + int(eps * 100)) for eps in (0.25, 0.20, 0.15, 0.10) for q in Q]
    with Pool(2) as p:
        res = dict(p.imap_unordered(sim, jobs))
    arg = argmax_quantiles()
    lines = ['"""Simulated critical values for the Bai-Perron tests (written by tools/simulate_bp_cv.py)."""', f"QMAX = {max(Q)}", "CV = {"]
    for k in sorted(res):
        lines.append(f"    {k!r}: {res[k]!r},")
    lines += ["}", f"ARGMAX = {arg!r}", ""]
    (pathlib.Path(__file__).resolve().parents[1] / "oes_engine" / "_bp_cv.py").write_text("\n".join(lines))
    print("done", arg)
