#!/usr/bin/env python3
"""M2 erfc 多项式近似拟合:erfc(x) = exp(-x^2) * W(x), W = 分段 Chebyshev 多项式。

目标:全区间 [0, 6.5] 相对误差 <= 1e-13(对 intrinsic double erfc),
给 A1 对齐(rel 1e-10)留 3 个量级余量。W(x) = erfc(x)*exp(x^2) 是整函数,
分段短区间收敛几何快;exp(-x^2) 与力公式第二项共享,一次计算两用。
采样权重 1/w(最小化相对误差);区间上界 6.5 覆盖生产 masking 语义下
x_max = alpha*(rc+skin) = 3.5*1.35 = 4.725(超出段截断到 6.5,
abs 误差 <= erfc(6.5) = 5.8e-20,不可见)。
"""
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import erfc as erfc_ref

PIECES = [(0.0, 1.5), (1.5, 3.0), (3.0, 6.5)]
TARGET = 1e-13


def fit_piece(lo, hi, deg, npts=4001):
    x = np.linspace(lo, hi, npts)
    w = erfc_ref(x) * np.exp(x * x)
    t = 2 * (x - lo) / (hi - lo) - 1
    V = C.chebvander(t, deg)
    A = V / w[:, None]
    coef, *_ = np.linalg.lstsq(A, np.ones(npts), rcond=None)
    w_fit = C.chebval(t, coef)
    rel = np.abs(w_fit - w) / np.abs(w)
    return C.cheb2poly(coef), rel.max()


if __name__ == "__main__":
    print("分段拟合 erfc(x) = exp(-x^2)*W(x),目标 max_rel <= 1e-13")
    chosen = {}
    for lo, hi in PIECES:
        print(f"\npiece [{lo}, {hi}]:")
        print(f"{'deg':>4} {'max_rel_err':>12}")
        for deg in (8, 10, 12, 14, 16, 18, 20):
            pc, mx = fit_piece(lo, hi, deg)
            print(f"{deg:>4} {mx:>12.2e}")
            if mx <= TARGET and (lo, hi) not in chosen:
                chosen[(lo, hi)] = (deg, pc, mx)
        if (lo, hi) not in chosen:
            raise SystemExit(f"piece [{lo},{hi}] 未达 {TARGET}")
    total = 0
    for (lo, hi), (deg, pc, mx) in chosen.items():
        print(f"\npiece [{lo},{hi}] deg={deg} max_rel={mx:.2e} ({len(pc)} coeffs)")
        total += len(pc)
        print(f"double ERFC_C_{lo:g}_{hi:g}[] = {{")
        for i in range(0, len(pc), 3):
            print("    " + ", ".join(f"{c:+.17e}" for c in pc[i:i + 3]) + ",")
        print("};")
    print(f"\ntotal coefficients: {total}")
