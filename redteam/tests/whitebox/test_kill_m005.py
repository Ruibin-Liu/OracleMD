"""演示用「模拟白盒红队」击杀测试。

真实工作流中由 harness 从白盒 delegate 报告落盘(白盒见过实现与存活变异体,
迭代喂的是击杀结果这一事实)。此文件演示:针对存活种子 m005(L 指数变异)
写出 spec 精确断言,把它杀死。
"""
import math

import kernels

K_COULOMB = 1.0 / (4.0 * math.pi * 8.8541878128e-12)


# kill: m004 (net_charge L-exponent mutant) — 弱基线测试放过的那个
def test_net_charge_exact_value():
    got = kernels.net_charge_finite_size_correction(2.0, 2.0, 0.25)
    want = -K_COULOMB * 4.0 * 0.25 / 4.0
    assert abs(got - want) / abs(want) < 1e-9
