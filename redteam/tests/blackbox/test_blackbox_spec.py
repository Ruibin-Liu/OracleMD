"""演示用「模拟黑盒红队」测试。

真实工作流中此文件由 harness 从黑盒 delegate 的报告落盘生成
(黑盒只见过 spec.md + interfaces.md + minefield/,从未见过实现)。
保留它是为了让 demo.sh 无 LLM 也能跑通全链路,并演示独立发现率度量。
"""
import math

import kernels

K_COULOMB = 1.0 / (4.0 * math.pi * 8.8541878128e-12)


# domain: net_charge | basis: spec.md 立方盒+锡箔 ΔE = -k q² α / L²
def test_net_charge_exact_cubic_tinfoil():
    got = kernels.net_charge_finite_size_correction(1.0, 2.0, 0.5)
    want = -K_COULOMB * 1.0 * 0.5 / (2.0 * 2.0)
    assert abs(got - want) / abs(want) < 1e-9


# domain: virial_pressure | basis: spec.md P=(2/3)(K−W)/V, 已知值断言
def test_virial_pressure_spec_value():
    assert abs(kernels.virial_pressure(3.0, 3.0, 0.0) - 2.0 / 3.0) < 1e-12
