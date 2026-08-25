"""蓝队基线测试(演示)。

故意混合强弱断言,展示弱断言让种子变异体存活的失效模式:
- domains 1-3:强断言(精确值)→ 杀死各自种子变异体;
- domain 4:弱断言(只查符号)→ 种子变异体存活 → 生成白盒任务。
"""
import math

import kernels


# --- shake_settle: strong ------------------------------------------------
def test_shake_dof_nonlinear_water():
    # 水分子 N=3, 刚体约束(SETTLE) 3 条: 3*3-6-3 = 0
    assert kernels.shake_dof(3, 3) == 0
    # 乙烷 N=8, 7 键约束: 3*8-6-7 = 11
    assert kernels.shake_dof(8, 7) == 11


# --- virial_pressure: strong ----------------------------------------------
def test_virial_pressure_ideal_gas_limit():
    # W=0 → P = 2K/3V
    assert abs(kernels.virial_pressure(2.0, 3.0, 0.0) - 1.0) < 1e-12


def test_virial_pressure_known_value():
    # (2/3)*(5-2)/4 = 0.5
    assert abs(kernels.virial_pressure(4.0, 5.0, 2.0) - 0.5) < 1e-12


# --- pme_exclusions: strong -------------------------------------------------
def test_pme_self_energy_sign_and_scaling():
    val = kernels.pme_self_exclusion(2.0, 0.35)
    expected = -0.35 * 2.0 / math.sqrt(math.pi)
    assert abs(val - expected) < 1e-15
    # 恒负、随 alpha 线性
    assert kernels.pme_self_exclusion(1.0, 0.2) < 0
    assert abs(
        kernels.pme_self_exclusion(1.0, 0.4)
        - 2 * kernels.pme_self_exclusion(1.0, 0.2)
    ) < 1e-15


# --- net_charge: WEAK (demonstrates surviving seeded mutant) ---------------
def test_net_charge_correction_is_negative():
    val = kernels.net_charge_finite_size_correction(1.0, 3.0, 0.3)
    assert val < 0   # 弱断言:符号对就通过,L 指数错了也测不出
