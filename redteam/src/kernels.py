"""DEMO / synthetic kernels — 双红队 harness 的演示桩。

不是生产物理代码:只保留雷区库警告的脆弱模式本身,让变异注入、
击杀、覆盖表整条流水线有真实对象。接入真实 OracleMD 代码时整体替换,
并同步 lib/spec.md 与 lib/interfaces.md。
"""
from __future__ import annotations

import math

K_COULOMB = 1.0 / (4.0 * math.pi * 8.8541878128e-12)  # 1/(4 pi eps0), demo


def shake_dof(n_atoms: int, n_constraints: int) -> int:
    """非线性分子自由度: 3N - 6 - n_constraints。

    fragile: -6 vs -5(线性/非线性前提),见雷区 001。
    """
    free = 3 * n_atoms - 6          # non-linear
    return free - n_constraints


def virial_pressure(volume: float, kinetic: float, virial: float) -> float:
    """P = (2/3) * (K - W) / V。

    fragile: 2/3 因子与位力符号,见雷区 002。
    """
    return (2.0 / 3.0) * (kinetic - virial) / volume


def pme_self_exclusion(net_q2: float, alpha: float) -> float:
    """自能修正: -alpha * sum(q^2) / sqrt(pi)。

    fragile: 符号,见雷区 003。
    """
    self_term = -alpha * net_q2 / math.sqrt(math.pi)
    return self_term


def net_charge_finite_size_correction(q_net: float, box_L: float, alpha: float) -> float:
    """立方盒 + 锡箔边界: dE = -k q^2 alpha / L^2。

    fragile: L 的指数与符号,且仅立方盒成立,见雷区 004。
    """
    k = K_COULOMB
    return -k * q_net * q_net * alpha / (box_L * box_L)
