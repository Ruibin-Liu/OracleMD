# 物理规格(黑盒红队的唯一真值来源)

> 黑盒红队**只允许**依据本文件 + `interfaces.md` + `minefield/` 写测试。
> 本文件由蓝队/harness 维护,每次接口变更必须同步。规格与实现不一致 = bug,
> 由黑盒测试暴露,这正是本机制的目的。

## 维护规则

- 每个公开函数:给出**可判定的**输入→期望(精确公式或参考值,不接受"应合理"这类措辞);
- 每条近似:写明**适用条件**(域、参数范围、前提),适用条件本身进入测试;
- 无法给出 oracle 的项:不要硬写,显式标记 `NO-ORACLE`,等待黑盒提 gap 或人工补推导。

## 演示桩规格(替换真实规格时删除本节)

### shake_dof(n_atoms, n_constraints) -> int
非线性分子:自由度 = 3N − 6 − n_constraints。
适用条件:N ≥ 2,非线性构型;线性分子(如 HCN)为 3N − 5 − n_constraints(本演示不实现)。

### virial_pressure(volume, kinetic, virial) -> float
P = (2/3)·(K − W)/V。W 为位力(含键合与非键合贡献之和),V > 0, K ≥ 0。

### pme_self_exclusion(net_q2, alpha) -> float
自能修正 = −α·Σq²/√π(net_q2 = Σq²)。恒为负;随 α 线性。

### net_charge_finite_size_correction(q_net, box_L, alpha) -> float
立方盒 + 锡箔边界约定:ΔE = −k·q²·α/L²,k = 1/(4πε₀)。
适用条件:立方盒、锡箔(导体)边界、点电荷;其他盒型/边界指数不同 → `NO-ORACLE` 之外人工推导。
