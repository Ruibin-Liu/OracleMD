# 004 · 净电荷微扰有限尺寸校正(net_charge)

## 为何脆弱
带净电荷的周期体系必须加中和背景与有限尺寸校正;校正常数**依赖约定**
(锡箔 vs 真空边界、盒型、求和顺序),结论简短、推导繁琐。典型 LLM 失效:
背下 −k·q²·ξ/L³ 的某个形式,丢掉「仅立方盒、仅锡箔」的前提——错的适用
条件不崩,只让 ΔG 偏 ~0.5 kcal/mol,恰在自由能计算的噪声地板之下、结论敏感区。

## 已知失效模式
- L 指数错(边界/盒型约定混用);
- 符号错(校正应降低 charged 体系能量);
- 非立方盒沿用立方公式。

## oracle
立方盒 + 锡箔:ΔE = −k·q²·α/L²。(DEMO 桩值;真实公式接文献后替换,指数与常数必须带出处)

## mutation
file: kernels.py
domain: net_charge
desc: L 指数从 2 改为 3(立方/锡箔约定错套为另一约定的形式)
before: |
    return -k * q_net * q_net * alpha / (box_L * box_L)
after: |
    return -k * q_net * q_net * alpha / (box_L * box_L * box_L)

## mutation
file: kernels.py
domain: net_charge
desc: 丢符号(有限尺寸校正应为负)
before: |
    return -k * q_net * q_net * alpha / (box_L * box_L)
after: |
    return +k * q_net * q_net * alpha / (box_L * box_L)

## enforcement
- test: 精确值断言(带约定前提);立方盒锡箔参考算例;
- hardcode: 非立方盒输入直接拒绝(适用条件检查),不许静默套公式。
