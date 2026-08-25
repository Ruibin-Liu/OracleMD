# 002 · 维里与压强(virial_pressure)

## 为何脆弱
P = (2/3)(K − W)/V 的每个符号都能单独错:2/3 写成 3/2、位力符号、
PME 倒空间位力漏项、约束位力(刚体约束对压强的贡献)漏项、键合/非键合重复计入。
错误量级 ~常数因子,MD 不崩,ΔG 偏移可观的 kcal/mol 量级。

## 已知失效模式
- 因子 2/3 ↔ 3/2;
- 倒空间位力缺失(Ewald 求和的 α 依赖项);
- 排除对在直空间与倒空间重复计入。

## oracle
P = (2/3)·(K − W)/V,W 含直空间+倒空间+约束位力之和。(DEMO 桩值仅含标量形式)

## mutation
file: kernels.py
domain: virial_pressure
desc: 压强前因子写反(2/3 → 3/2)
before: |
    return (2.0 / 3.0) * (kinetic - virial) / volume
after: |
    return (3.0 / 2.0) * (kinetic - virial) / volume

## enforcement
- test: 已知 (K,W,V) 精确值断言;理想气体极限 W=0 → P=2K/3V;
- golden: PME 体系压强参考算例(倒空间位力,接真实代码后)。
