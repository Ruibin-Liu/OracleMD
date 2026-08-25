# 001 · 约束算法自由度计数(shake_settle)

## 为何脆弱
推导繁琐、结论简短:结论(`3N−6−约束数`)人人背得出,但**前提**常被丢掉——
线性分子是 3N−5;SETTLE 只对 3 位点刚体水成立;SHAKE 的约束数与 RATTLE 的
速度约束数在半步/整步格式下不同。错的自由度数不会让代码崩,只会让温度、
压强、ΔG 全部带系统性偏移。

## 已知失效模式
- 线性/非线性分子的 −6/−5 混用;
- 约束矩阵秩 ≠ 声明的约束数(冗余约束);
- 对 >3 位点刚体套 SETTLE。

## oracle
非线性:DOF = 3N − 6 − n_constraints;线性:3N − 5 − n_constraints。(DEMO 桩值)

## mutation
file: kernels.py
domain: shake_settle
desc: 非线性平移/转动扣除误用 -5(线性分支结论错放到非线性)
before: |
    free = 3 * n_atoms - 6          # non-linear
after: |
    free = 3 * n_atoms - 5          # non-linear

## enforcement
- test: 逐构型精确断言 DOF(非线性 N=3 无约束 → 3;线性 HCN N=3 → 4);
- hardcode: 「SETTLE 仅 3 位点」写成接口前置条件检查;
- golden: 约束数=秩 的参考算例(接真实代码后)。
