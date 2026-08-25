# 003 · PME 自能/排除项修正(pme_exclusions)

## 为何脆弱
PME 总能 = 直空间 + 倒空间 + 修正项,修正项全是小量、符号密集:
自能 −α·Σq²/√π 恒负;排除对必须从倒空间贡献中剔除(否则分子内 1-2、1-3 对被
重复计入);净电荷体系还要加中和背景项。LLM 记得住"有个修正",记不住哪一项
在哪个求和里该出现/剔除。

## 已知失效模式
- 自能符号翻转;
- 排除项漏剔或重复剔除;
- 净电荷中和项缺失(与 004 耦合)。

## oracle
E_self = −α·Σq²/√π。(DEMO 桩值;排除项与背景项接入真实代码后补参考算例)

## mutation
file: kernels.py
domain: pme_exclusions
desc: 自能项符号翻转(− → +)
before: |
    self_term = -alpha * net_q2 / math.sqrt(math.pi)
after: |
    self_term = alpha * net_q2 / math.sqrt(math.pi)

## enforcement
- test: 自能恒负、随 α 线性、q=0 时为 0;
- golden: 排除对能量(分子内拓扑)参考算例;
- hardcode: 修正项清单(自能/排除/背景)作为不变量断言存在。
