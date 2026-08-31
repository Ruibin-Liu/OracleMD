# Q-006 定标实验:邻居列表重建间隔(spec §5.2 / §14.1 Q-006/6b/6c)

## 目的

把 spec Q-006(`重建间隔 ~X fs`,最后一个活占位符)与 Q-006b/c(每 step 越界
概率 / 每 ns 重建次数)从占位变为实测定标,并对 Q-005 的 skin 预算分解给出
窗口级实证。

## 仪器选择(E0a 纪律)

轨迹用 **OpenMM 8.6(CPU 平台)**积分;A1 已证 opus 参考实现与其能量/力逐项
一致(rel<1e-10)。Q-006 是物理定标(稠密水位移统计),不是引擎正确性比较。
opus 引擎 direct_space 为逐对 Python 循环,实测 N=1020 单次力计算 23.7 s
⇒ 200 ps 不可行(登记于 2026-08-28)。

## 体系与参数(生产同构)

- TIP3P-FB 刚性水(HBonds + rigidWater),PME α=0.35 nm,rc=1.0 nm
- LangevinMiddleIntegrator(BAOAB 族,同 spec §6.1),γ=1 ps⁻¹,T=300 K
- 稳定性独立验证:O 与 H(×3 HMR)速度分布与 300 K Maxwell 精确吻合
  (⟨|v|⟩/rms = 0.92,Maxwell 理论值),PE/water ≈ −55 kJ/mol 正常
- 初始构象经 200 步最小化(G3b 纪律:addSolvent 直填 max|F| ~1.4e5 > 门)
  + 20 ps 平衡,测量 200 ps(seed 7/11 双种子)

## 记录语义(与 spec §5.2 逐条对应)

| 量 | 定义 | 记录 |
|---|---|---|
| 重建判据 | `max_a \|x − x_ref\| > d_thresh`,d_thresh = 0.05 nm | 内联 renewal(越界即回置参考点) |
| 全原子 / 重原子两套语义 | heavy = 质量 > 3.5(HMR×3 后 H=3.02 被排除) | 各自独立参考点 |
| C1 窗口模拟 | 每 100 fs(K=25@4fs)窗口内越界 = flag | 窗口原点 max 位移 |
| 窗口位移分布(干净) | 自 episode 原点(5 ps 内不回置)按 lag 对齐 | `max_d` / `mde_heavy` |
| D(原位) | MSD(episode 原点)线性段斜率 / 6 | `msd` / `msd_heavy` |

masking 严格(柱 5)⇒ 重建事件不改变轨迹值 ⇒ 单条轨迹支持事后多 d 值分析。

## 臂设计

| 臂 | box³ (nm) | 原子数 | dt | HMR | 用途 |
|---|---|---|---|---|---|
| b22_dt4_hmr | 2.2³ | 1020 | 4 fs | ×3 | 生产臂 |
| b30_dt4_hmr | 3.0³ | 2640 | 4 fs | ×3 | ln N 尺度 |
| b36_dt4_hmr | 3.6³ | 4515 | 4 fs | ×3 | ln N 尺度 |
| b22_dt2_nohmr | 2.2³ | 1020 | 2 fs | 无 | 步长 ∝ 检验(Q-006b) |
| b22_dt4_hmr_s2 | 2.2³ | 1020 | 4 fs | ×3 | 种子稳健性 |

## 文件

- `run_q006.py`:轨迹生成(OpenMM)+ 内联统计
- `analyze.py`:renewal 统计(bootstrap CI)/ 窗口位移分布 / D 拟合
- `results/*.npz|json`、`results/summary.json`

## 结果

定标完成(2026-08-28),全量结果与推导:**docs/m0/q006_rebuild.md**;spec 回填 v1.1.5(Q-006/6b/6c + 开放项 7)。要点:

- 全原子 renewal 17–19 fs(53–60/ns),重原子 37–41 fs(24–27/ns)@ d=0.05
- 弹道/转动主导(t∝√m_H 实测 1.74≈√3;d²/D 不适用)
- P(越界@K=25 窗口)=1.000(两种语义)⇒ 开放项 7:C1/Q-005 裁决件
- PME α 单位地雷(0.35/nm ↔ 0.35 Å⁻¹)登记;G4 建议 α·rc≥3 子句
- 五臂原始数据 results/*.npz,summary.json,analyze_stdout.txt
