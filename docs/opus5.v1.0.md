# 自研 GPU 自由能计算引擎 — 技术规格

**版本** v1.0 · 2026-08-25 · 状态：M0 启动基线

---

## 0. 一页摘要

| 维度 | 决定 |
|---|---|
| **目标负载** | 3–6 万原子显式溶剂，RBFE / ABFE，24 λ 窗口 × 2 独立重复 = 48 副本 |
| **硬件基线** | A100 80GB，**1 进程 : 1 GPU**，无 MPI / 无 NCCL |
| **2 卡用法** | 两个独立配体对并行（容量并行），或 1 卡生产 + 1 卡 MIG 跑小任务 |
| **精度** | 全程 **FP64** 计算 + **Q24.40 int64 定点**累加；FP32(SPFP) 留模板不实现 |
| **核心差异化** | λ 作为一等公民的 IR；$\partial U/\partial\lambda$ 自动微分；卡内零成本 HREX |
| **不自研** | 力场分派、参数化、拓扑映射、MBAR 求解器、体系构建 |

---

## 1. 范围与非目标

### 1.1 在范围内

- **输入**：OpenMM System XML + 坐标/盒矢量（唯一入口）
- **力项**：`HarmonicBond`、`HarmonicAngle`、`PeriodicTorsion`、`NonbondedForce`（PME + 排除 + 1-4 缩放）、`CMMotionRemover`、位置/距离限制势
- **炼金术**：Beutler–Steinbrecher 软核 LJ、静电线性/分离缩放、λ 向量
- **采样**：BAOAB Langevin、HREX、λ-dynamics / MSλD + ALF 偏置
- **输出**：完整 $u_{kn}$ 矩阵、$\partial U/\partial\lambda$ 时间序列、轨迹（可选下采样）

### 1.2 明确不做

| 项 | 理由 |
|---|---|
| 多卡空间分解 / PME 分解 | 3–6 万原子规模下通信吃掉收益；容量并行严格更优 |
| FP16 任何用途 | 力的动态范围超 $10^5$，FP16 尾数 10 位不可用 |
| 生产阶段 NPT | 多副本共享邻居列表要求共享盒子 |
| 解析维里 / 压强张量 | 用 MC barostat 绕开；有约束 + PME + λ 时极易出错 |
| 对微扰键施加约束 | ingest 阶段直接报错；改用柔性键 + HMR |
| 力场文件解析 / 参数化 | OpenMM 已解决 |
| 自研 MBAR 求解器 | 用 pymbar 4 |
| Amber prmtop 入口 | 与 OpenMM 炼金术表示非一对一映射，单独规划 |

---

## 2. 数据布局

### 2.1 副本维度为最内层（最重要的单一决策）

$$\texttt{x}[a][r], \quad a \in [0,N), \; r \in [0,R)$$

$R$ 编译期固定为 **48**（24 λ × 2 独立重复），padding 到 8 的倍数。

**收益**：
- 参数读取为广播（走 L1/常量缓存）
- 坐标访问完全合并
- **零 warp divergence** —— 副本间只有 λ 不同，指令流完全一致
- 邻居列表可跨副本共享

**Tile 结构**：`[8 atoms] × [8 atoms] × [R replicas]`，cluster-pair 式。A100 的 164 KB/SM 共享内存（vs V100 96 KB）是 tile 尺寸搜索的主要空间。

### 2.2 显存预算（$N=60000$, $R=48$, 全 fp64）

| 项 | 占用 |
|---|---|
| 坐标 / 速度 / 力 / 上步力 | 138 MB |
| 定点力累加器 + $\partial U/\partial\lambda$ 累加器 | 138 MB |
| PME 网格 $128^3$ complex fp64 × 48 | 1.6 GB |
| cuFFT 批量工作区 | 3–6 GB |
| 共享邻居列表 | ~300 MB |
| $u_{kn}$ 缓冲 / 参数表 / 排除表 | < 200 MB |
| **合计** | **~8 GB / 80 GB** |

余量留给更大体系或更多副本，**不留给多卡缓冲**。

---

## 3. 精度模型

```
PrecisionPolicy {
  force_compute : f64            // M0–M3 唯一实现
  force_accum   : Q24.40 int64   // 无条件，位级确定性来源
  energy_accum  : f64 固定形状树形归约  或  Q34.30
  pme_recip     : f64            // FFT 占比过高时可单独降 f32，用 Madelung 验证
}
```

### 3.1 Q24.40 定点累加

$$F_{\text{fixed}} = \text{round}\!\left(F_{\text{float}} \times 2^{40}\right)$$

24 位整数部分 + 40 位小数部分。分辨率 $2^{-40} \approx 9\times10^{-13}$ kJ/mol/nm，范围 $\pm 8.4\times10^6$。沿用 Amber SPFP 的定标选择（其日志字面为 `Single Precision Forces, 64-bit Fixed Point`，**不含任何 FP16**）。

整数加法满足交换律与结合律 → `atomicAdd(int64)` 结果与线程调度顺序无关 → **位级确定性免费获得**。

定标位数为编译期常量，可在溢出断言保护下搜索。

**溢出检测**：debug 模式强制检查累加器接近 int64 边界（静默回绕成反号力是灾难性的）。release 关闭但代码必须存在。

### 3.2 为什么 A100 上选全 FP64

| GPU | FP32 | FP64 | 比例 |
|---|---|---|---|
| **A100** | 19.5 TF | **9.7 TF** | **2:1** |
| RTX 4090 / L40S | ~90 TF | ~1.4 TF | 64:1 |

A100 的 FP64 是硬件级 2:1，且 MD 力计算混杂大量非 FLOP 开销（邻居列表遍历、访存、整数索引、atomic），实测端到端损失估计 **1.3–1.8×**。消费卡的 fp64 比 A100 低约 10 倍[ref:6]，这是"全 fp64"只在 A100 上成立的原因。

换来的：oracle 与生产同源、单路径迭代（不必在两条精度路径上验证每个优化）、位级确定性无条件成立。

**SPFP 保留为模板实例**，不实现。将来实现时 **DPFP 就是它的天然 oracle**（同一份定点累加器，断言力差 rel $< 10^{-6}$）。Amber 的 TI/FEP 生产流程长期跑在 SPFP 上，说明 fp32 力 + 定点累加对自由能计算精度是够用的。

> **前置实验 E0（写代码前必做）**
> OpenMM 在本机 A100 上同一 60k 体系 `mixed` vs `double` platform 的 ns/day 对比。排除 JIT 与预热，多次取中位数。
> 若损失 > 3×，回退到 `f32 计算 + Q24.40 累加 + f64 oracle 专用路径`。

---

## 4. λ 层 IR（核心差异化）

### 4.1 IR 结构

```
ForceTerm {
  kind     : LJ | Coulomb | Bond | Angle | Torsion | Restraint
  atoms    : [i, j, ...]
  params   : [ParamExpr, ...]        // 参数是表达式，不是数
  softcore : SoftcorePolicy | None
}

ParamExpr = Const(v)
          | Interp { a: v0, b: v1, schedule: Sched }   // (1-f)·v0 + f·v1
          | Scale  { base: v, schedule: Sched }
          | Custom(expr_tree)

Sched     = Linear | SoftcoreOffset | Piecewise | LambdaComponent(i)
```

### 4.2 $\partial U/\partial\lambda$ 由 IR 自动微分，不手写

$$\frac{\partial U}{\partial \lambda_i} = \sum_{\text{terms}} \sum_{p} \frac{\partial U}{\partial \theta_p} \cdot \frac{\partial \theta_p}{\partial \lambda_i}$$

内核在算力时顺路算 $\partial U/\partial\theta_p$（LJ/Coulomb 解析且便宜），乘上 IR 给出的 $\partial\theta_p/\partial\lambda_i$，累加进独立定点累加器。

**边际成本估计 10–20%**，换来 TI、λ-dynamics、MSλD、$u_{kn}$ 全部免费。

### 4.3 软核必须在 IR 层表达

不得藏进内核 —— 否则无法与 openmmtools 做逐项差分测试。

$$U_{\text{LJ}}^{sc} = 4\epsilon(\lambda)\left[\frac{1}{\left(\alpha(1-\lambda)^2 + (r/\sigma)^6\right)^2} - \frac{1}{\alpha(1-\lambda)^2 + (r/\sigma)^6}\right]$$

$\alpha$、指数 $a,b,c$、静电是否与 LJ 同步缩放，全部为 policy 参数。

### 4.4 λ 存储与求值

每副本一个 $\boldsymbol\lambda^{(r)} \in \mathbb{R}^{L}$（$L$ = 微扰位点数 + REST2 的 $\beta$ 缩放分量）。

参数在 kernel 内**实时求值**（λ-dynamics 下 λ 每步变），不预计算展开。IR 编译期做常量折叠 + 把表达式树 lower 成内联算术，**不做运行时解释**。

### 4.5 MSλD 的直接推论

$\partial U/\partial\lambda_i$ 已有 → $\lambda_i(\theta)$ 是几十行 host 代码 → θ 的 BAOAB 是几十行。**不需要任何新内核。这是从零自研引擎唯一真正划算的理由。**

---

## 5. 非键相互作用

### 5.1 PME

- **空间并列**：$R$ 份网格同时存在 + 一次批量 cuFFT，形状完全静态（CUDA Graph 友好）
- **不做时间复用**（80GB 下无必要，且会破坏 graph 结构、降低 FFT 批量效率）；`PME_TILE ∈ {1,4,8,R}` 开关保留，默认 $R$
- **网格不可跨副本共享**：各副本电荷经 λ 缩放不同

**三处必须一致缩放**（雷区，M0 即断言）：倒空间、正空间排除修正、自能项。不一致会产生 0.1–1 kcal/mol 的**静默**误差（代码不崩，$\Delta G$ 偏移）。

**倒空间对电荷是二次依赖** → 电荷线性插值时 $u(\lambda)$ 非线性。这是最容易被写错的单点。

### 5.2 邻居列表：单份共享

- 所有副本共用一份列表
- 重建判据：$\max_{a,r} \lvert x_a^{(r)} - x_a^{(r),\text{ref}} \rvert > \text{skin}/2$
- skin 加大至 **0.25 nm**（换来列表显存与构建成本 $\div R$）
- 构建使用**所有副本坐标的并集**，保证列表是各副本真实列表的**超集**

副本几何高度相似（同构象出发，仅 λ 不同），这是单卡多副本的最大红利之一。

---

## 6. 积分器、约束、恒压

### 6.1 BAOAB Langevin (VRORV)

同阶方法中配置空间离散化误差最小，而自由能计算只关心配置空间分布。

$$\underbrace{v \mathrel{+}= \tfrac{h}{2m}F}_{\text{B}} \;\; \underbrace{x \mathrel{+}= \tfrac{h}{2}v}_{\text{A}} \;\; \underbrace{v \leftarrow e^{-\gamma h}v + \sqrt{k_BT(1-e^{-2\gamma h})/m}\;\xi}_{\text{O}} \;\; \underbrace{\phantom{x}\text{A}\phantom{x}}_{} \;\; \underbrace{\phantom{v}\text{B}\phantom{v}}_{}$$

### 6.2 随机数：必须 counter-based

Philox / Threefry，种子 = `(global_seed, step, atom, replica, dof)`。

结果与线程调度、副本数、卡数无关。**禁用 cuRAND 状态式生成器** —— 它会破坏位级可复现性。

### 6.3 约束

SETTLE（刚性水）+ CCMA（其他）。

**微扰键禁止约束** —— ingest 阶段校验并报错。微扰区域用柔性键 + HMR + 小步长。这比正确处理约束的 λ 导数与自由度重计数便宜得多，且几乎不损失通用性。

### 6.4 恒压

- **预平衡**：Monte Carlo barostat。只需能量、不需维里 → 正确性仅依赖能量的正确性（已被外部 oracle 覆盖）。每 25–100 步一次尝试，需重算全能量含 PME，频率低可接受
- **生产**：**NVT**，体积取预平衡 NPT 平均后固定

这是明确的"用性能换正确性"选择：把维里从关键路径上移走。等引擎稳定后再考虑加解析维里（那时已有完整 oracle 套件可验证它）。

---

## 7. CUDA Graph

```
loop:
  rebuild_neighborlist()      # host，数据依赖分支
  graph_replay(K steps)       # 一次 launch，K = 20–50
  maybe_hrex_swap()           # host，每 50–100 步
  maybe_barostat()            # host，仅预平衡期
  maybe_alf_update()          # host，迭代级
```

| Graph 内 | Graph 外 |
|---|---|
| 全部力计算、积分、约束、$\partial U/\partial\lambda$ 累加、能量累加、**θ 的 Langevin** | 邻居列表重建、HREX 决策、MC barostat 判据、ALF 偏置**系数**更新、PLUMED 耦合 |

注意：ALF 偏置**力的求值**在 graph 内，只有系数更新在外。θ 积分虽小（$L \times R$ 自由度）但每步都要，必须在 graph 内。

**动机**：3–6 万原子 × 48 副本的单步内核数百微秒，而数十个 kernel × 5–10 μs launch 开销占 10–30%。Graph 化能拿回大部分。

---

## 8. HREX（单卡最大红利）

### 8.1 交换 λ 而非坐标

副本全在同一张卡的显存里 → **交换 = 交换 λ 向量索引，坐标不动**。成本接近零，且与体系大小完全解耦。

这不只是优化，而是设计原则：即使将来跨卡，λ 交换也只有几十个 double。

### 8.2 激进的交换策略

传统实现受通信成本限制，交换间隔通常 ≥ 1000 步。本设计可以：

- **每 50–100 步交换一次**
- **Gibbs sampling over 全部副本对**，而非仅相邻交换

### 8.3 $u_{kn}$ 是顺路产物

全 $u_{kn}$（$48\times48$）每 $M$ 步算一次。**同一构象在不同 λ 下的能量可复用同一份邻居列表与同一份距离计算**，只是参数不同 → 一个 kernel 内循环 λ，成本约 **3–5× 单次能量求值**，而非 $R^2$ 倍。

因此 HREX 判据与 MBAR 输入数据一次拿到。$u_{kn}$ 落库 schema 第一天定对，它是后续 MBAR、方差估计、EVSI 分配、难度模型训练的共同基础数据。

MBAR 求解与不确定度用 pymbar 4。

---

## 9. 硬件策略与调度

### 9.1 引擎的硬件模型

```
引擎      : 1 进程 : 1 GPU : 1 CUDA context : 1 Graph，无 MPI / 无 NCCL
2 卡默认  : 两个独立配体对并行（容量并行，扩展效率定义上 100%）
2 卡备选  : 1 卡整卡跑生产 FEP + 1 卡 MIG 切片跑预平衡 / ALF 迭代 / 参考态
FP64 TC   : 不用（仅对 DGEMM 有效，MD 力计算与 FFT 用不上）
异步拷贝  : cp.async 预取邻居列表 / 参数表
```

**为什么单卡独立跑严格更优**：
- 一个完整 RBFE（48 副本、60k 原子、8 GB、吃满 108 SM）本身就是不可再分也不需再分的工作单元
- 决策目标是**吞吐**（每 GPU-day 产出的收敛 $\Delta G$），不是单任务延迟。双卡跑一个任务与双卡跑两个任务产出相同，但后者复杂度低一个量级
- 单卡任务是调度器最友好的形状 —— 装箱效率 100%，永不因凑不齐资源而排队。只有 2 张卡时，一个双卡任务会让系统在其跑完前无法接受新任务
- 故障隔离：一卡失败只损失一个任务

**删掉多卡路径直接消除**：NVLink/NCCL 通信层、跨卡 HREX 交换、跨卡 $u_{kn}$ 归约、多卡故障恢复、多 rank 种子协调。多卡是"偶发不可复现失败"的主要来源。

**多卡协作（backlog）**：仅在 (a) 体系 > 10 万原子 + 2D 交换网络（24 λ × 4 REST2 温度 = 96 副本）(b) 单个难配体对紧急深挖。且实现为**扩采样量（更多副本）**，不是切并行度。

### 9.2 调度层

$$\text{队列} = \{(\text{ligand pair},\; \text{预算 GPU-hour},\; \text{优先级})\}, \qquad \text{资源} = \{2\ \text{slots}\}$$

SQLite + 守护进程，**几百行**。用不着 Slurm / k8s。

支持：抢占（checkpoint 后换任务）、EVSI 动态预算调整、方差不足时续跑、杀掉不值得继续的配体对。

资源单位是"**GPU 切片 + 显存配额**"而非"GPU"，以容纳 MIG 场景。

**IO**：$u_{kn}$ **批量追加 + 异步写**（为将来 8 卡扩容时避免 IO 竞争预留）。

---

## 10. 性能靶子

| 场景 | A100 参考值 |
|---|---|
| 单副本 60k PME 2 fs，fp32 | 200–400 ns/day |
| 单副本 fp64 | 130–250 ns/day |
| **48 副本批量 fp64（本引擎目标）** | **每副本 30–70 ns/day，总 1.5–3.4 μs/day** |

**批量红利来源**：邻居列表构建摊薄 15–25% · 参数读取变广播 5–10% · launch 开销摊薄 10–30% · occupancy 改善。

**M2 验收**：总吞吐 ≥ 单副本 fp64 的 **5×**。
- < 4× → **red flag**，批量设计未做对
- \> 10× → 优秀

**性能评估规则**：
1. benchmark 参数**随机化**（体系大小、密度、三斜盒、离子浓度、λ 窗口数、副本数），报告**最差情况**而非平均
2. 顶层指标 = **每 GPU-day 产出的收敛 $\Delta G$ 数量**（内生化内核速度、批量效率、采样效率、收敛门控）；ns/day 仅作日级代理指标

---

## 11. 正确性验证

### 11.1 外部参照（版本钉死，仓库外）

| ID | 内容 | 容差 |
|---|---|---|
| **A1** | OpenMM 逐 Force 能量与力（同 XML 同坐标） | rel $< 10^{-10}$ |
| **A2** | openmmtools `AbsoluteAlchemicalFactory` + Beutler 软核（$\alpha{=}0.5,a{=}1,b{=}1,c{=}6$）的全 $u_{kn}$ 层，固定构象集，**零动力学** | rel $< 10^{-9}$ |
| **A3** | 解析解：谐振子 $\Delta G$、Madelung 常数、两态 BAR 理论方差、λ 路径无关性 | 精确 |
| **A4** | 公开基准：FreeSolv 实验值、JACS set 文献 RMSE | 统计一致性 |
| **A5** | 守恒律：NVE 能量漂移、动量守恒、量纲检查 | 物理定律 |

### 11.2 变形关系（位级，零容差）

| ID | 断言 |
|---|---|
| **M1** | $R$ 副本批量 $\equiv$ $R$ 次 $R{=}1$ 串行 |
| **M2** | `graph_replay(K)` $\equiv$ $K \times$ `single_step()` |
| **M3** | 结果与线程块尺寸 / occupancy / tile 形状无关 |
| **M4** | 结果与副本数 $R$ 无关（固定 seed + counter-based RNG） |
| **M5** | 平移 / 旋转 / 周期映像不变性 |
| **M6** | 原子重排序不变性（定点累加保证） |

### 11.3 近似变形（须断言容差上界）

| ID | 断言 | 容差 |
|---|---|---|
| **M7** | 共享邻居列表 vs 每副本独立列表 | rel $< 10^{-6}$ |
| **M8** | 不同 skin 厚度 | rel $< 10^{-7}$ |
| **M9** | λ 端点 vs 对应的非炼金术 System | rel $< 10^{-9}$ |
| **M10** | SPFP vs DPFP（将来） | rel $< 10^{-6}$ |

### 11.4 雷区清单

| # | 雷区 | 断言方式 |
|---|---|---|
| 1 | PME 倒空间对电荷**二次**依赖 → 电荷线性插值时 $u(\lambda)$ 非线性 | A2 多 λ 差分 |
| 2 | PME 排除修正 / 自能项的 λ 缩放须与倒空间一致 | 三处独立断言 |
| 3 | 净电荷改变的微扰：单位盒须电中性，需显式有限尺寸修正 | 文献锚 + 盒尺寸扫描 |
| 4 | 盒尺寸效应在中性溶剂化中可忽略，**对带电体系不成立**（条件性成立的知识） | 双体系对照 |
| 5 | 微扰原子上的约束 → 自由度计数与约束 λ 导数 | ingest 报错 |
| 6 | 软核形式 / $\alpha$ / 指数与参考实现不一致 | A2 逐项 |
| 7 | 限制势的标准态修正（ABFE） | 解析解 |
| 8 | dummy 原子的约束与内坐标处理 | 显式测试 |
| 9 | 定点累加器溢出（静默回绕成反号力） | debug 断言 |
| 10 | RNG 种子维度缺失 → 副本间相关 | M4 断言 |
| 11 | 共享邻居列表非超集 | M7 断言 |
| 12 | HREX 判据的 $u_{kl}$ 索引错位 | 注入测试 |

**高风险域**（推导繁琐但结论简短，易记住结论而丢掉适用条件；错的适用条件不会让代码崩，只让 $\Delta G$ 偏 0.5 kcal/mol）：约束自由度计数、维里与压强、PME 自能/排除修正、净电荷有限尺寸校正。这些必须硬编码进测试，不能靠推理覆盖。

---

## 12. 里程碑

### M0 — 静态正确性（无动力学）

> **验收**：任意 System XML + 坐标，逐 Force 单点能量与力对齐 OpenMM，fp64 相对误差 $< 10^{-10}$，涵盖 Bond / Angle / Torsion / Nonbonded(PME + 排除 + 1-4) / 约束。$R$ 副本批量与串行**位级相同**。

交付：A1/A2 harness（版本钉死）+ A3 解析体系库 · Q24.40 累加器 + 溢出断言 · counter-based RNG · delta debugging（失败用例自动最小化：原子数、λ 窗口数、步数、力项子集）· **前置实验 E0**

*不含性能。慢的 $O(N^2)$ fp64 直算版本即可定义语义。*

### M0.5 — 动力学
NVE 能量漂移 < 阈值 · NVT 温度分布 · 动量守恒 · SETTLE/CCMA 收敛性 · M2 graph 断言

### M1 — λ 层
IR + $\partial U/\partial\lambda$ 自动微分 · 软核对齐 openmmtools 全 $u_{kn}$ 层 · M9 端点断言 · λ 路径无关性 · PME 三处一致缩放断言

### M2 — 批量与性能
共享邻居列表（M7/M8）· CUDA Graph · tile / 共享内存搜索 · **总吞吐 ≥ 5× 验收**

### M3 — 采样层
HREX（λ 交换 + Gibbs 全对）· $u_{kn}$ 落库 schema · λ-dynamics / MSλD + ALF · pymbar 4 接入 · 谐振子端到端 $\Delta G$ 验证

### M4 — 生产验证
FreeSolv 子集溶剂化自由能 · JACS set RBFE vs 文献 RMSE · 调度器 + EVSI 分配器

---

## 13. 关键开放项

| # | 待定 | 影响 | 决策时点 |
|---|---|---|---|
| 1 | E0 实测 fp64 损失是否 < 3× | 单路径 vs 双路径（渗透每行内核） | **M0 之前** |
| 2 | System XML 是否会出现 `CustomNonbondedForce` | IR 是否需通用表达式求值器 | M1 之前 |
| 3 | 是否需 Amber prmtop 入口 | ingest 层 + TI 设置（`icfe`/`ifsc`/`timask`/`scmask`）映射 | M4 之后 |
| 4 | GPU 是否会扩到 8 张 | host CPU 核数、NUMA 拓扑、IO schema | M3 之前 |
| 5 | HREX 交换频率最优值 | 在收敛速度指标下搜索 | M3 |

---

## 附：设计信条

1. **先有慢而正确的语义定义，再让它变快。** 性能优化只在位级不变性 gate 通过后才允许。
2. **副本为最内层维度。** λ-FEP 的礼物是副本间几何相似、指令流一致。
3. **累加器无条件定点。** 位级确定性不是奢侈品，它是所有变形测试的前提。
4. **容量并行 > 能力并行。** 吞吐目标下，扩展效率 100% 的最简实现就是各跑各的。
5. **交换 λ，不交换坐标。** 让通信成本与体系大小解耦。
6. **把维里从关键路径移走。** 能不算的物理量就不算。
7. **$u_{kn}$ 是顺路产物，不是额外开销。** 这决定了 HREX、MBAR、方差估计共用一份数据通路。

