# E0 前置实验脚本(spec §3.5)

对应 `docs/opus5.md`(v1.1.1)的 E0 系列与两项已完成实测。

## 环境(A100 pod)

- A100 80GB PCIe,driver 580.105.08,CUDA 13.0
- cuPy 14.2.0(捆绑 cuFFT 12.x)——位级测试
- OpenMM 8.2(conda-forge,含 CUDA 平台;PyPI wheel 无 CUDA 平台,已验证弃用)
- 远程部署:`~/.venvs/e0`(cuPy)、`~/envs/e0omm`(OpenMM)

## 已完成(结果记录于 docs/reviews/opus5-review.md 第四、五轮)

| 脚本 | 测试 | 结论 |
|---|---|---|
| `e0e_cufft_bitwise.py` | cuFFT fp64 位级:同 plan 重复 / batch {1,4,8,48} / in-place | **全部位级相同**(c2c 64³/128³) |
| `e0e2_cufft_r2c.py` | r2c fwd / c2r inv,96³–128³(含非 2 幂),batch {1,24} | 全部位级相同(注意 `axes=` 参数教训:漏写会假 DIFF) |
| `e0e3_workspace_pressure.py` | 显存压力(压至 1.5 GiB)+ plan 缓存清空强制重 plan | 仍位级相同 ⇒ I-012 获两个独立压力维度支持 |
| `b1_atomic_spread.py` | fp64 atomicAdd 铺展 vs int64 Q16.48 | fp64 **连 run-to-run 都位级不同**(同配置 ×5 全 DIFF);int64 跨 block 配置位级相同 ⇒ 定点网格为 M0 前置 |
| `e0a_openmm_precision.py` | OpenMM 60k 原子 mixed vs double(排除 JIT/预热,中位数;util<5% USABLE,2026-08-28) | mixed 341.7 vs double 136.0 ns/day @2fs ⇒ **fp64 损失 2.51× < 3× ⇒ 开放项 1 关闭:fp64 单路径保留**;136.0 ns/day 锚定 §10.3 硬地板基线(Q-004b 外部锚点) |
| `e0b_fft_throughput.py` | 批量 cuFFT 128³ c2c 吞吐,batch {1,4,8,16,48}(util<5% USABLE,2026-08-28) | per-transform ~0.28 ms **与 batch 无关**(~1.54–1.59 TFLOP/s,~480 GB/s,带宽受限)⇒ FFT 批量红利 ≈1.0;「FFT 红利故事不同于直空间」的预测获确认 |
| `e0b_direct_pairbench.py` | 直空间对循环微基准,f64/f32 × R∈{1,8,48}(util 0%,2026-08-28;看门狗轮因 GPU 忙无效后手动补跑) | f64 R=1→48 = 14.0→29.5 Gpair/s(2026-09-01 修正自变量笔误 erfc(α·invr)→erfc(α/invr),计时结论不变;数值语义错误由 E0f xcheck 抓出)(η 8.7%→18.3%,60 FLOP/对口径,intrinsic erfc 慢路径,**R 标度比为本测试目的**)⇒ 直空间红利 2.10×(f64)/5.98×(f32);E0d 直空间分量 f32/f64@48 = **2.84×** |
| `e0f_erfc_poly_bench.py` | 直空间内核 poly-erfc 变体 + 力交叉验证(M2 第一件;util 0%,2026-09-01) | **erfc = exp(−x²)·W(x),W 三段 Chebyshev(deg 18/14/18),设备端 vs intrinsic max_rel 3.11e-14**([0,6.5];系数 fit_erfc_poly.py 程序化生成);力输出一致性 **2.2e-16**;f64 R48 = **33.7 Gpair/s(+20% vs intrinsic 28.1)**,批量红利 2.35×,R1 1.01×(串行延迟受限,erfc 非瓶颈——E0b 的判断获确证) |
| `e0g_ilp_bench.py` + `e0g_kernels.cu` | 直空间内核 ILP/算术阶梯 + 紧凑表对照(物理几何:真实 1.35 nm 表、α=3.5、rc mask;util 0%,2026-09-08) | **合规生产形态 k2(算术重构+预折叠):直空间 16.0 ms/步@R48,37.4 计算对 Gpair/s(+18% vs naive)**;k3 展开×4 倒退(-16%,mask 分支与展开互斥);**k4/k5 紧凑表 +38% 但规格不合规**——rc 处截断丢环形区,窗口内向内漂移对漏算(xcheck 3.4e-10 = 漏对指纹;柱 5「+0≡缺席」仅对 r>rc 成立);环形区占直空间 26%(skin 的结构性代价)。Q-002:直空间 16.0 ms 实测 ⇒ 总步时 ~24 ms,provisional 带(610–760 ns/day)从下方获确认 |
| `e0h_pme_spread.py` | PME 铺展/回插微基准(order-4 样条,Q16.48 int64 atomicAdd;权重与 opus/pme 参考精确一致 2.8e-16;util 0%,2026-09-08) | 铺展+清零 **10.7 ms/步**@R48(17.2 G-atomic/s)、回插 **0.9 ms**;**柱 2 GPU 端首次直接验证:两遍铺展 bitwise identical**;电荷守恒 rel 5.9e-14;PME 全链 24.9 ms ⇒ 总步时 ~43+ ms、~380 ns/day@48(**roofline PME=40%×直空间假设被否,归因入 Q-002**) |
| `e0h2_spread_tile.py` | 铺展优化:cell 排序 + shared tile 暂存(flush 只写 cell 拥有点;2026-09-08) | **tile 4.69 ms vs 全局原子 v1 13.0 ms = 2.78×**;oracle:v1≡tile bitwise(整数和交换律)、电荷三方精确一致、两遍确定性;**顺带揪出 E0h 两个潜伏 bug**:①weights4 的 anchor[1..3] 从未赋值(错位沉积,总和检查空转掩盖——逐点 numpy oracle 抓出,E0h 原 10.7 ms 作废)②Q 常数误用 2^52(应 2^48,守恒检查两侧同除自洽掩盖) |
| `e0i_stream_overlap.py` | 直空间∥PME 双流重叠研究(k2 真实内核 + cuFFT r2c/c2r 链;util 0%,2026-09-08) | **负结果:重叠零收益**(44.05 vs 44.15 ms)——两 kernel 均饱和 SM,同卡工作量守恒;分量复测:direct 16.0、pme_chain 28.4(v1 铺展口径;tile 后 ~20);「FFT 重叠」杠杆判死,登记 |
| `e0j_constrain_integrate.py` | 约束/积分微基准(BAOAB 流式含 philox counter-RNG Box-Muller;刚性水 SHAKE 3 约束×12 固定迭代;2026-09-09) | 积分 0.25 ms(RNG 子分量 0.11)、**SHAKE 1.27 ms((水,副本)并行版;首版每水串行 48 副本 6.52 ms,慢 5.1×)**;合计 **1.51 ms/步**;**分量地板表收官:16.0 + 4.7 + 13.3 + 0.9 + 1.5 ≈ 37.5 ms ⇒ ~440 ns/day@48** |
| `e0g_ilp_bench.py`(A/B 版) | 直空间深水区:`-prec-div/-prec-sqrt=false` 交替 A/B(2026-09-09) | **负结果 1.00×,力差 0.0**(代码生成未变);至此微杠杆全灭(展开/紧凑/重叠/fast-div);**占用守卫复测(331 污染迭代丢弃):k2 干净 min 15.65 ms**;fast-div 复判死;「布局下限」修正为「实测最优」——带宽墙 vs 延迟墙未定,ncu 剖析为架构路线前置 |
| `e0k_ablation.py` | 消融阶梯 P1→P3b→P3→k2→P4(独占窗口,vLLM 按约暂停;2026-09-10) | **成本栈定案**:访存 5.65(36%)/ LJ+div +3.12 / 库仑数学 +7.06(poly-erfc 已省 6.5,intrinsic 单项 11.06);**P4 poly-exp 仅 +0.3 ms**(poly 成本≈exp 成本);杠杆总账与 H(x) 融合力函数 backlog 见 review;守卫独占自咬修复(exclusive 探测) |
| `e0l_graph_step.py` | CUDA Graph + K 窗口装配原型(两段图夹 FFT 链;设备端步数计数器;C1 flag 重原子 d=0.14;2026-09-10) | **图重放 ≡ 串行位级三全(x/v/grid)**;launch 开销 <0.5%(Q-003 修正);C1 flag 图内工作;FFT 不入图登记(cupy 无 out=,生产用 cufft exec 节点);NaN≠NaN 假阳性追查记录(位视图比较教训) |
| `e0c_n_scaling.py` | 单副本 N 标度{5k,20k,60k},CUDA,double 主路径 + mixed 参照(util 0%,2026-09-01) | double:523.1/314.0/134.9 ns/day(0.330/0.550/1.281 ms/step);60k 与 E0a 136.0 交叉一致;**fp64 损失随 N 增长:1.71×@5k → 2.25×@20k → 2.54×@60k**(开放项 1 的 3× 判据在更大 N 有逼近风险,趋势登记) |

时序类三项均于 2026-08-28 落地(看门狗 168h 窗口 + 手动补跑),原始日志:a100-pod `/root/e0_results.log`;环境快照:CUDA 13.0 / driver 580.105.08 / cuPy 14.2.0 / OpenMM 8.2。回填位置:spec v1.1.3 的 Q-004/Q-004b/c、§10、§13 开放项 1 与 `docs/m0/feasibility.md` 末节。Q-002 保持 provisional(intrinsic erfc 使微基准绝对值不可直接替换,替换点推到 M2 生产内核)。**E0b-constr 分量未单测**,并入 M2 归因。

## 时序类三项已于 2026-08-28 完成(见上表);看门狗脚本保留备复测

| 脚本 | 内容 |
|---|---|
| `e0_watchdog2.sh` | 看门狗:util≤20% 持续 2 分钟自动顺序跑 E0a/E0b-FFT/E0b-direct,带环境快照与置信度分级(util<5% = USABLE,5–20% = REFERENCE-ONLY),结果写 `~/e0_results.log` |

按 spec §10.3(基线来源纪律):M2 硬地板基线 = max(E0a OpenMM double, e0b_direct R=1);M0 参考实现**不**作基线。
