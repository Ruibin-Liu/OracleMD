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
| `e0b_direct_pairbench.py` | 直空间对循环微基准,f64/f32 × R∈{1,8,48}(util 0%,2026-08-28;看门狗轮因 GPU 忙无效后手动补跑) | f64 R=1→48 = 14.0→29.5 Gpair/s(η 8.7%→18.3%,60 FLOP/对口径,intrinsic erfc 慢路径,**R 标度比为本测试目的**)⇒ 直空间红利 2.10×(f64)/5.98×(f32);E0d 直空间分量 f32/f64@48 = **2.84×** |

时序类三项均于 2026-08-28 落地(看门狗 168h 窗口 + 手动补跑),原始日志:a100-pod `/root/e0_results.log`;环境快照:CUDA 13.0 / driver 580.105.08 / cuPy 14.2.0 / OpenMM 8.2。回填位置:spec v1.1.3 的 Q-004/Q-004b/c、§10、§13 开放项 1 与 `docs/m0/feasibility.md` 末节。Q-002 保持 provisional(intrinsic erfc 使微基准绝对值不可直接替换,替换点推到 M2 生产内核)。**E0b-constr 分量未单测**,并入 M2 归因。

## 时序类三项已于 2026-08-28 完成(见上表);看门狗脚本保留备复测

| 脚本 | 内容 |
|---|---|
| `e0_watchdog2.sh` | 看门狗:util≤20% 持续 2 分钟自动顺序跑 E0a/E0b-FFT/E0b-direct,带环境快照与置信度分级(util<5% = USABLE,5–20% = REFERENCE-ONLY),结果写 `~/e0_results.log` |

按 spec §10.3(基线来源纪律):M2 硬地板基线 = max(E0a OpenMM double, e0b_direct R=1);M0 参考实现**不**作基线。
