# E0 前置实验脚本(spec §3.5)

对应 `docs/opus5.md`(v1.1.1)的 E0 系列与两项已完成实测。

## 环境(A100 pod)

- A100 80GB PCIe,driver 580.105.08,CUDA 13.0
- cuPy 14.2.0(捆绑 cuFFT 12.x)——位级测试
- OpenMM 8.6(conda-forge,含 CUDA 平台;PyPI wheel 无 CUDA 平台,已验证弃用)
- 远程部署:`~/.venvs/e0`(cuPy)、`~/envs/e0omm`(OpenMM)

## 已完成(结果记录于 docs/reviews/opus5-review.md 第四、五轮)

| 脚本 | 测试 | 结论 |
|---|---|---|
| `e0e_cufft_bitwise.py` | cuFFT fp64 位级:同 plan 重复 / batch {1,4,8,48} / in-place | **全部位级相同**(c2c 64³/128³) |
| `e0e2_cufft_r2c.py` | r2c fwd / c2r inv,96³–128³(含非 2 幂),batch {1,24} | 全部位级相同(注意 `axes=` 参数教训:漏写会假 DIFF) |
| `e0e3_workspace_pressure.py` | 显存压力(压至 1.5 GiB)+ plan 缓存清空强制重 plan | 仍位级相同 ⇒ I-012 获两个独立压力维度支持 |
| `b1_atomic_spread.py` | fp64 atomicAdd 铺展 vs int64 Q16.48 | fp64 **连 run-to-run 都位级不同**(同配置 ×5 全 DIFF);int64 跨 block 配置位级相同 ⇒ 定点网格为 M0 前置 |

## 待 GPU 空闲窗口(时序类,util 门卫 >20% 拒跑)

| 脚本 | 内容 |
|---|---|
| `e0a_openmm_precision.py` | OpenMM ~60k 原子 mixed vs double(自包含种子 PDB;conda python 运行) |
| `e0b_fft_throughput.py` | 批量 cuFFT 128³ 吞吐标度,batch {1,4,8,16,48} |
| `e0b_direct_pairbench.py` | 直空间对循环微基准:replica 最内层 + 共享列表,f64/f32 × R∈{1,8,48}(E0b-direct + E0d 直空间分量;M2 硬地板基线来源之一) |
| `e0_watchdog2.sh` | 看门狗:util≤20% 持续 2 分钟自动顺序跑 E0a/E0b-FFT/E0b-direct,带环境快照与置信度分级(util<5% = USABLE,5–20% = REFERENCE-ONLY),结果写 `~/e0_results.log`,24h 超时 |

按 spec §10.3(基线来源纪律):M2 硬地板基线 = max(E0a OpenMM double, e0b_direct R=1);M0 参考实现**不**作基线。
