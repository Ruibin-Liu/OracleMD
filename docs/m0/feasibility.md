# M0 可实现性证明(spec §12:M0 双交付的第二半)

> 每条 M0 立下的断言,在目标 GPU 实现中由哪个机制保证;以及在本参考
> 实现中已由哪条测试验证。参考实现 = `opus/`(结构同构:PME 网格 +
> 定点累加 + R=1 语义;无 tile 搜索,慢)。

## 断言 → 机制映射

| M0 断言 | 参考实现机制(已验证) | GPU 实现机制(设计) | 测试 |
|---|---|---|---|
| M1 批量≡串行 位级 | 共享并集列表 + 乘0 masking + Q24.40 整数累加 | 同(柱1+5) | `test_m1_batch_equals_serial_bitwise` ✓ |
| M3 形状无关 | 单实现,数值由定点保证 | 定点累加顺序无关;CCMA 固定迭代数(柱4) | (GPU 期 M3 扫描) |
| M4 R 无关 | R 为循环维度,每副本独立计算 | R 模板参数;跨二进制依赖柱1–5 | `test_m1`(R=8 vs 1 为其子集;完整 M4 在 GPU 期跨实例化) |
| M5a 格矢平移 位级 | 分数坐标折叠 + MIC 均以盒为常量 | 同;前提=二进制可表示(登记) | `test_m5a` ✓ |
| M6 重排 位级 | 每原子定点累加 + Q34.30 定点能量(项序无关) | 同 | `test_m6` ✓ |
| M7/M8 列表无关 位级 | 超集列表 + 超 cutoff 贡献精确 0 | 同(柱5) | `test_m7_m8` ✓(实测抓到并集构建漏 MIC 的真 bug) |
| M15 run-to-run | 纯函数,无状态 | 定点铺展(柱2;A100 实测 fp64 atomic 连 run-to-run 都失败,`experiments/e0`) | `test_a3_pme_matches_naive`(确定性逐次) |
| 饱和+粘滞 | 量化前域溢出检测(B10) | 同,replay 边界检查 | `test_saturation_sticky_fires` ✓(实测触发) |
| A1 对齐 OpenMM 1e-10 | 全部件对齐 Reference 平台(KE、½k 键角、−phase 二面角、erf 例外、OpenMM 尾部公式、MIC/exceptionsUsePeriodic) | 语义继承;GPU 实现以本参考为 oracle | `test_a1_*` ✓(11 体系) |
| A3 解析 | Madelung 1.7476@2e-3、PME↔朴素 Ewald 3e-6 | 同 | `test_a3_*` ✓ |
| ingest 门控 G1–G4 | 虚位点/未钉 PME/包络/max-force 拒绝 | 同 | `test_gate_*` ✓ |
| IR round-trip | IR→OpenMM→逐 Force 对比 | 同 | `test_ir_roundtrip` ✓ |
| delta debugging | 贪心原子删除缩小 | 同 | `test_delta_debug` ✓(合成 bug→2 原子) |

## 参考 implementation 期间抓到并修复的缺陷(审计记录)

1. LJ/库仑力多一个 1/r 因子(直空间因 cutoff 外无对而长期未暴露——
   M 系列测试的价值实证)
2. 并集列表构建漏最小映像(M7 抓到:raw 2.97 nm 的对 MIC 后 0.77 nm
   在 cutoff 内)
3. Q24.40 溢出检测在 cast 之后(int64 回绕成垃圾才检查)——B10 的
   「可复现静默反号力」现场版,改为量化前域检测
4. OpenMM 尾部修正:类对自包含 + N(N+1)/2 归一 + **含 r⁻¹² 积分项**
   + 1/V(源码与 wheel 行为不一致处以行为为准,I-012 式实测锚定)
5. exceptionsUsePeriodic 是真语义参数(ingest 漏读)
6. M5a 的二进制可表示性前提(Q-009 文档登记)

## 与 spec 的两处语义补全(回写 v1.1.2 候选)

- **energy_accum 落地为 Q34.30**(v1.1.1 允许 f64 树或 Q34.30;参考
  实现证明只有定点能量使 M6 能量位级——f64 树是项序相关的)
- M5a 前提登记(见 Q-009 §5)

## 待窗口项(E0 时序类,不阻塞 M0 语义验收)

E0a(OpenMM mixed/double)、E0b-FFT/direct(η 矩阵)、E0d(f32 杠杆)——
A100 看门狗已部署,结果落地后回填 Q-002/Q-004b/c。
