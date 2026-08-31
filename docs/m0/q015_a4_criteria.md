# Q-015 — A4 判据定标(FreeSolv 水合 + JACS set RBFE 的 RMSE 阈值与置信区间)

**登记**:spec §14.1 Q-015(原 UNSOURCED,阻塞 M4);本文给出阈值、置信区间与统计方法的文献来源。
单位:门值双标 kcal/mol(文献口径)/ kJ/mol(spec 单位制),换算 1 kcal/mol = 4.184 kJ/mol。

## 1. 文献带(全部可核验来源)

### 1.1 JACS set RBFE(Wang 2015 八靶,330 扰动)

| 来源 | 指标 | 值(kcal/mol) | 备注 |
|---|---|---|---|
| Wang 2015(FEP+, OPLS2.1)| AUE | **0.925 ± 0.041**(3.87±0.17 kJ/mol)| 经 Gapsys 2020 全文转引(ACS 原文 403 未直接核) |
| Harder 2016(OPLS3)| AUE | 0.803 ± 0.036(3.36±0.15 kJ/mol)| 同上转引 |
| Gapsys 2020 复跑(FEP+, OPLS3, 3 重复)| AUE | 0.875 ± 0.033;Pearson 0.69±0.03 | 482 扰动/13 靶全集 |
| Gapsys 2020(pmx consensus, 开源)| AUE | 0.856–0.884;Pearson 0.63±0.03 | GAFF+CGenFF 共识 |
| Gapsys 2020 转引(Amber18 TI)| AUE | ≈1.17(4.9 kJ/mol);r=0.48 | 显著劣于上两者 |
| Transformato 2022(OpenMM/CGenFF, 开源)| **RMSE** | **1.18,CI₉₅ [0.98; 1.38]**;MAE 0.87 [0.72;1.02];R 0.57 [0.36;0.71];ρ 0.48 | 76 扰动/5 靶;CI = percentile bootstrap |

逐靶 RMSE(ΔΔG,kcal/mol;Transformato Table 2 汇总,FEP+ 列为 Wang 2015 发表值的最短路径重组):
JNK1:FEP+ 1.02 / pmx 0.95 / AMBER-TI 1.45;CDK2:1.16 / 1.13 / 1.16;TYK2:0.95 / 1.61 / 1.29;
开源最差靶 TYK2(pmx 1.61、Transformato 1.74、pmx ΔG 1.87)——文献带外沿 ≈ **1.9**。
Transformato 全量 68.4% 扰动落在 ±1 kcal/mol 内。

### 1.2 FreeSolv 水合(数据库 v0.52,2017-06-11 快照)

第一方重算(MobleyLab/FreeSolv `database.txt` 自带 GAFF/AM1-BCC/TIP3P 计算列,n=642,2026-08-28):

| 指标 | 值(kcal/mol) |
|---|---|
| RMSE | **1.542** |
| MUE | 1.114(RMSE/MUE = **1.38**) |
| 偏置 ME | +0.317 |
| Pearson r | 0.933 |
| \|err\| ≤ 1 kcal/mol | 52.6% |

实验不确定度列(d_expt):中位 0.60,但 **459/642(71%)是 0.60 缺省赋值**(非缺省 183 个:中位 0.20,均值 0.485,最大 1.93)。⇒「实验噪声地板 0.6」的说法对 FreeSolv 不成立为实测依据,**不得作判据锚点**。

## 2. A4 判据(冻结提案,带量纲)

统计方法(两门共用):**percentile bootstrap,10⁴ 重采样,单元 = 独立分子/扰动**(非轨迹帧、非 u_kn 样本,防自相关假精度);bootstrap RNG = counter RNG(柱 3),种子入 manifest;MBAR/TI 估计器自身的统计不确定度与数据集重采样 CI **分开报告**,不合并。

### A4-R 水合门(FreeSolv 冻结子集;n ≥ 50,子集+参考值哈希入 manifest)

| 门 | 阈值 | 依据 |
|---|---|---|
| 地板:RMSE ≤ | **1.9 kcal/mol(7.95 kJ/mol)** | 第一方 GAFF 基线 1.542 × 1.23 |
| CI 门:bootstrap95% 上界 ≤ | **2.2 kcal/mol(9.20 kJ/mol)** | 基线 × 1.43(允许抽样噪声,不允许系统性劣于 2014 年公开流水线) |
| 偏置门:\|ME\| ≤ | **0.6 kcal/mol(2.51 kJ/mol)** | 抓单位/参考态/净电荷接线错(雷区 #4 类);基线偏置 +0.317 |
| 相关门:r ≥ | 0.8 | 基线 0.933 |

### A4-B RBFE 门(JACS set 冻结子集;n ≥ 30 扰动 ∧ ≥ 3 靶;扰动集+λ 协议冻结入 manifest)

| 门 | 阈值 | 依据 |
|---|---|---|
| 整体地板:RMSE ≤ | **1.35 kcal/mol(5.65 kJ/mol)** | 开源第一梯整体 1.18 [0.98;1.38](Transformato),点估计不劣于其 CI 上沿 |
| 整体 CI 门:上界 ≤ | **1.60 kcal/mol(6.69 kJ/mol)** | ≈ Amber18-TI 换算带 1.58 的上沿;停在 TI 水平即不过 |
| 单靶哨兵:无单靶 RMSE > | **2.0 kcal/mol(8.37 kJ/mol)** | 文献开源最差靶带外沿 1.87;防「一靶爆炸被平均掩盖」 |
| 相关门:r ≥ 0.5 ∧ CI₉₅ 下界 ≥ 0.25 | | 文献带 0.57 [0.36;0.71](开源)~ 0.69(FEP+) |

归因段专用(报告但无斩杀):within-1kcal 分数(文献带 52.6%–68.4%)、ME 偏置、Spearman ρ。
地板抓谎言,归因抓意外——与 §10.3 同构。

## 3. 换算与假设登记

- **RMSE/MUE ≈ 1.35–1.40**:两独立实测一致(FreeSolv 第一方 1.38、Transformato 1.36)。仅用于跨文献比较(Wang/Harder/Gapsys 报 AUE),**不进门**。
- Wang/Harder 的 AUE 经 Gapsys 2020(开放获取全文 XML,PMC8145179)转引;ACS/RSC 原文表格未直接核(403)。已用两个独立转引源交叉(Gapsys 全文 + Transformato Table 2)。
- FreeSolv 0.60 不确定度缺省赋值问题(§1.2)如实登记;偏置门阈值与此解耦。

## 4. 失效触发(回写 spec Q-015 行)

换力场;换实验参考版本(数据库版本);子集/扰动集变更(= manifest 哈希变 + 重跑);n 或靶数跌破下限;bootstrap 方法变更。

## 5. 已知限制

1. TYK2 类系统的 CGenFF 参数缺陷是文献登记的系统效应(Transformato §3.4.2:环丙基水合 CGenFF +2.17 vs 实验 +0.75 kcal/mol)。若引擎用 OPLS 系 FF,该靶偏离归因应指向 FF 而非引擎——单靶哨兵门抓的是引擎级灾难,不是 FF 级缺陷。
2. 阈值数字是「开源第一梯 ± 抽样余量」的工程定标,不是物理常数;M4 首轮实测后允许按 §14.6 流程修订(触发行声明 + 回查),但**只许收紧或换锚,不许因首轮不过而放宽**(判据先于数据冻结——与雷区 #014 instantiation gate 同一原则)。

## 6. 来源清单

- Mobley & Guthrie, J Comput Aided Mol Des 28:723(2014),DOI 10.1007/s10822-014-9747-x;数据库 v0.52:github.com/MobleyLab/FreeSolv(`database.txt`,重算脚本见 §1.2 描述)
- Wang et al., JACS 137:2695(2015),DOI 10.1021/ja512751q
- Harder et al., JCTC 12:281(2016),DOI 10.1021/acs.jctc.5b00864(经 Gapsys 转引)
- Gapsys et al., Chem Sci 11:1140(2020),DOI 10.1039/C9SC03754C,PMC8145179(全文 XML 本地:/tmp 拉取,Europe PMC REST)
- Wieder et al., Front Mol Biosci 9:954638(2022),DOI 10.3389/fmolb.2022.954638

**结论**:Q-015 清零(2026-08-28),A4 判据带量纲落地,M4 阻塞解除(M4 执行仍等 GPU 窗口)。
