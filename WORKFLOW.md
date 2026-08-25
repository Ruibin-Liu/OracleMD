# 双红队工作流(可执行版)

> 本文把 `red_and_white.md` 的设计约束转成**可执行、可校验、可复盘**的工作流。
> 落地代码在 `redteam/`,pi 侧执行规程在 skill `oraclemd-redteam`。
> 原则:**凡是"必须遵守"的约束,一律由 harness 机器检查(E1–E6),不靠自觉。**

---

## 0. 一句话版本

```
蓝队提交代码 → 变异注入器生成已知 bug → 跑测试 → 存活变异体
  ├─ → 白盒红队(看实现,专杀存活变异体)——每次提交触发
  └─ → 黑盒红队(只看物理规格,冷启动)——每周触发
→ 汇总:物理域 × 覆盖状态表 → 你只对「未覆盖无 oracle」「双红队皆盲」两列做决策
```

## 1. 角色与信任边界

| 角色 | 是谁 | 能看到什么 | 产出 |
|---|---|---|---|
| **蓝队** | 开发 agent(主会话) | 一切 | `redteam/src/` 代码 |
| **Harness** | 主会话 + `redteam/scripts/` | 一切(唯一全知者) | 轮次产物、表、校验 |
| **白盒红队** | bg_delegate,pinned 模型 A | 实现 + diff + 现有测试 + 存活变异体 | 击杀测试 |
| **黑盒红队** | bg_delegate,pinned 模型 B(≠A) | **只有**:规格 + 接口签名 + 雷区库 | ①独立测试 ②规格空白标记 |
| **变异注入器** | 纯脚本,无 LLM | src + 雷区库 | 变异体 + manifest |

信任边界只有一条铁律:**只有 harness 同时看到所有红队的报告;红队之间、黑盒与蓝队历史之间,零接触。**

## 2. 状态机

`redteam/state.json` 单一事实源,脚本与 agent 共同维护:

```
IDLE
 └─(蓝队提交/手动)→ MUTATE  scripts/mutate.py
MUTATE → RUN               scripts/run_tests.py
RUN   ├─ 存活种子变异体>0 → WHITE_ROUND   (agent 步骤,见 §4)
      └─ 否则 → IDLE
WHITE_ROUND → RUN(新测试并入后重跑,迭代 ≤ max_whitebox_iters)
RUN   └─ 每周到期 或 新增雷区域 → BLACK_ROUND (agent 步骤,冷启动)
BLACK_ROUND → MEASURE         scripts/coverage_table.py
MEASURE → USER_DECISION       (表已生成)
USER_DECISION → IDLE          (决策写回:雷区库/蓝队任务/测试晋升)
```

## 3. 强制约束(机器检查项)

| # | 约束(源自 red_and_white.md) | 执行机制 | 检查 |
|---|---|---|---|
| E1 | 两个红队用**不同基础模型** | `config.json` 路由 pin;delegate 必须显式 route | `verify.py` |
| E2 | 黑盒**看不到**实现/白盒输出/蓝队历史 | `pack.py` 只从 `lib/` 组包,manifest 记录 sha256 | `verify.py` 复核包纯度 |
| E3 | 黑盒每轮**冷启动**新会话 | skill 规定:每轮一个新 delegate,绝不携带往轮报告;规格空白 backlog 只进决策表,不回喂黑盒 | skill + 轮次 manifest 记录 session id |
| E4 | 黑盒产出**仅两类**:可执行测试 / 空白标记 | `blackbox/` 目录 schema 校验 | `verify.py` |
| E5 | 黑盒测试必须在**干净实现上通过**(否则是坏测试,不是发现) | runner 基线运行 | `run_tests.py` 标记 invalid |
| E6 | 种子 bug 必须带 ground truth | manifest 记录 seeded 标志 + 域 | `verify.py` 完整性 |

**冷启动规则的精确边界**(避免误伤):E3 只约束**观点性复审**——黑盒第二轮不得看到任何人的先前意见。而白盒的**变异击杀循环**喂回去的是测试执行结果(事实),不是观点,允许迭代;这正是 Meta 变异引导测试生成的机制。

## 4. 步骤契约(触发/输入/动作/输出/检查)

每轮一个目录 `redteam/rounds/<N>/`,产物如下:

| 步骤 | 触发 | 输入 | 动作 | 输出 | 检查 |
|---|---|---|---|---|---|
| S0 初始化 | 一次性 | 文献/经验 | 建雷区库 v0(5 个域)、config、规格骨架 | `lib/` | `verify.py` |
| S1 变异 | 蓝队提交或手动 | `src/` + 雷区 mutation 块 | `scripts/mutate.py` | `rounds/N/mutants/*` + manifest | 种子/非种子分类完整(E6) |
| S2 运行 | S1 后 | `tests/` 全量 + mutants | `scripts/run_tests.py` | `results.json`(击杀矩阵、捕获率)、`alive.md` | 干净实现全通过(E5) |
| S3 白盒轮 | 存活种子变异体>0 | `pack.py whitebox N`:src+diff+测试+存活清单 | delegate(模型 A)按 `lib/brief_whitebox.md` 产击杀测试;harness 写入 `tests/whitebox/` 并重跑;击杀结果回喂迭代(≤2) | 新测试 + 增量捕获率 | 断言强度由击杀自动验证,杀不掉即丢弃 |
| S4 黑盒轮 | 每周 或 新增雷区域 | `pack.py blackbox N`:**仅** spec+interfaces+minefield | delegate(模型 B)按 `lib/brief_blackbox.md`,新会话冷启动;产出写 `rounds/N/blackbox/{tests,gaps}/` | 独立测试 + 空白标记 | E2/E3/E4/E5 |
| S5 度量 | S4 后(或随时) | results.json + gaps | `scripts/coverage_table.py` | `table.md` + 独立发现率 | — |
| S6 决策 | 表生成后 | table.md | **你只处理两列**(见 §6),决策写回雷区库/蓝队任务 | `rounds/N/decisions.md` | — |

## 5. KPI 与阈值(`config.json: thresholds`)

| 指标 | 定义 | 阈值动作 |
|---|---|---|
| 种子捕获率 | 被任一测试杀死的种子变异体 / 总数 | < 0.9 → 白盒轮继续;连续两轮不动 → 升级为决策 |
| 白盒增量 | 本轮新增测试带来的捕获率增量 | ≈0 → 白盒饱和,剩余变异体转雷区 |
| 黑盒增量 | 黑盒新测试捕获的**此前已存活**种子数 | 0 且独立发现率高 → 黑盒只在做重复劳动,换分工 |
| 独立发现率 | 种子 bug 上 W∩B / W∪B(W=白盒来源测试击杀集,B=黑盒) | > 0.7 → 盲区相关严重,换模型(E1 重选);并集≫单侧 → 机制有效,维持 |
| 双盲域数 | 「双红队皆盲」列的域数 | 只许经由雷区硬编码下降,不许经由"多测几次"自欺 |

## 6. 每周表(你唯一需要看的东西)

| 物理域 | 已覆盖 | 未覆盖可测 | 未覆盖无 oracle | 双红队皆盲 | 动作 |
|---|---|---|---|---|---|
| shake_settle | ✅ 3/3 种子 | — | — | — | 自动流转 |
| net_charge | ✅ 1/2 | ⚠️ m4b | — | — | 白盒任务已生成 |
| conditional_approx | — | — | ⛔ gap-001 | — | **你来决策** |
| … | | | | ⛔ 双盲 | **你来决策(硬编码进雷区)** |

- 前两列自动流转(S3 白盒任务、测试晋升),**你只在后两列出现时行动**;
- 后两列是唯一真正需要人类判断的输入,决策三类:①补 oracle(写解析参考)②硬编码检查(推导繁琐结论简短的域,直接写成雷区断言)③接受风险并记录。

## 7. 在 pi 里的执行映射

| 工作流元素 | pi 机制 |
|---|---|
| 红队(隔离 + 路由 pin) | `bg_delegate`,显式 `route: {provider, model}`,inspect-only(读 pack、返回文本,harness 落盘执行) |
| 脚本步骤 | `bash` / `bg_run`(长跑) |
| 全过程规程 | skill `oraclemd-redteam`(`~/.pi/agent/skills/oraclemd-redteam/SKILL.md`) |
| 决策记录 | `rounds/<N>/decisions.md` + 雷区库 git 提交 |

红队 delegate 是 inspect-only 是**特性不是限制**:测试的执行与合并只能由 harness 做,这正是"报告独立提交、只在你面前汇总"的物理实现。

## 8. 节奏

- **每次蓝队提交**(触及保护域):S1→S2→(S3)。
- **每周**:S4(冷启动)→ S5 → S6。
- **雷区库新增域**:立即触发一次黑盒轮。
- **不设"感觉差不多了"**:表收敛(后两列清空或全部有决策记录)才算 M-1 完成这一层。

## 9. 验收标准(工作流"被遵守"的定义)

1. 每次触及保护域的提交都留下完整轮次目录(manifest+results+table);
2. 任一黑盒包的 manifest 经 E2 复核不含 `src/`、`rounds/`、`tests/` 的任何字节;
3. config 中白盒/黑盒模型不同,且至少一个不同于蓝队(E1);
4. 每周表按 cadence 刷新,后两列每项都有 decisions.md 记录;
5. 独立发现率每轮被计算并存档(哪怕为空)。

## 10. 留给你的初始决策(一次)

- [ ] 填 `config.json` 三条路由(蓝/白/黑,满足 E1);
- [ ] 认可 5 个初始雷区域划分,或增删;
- [ ] 确认周 cadence 与白盒迭代上限(默认 2);
- [ ] 把 `redteam/src/kernels.py`(演示桩)换成真实代码,规格 `lib/spec.md` 同步。
