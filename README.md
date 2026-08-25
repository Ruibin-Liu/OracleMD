# OracleMD

LLM 生成 MD/自由能计算代码的双红队验证机制。

- `red_and_white.md` — 设计讨论(原始输入,why)
- `WORKFLOW.md` — **可执行工作流**(how:状态机、E1–E6 强制约束、KPI、节奏)
- `redteam/` — harness(脚本 + 雷区库 + 演示桩;README 有快速开始)
- skill `oraclemd-redteam` — pi 侧执行规程(`~/.pi/agent/skills/oraclemd-redteam/`)

演示(无 LLM,全脚本链路):

```bash
cd redteam && bash scripts/demo.sh
```

接入真实代码前的四项初始决策见 `WORKFLOW.md` §10(填路由、认域、定节奏、换桩)。
