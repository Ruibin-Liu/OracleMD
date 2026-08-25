# redteam/ — 双红队 harness

`red_and_white.md` 的可执行落地。规程:`../WORKFLOW.md`;pi 侧执行:skill `oraclemd-redteam`。

## 布局

```
config.json          路由(蓝/白/黑,满足 E1)、域清单、阈值、节奏
state.json           状态机(单一事实源)
lib/                 黑盒可见区:spec.md, interfaces.md, brief_*.md, minefield/
src/                 蓝队代码(演示桩 kernels.py)
tests/               测试套件,来源即溯源: test_*.py(蓝) whitebox/ blackbox/
scripts/             mutate / run_tests / pack / coverage_table / verify / state
rounds/<N>/          每轮: mutants/ results.json alive.md packs/ blackbox/ table.md
```

## 快速开始(演示桩,无 LLM)

```bash
cd redteam && bash scripts/demo.sh
```

## 真实一轮(LLM 步骤由 pi 按 skill 执行)

```bash
N=$(python3 scripts/state.py next-round)
python3 scripts/mutate.py  --round "$N"
python3 scripts/run_tests.py --round "$N" --include all
# 若 state=white_round -> delegate(白盒, route A)按 lib/brief_whitebox.md 产击杀测试
#   harness 落盘 tests/whitebox/ 后重跑 run_tests,击杀结果回喂迭代(≤2)
# 每周/cold start -> pack.py blackbox "$N";delegate(黑盒, route B≠A)按 brief_blackbox.md
#   产出 tests + gaps -> harness 落盘 rounds/$N/blackbox/{tests,gaps}/
python3 scripts/coverage_table.py --round "$N"
python3 scripts/verify.py     # E1–E6
```

## 备注

- `mutate.py` 的通用变异规则是演示级;接真实代码后换 mutmut/cosmic-ray,
  manifest 契约(`id/domain/seeded/desc`)保持不变,下游无需改。
- 雷区库条目的公式为 DEMO 桩值;接入真实代码时替换为带出处的公式,
  **适用条件字段必须保留**——那是雷区库最值钱的部分。
