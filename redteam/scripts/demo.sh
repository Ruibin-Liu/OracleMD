#!/usr/bin/env bash
# 端到端演示:全脚本链路(无 LLM 也能跑通),标注真实工作流中 delegate 的位置。
#   S1 mutate -> S2 run(仅基线)-> [S3 白盒轮=LLM] -> S2 run(并入红队测试)
#   -> S4 黑盒轮=LLM(此处模拟) -> S5 table -> E1-E6 verify
set -euo pipefail
cd "$(dirname "$0")/.."

N=$(python3 scripts/state.py next-round)
echo "==================== ROUND $N ===================="

echo "--- S1 变异注入(种子来自雷区库 + 2 个通用变异)"
python3 scripts/mutate.py --round "$N"

echo
echo "--- S2 运行(仅蓝队基线)—— 弱断言的域会留下存活种子"
python3 scripts/run_tests.py --round "$N" --include baseline
python3 scripts/coverage_table.py --round "$N" >/dev/null
cp "rounds/$N/table.md" "rounds/$N/table_after_baseline.md"
echo "    [基线后表已存: rounds/$N/table_after_baseline.md]"

echo
echo "--- S3 白盒轮(真实流程: bg_delegate, route=config.routes.whitebox)"
python3 scripts/pack.py whitebox "$N"
echo "    [demo 跳过 LLM: tests/whitebox/ 预置击杀测试,针对存活种子 m004]"

echo
echo "--- S4 黑盒轮(真实流程: bg_delegate 冷启动, route=config.routes.blackbox, 模型必须≠白盒)"
python3 scripts/pack.py blackbox "$N"
mkdir -p "rounds/$N/blackbox/gaps"
cat > "rounds/$N/blackbox/gaps/gap-001.md" <<'EOF'
# gap-001  (SIMULATED — 真实流程由黑盒 delegate 产出,harness 落盘)
domain: conditional_approximations
spec_ref: 雷区 005 全部条目
what_is_missing: 倒空间对电荷二次依赖的网格/插值阶数选择规则无定量规格,无法写可判定断言;需先补推导或硬编码为前置条件
EOF
echo "    [demo 模拟黑盒: tests/blackbox/ 预置 spec 测试 + gap-001]"

echo
echo "--- S2' 全量重跑(基线 + 黑盒 + 白盒测试)"
python3 scripts/run_tests.py --round "$N" --include all

echo
echo "--- S5 度量: 每周表 + 独立发现率"
python3 scripts/coverage_table.py --round "$N"

echo
echo "--- E1–E6 约束校验"
python3 scripts/verify.py

echo
echo "==================== ROUND $N DONE ===================="
echo "产物: rounds/$N/{mutants,results.json,alive.md,packs,table.md,table_after_baseline.md}"
