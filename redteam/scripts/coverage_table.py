#!/usr/bin/env python3
"""度量与每周表(S5):物理域 × 覆盖状态 + 双红队独立发现率。

列(与 WORKFLOW.md §6 一致): 已覆盖 / 未覆盖可测 / 未覆盖无oracle / 双红队皆盲
状态规则(按域,基于种子变异体与 gap 标记):
  gap 标记存在            -> 未覆盖无 oracle(唯一需要你决策的两列之一)
  种子全部被杀            -> 已覆盖
  部分被杀(余量可测)      -> 未覆盖可测(白盒任务自动流转)
  种子全存活且无 gap      -> 双红队皆盲(硬编码进雷区,不许靠"多测几次"自欺)
独立发现率 = |W∩B| / |W∪B|,W/B 为白盒/黑盒来源测试杀死的种子集合。
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re


def load_kill_sets(results: dict):
    w, b = set(), set()
    for mid, killers in results["killed_by"].items():
        for k in killers:
            if k.startswith("whitebox/"):
                w.add(mid)
            elif k.startswith("blackbox/"):
                b.add(mid)
    return w, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--rounds-dir", default="rounds")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    round_dir = pathlib.Path(args.rounds_dir) / args.round
    results = json.loads((round_dir / "results.json").read_text())
    config = json.loads(pathlib.Path(args.config).read_text())
    mutants = json.loads((round_dir / "mutants" / "manifest.json").read_text())["mutants"]
    meta = {m["id"]: m for m in mutants}

    gaps = []
    gaps_dir = round_dir / "blackbox" / "gaps"
    if gaps_dir.exists():
        for g in sorted(gaps_dir.glob("*.md")):
            d = re.search(r"domain:\s*(\S+)", g.read_text())
            if d:
                gaps.append({"id": g.stem, "domain": d.group(1)})

    W, B = load_kill_sets(results)
    seeded = [m["id"] for m in mutants if m.get("seeded") and "error" not in m]
    inter, union = W & B, W | B
    indep = round(len(inter) / len(union), 3) if union else None

    rows, json_rows = [], []
    for domain in config["domains"]:
        ids = [m for m in seeded if meta[m]["domain"] == domain]
        d_gaps = [g["id"] for g in gaps if g["domain"] == domain]
        killed = [m for m in ids if results["killed_by"].get(m)]
        w_d = {m for m in ids if m in W}
        b_d = {m for m in ids if m in B}
        if not ids and not d_gaps:
            state, note = "已覆盖", "无种子(未接线)"
        elif d_gaps:
            state, note = "未覆盖无oracle", f"gap: {', '.join(d_gaps)}"
        elif ids and len(killed) == len(ids):
            state, note = "已覆盖", f"{len(killed)}/{len(ids)} 种子"
        elif killed:
            state, note = "未覆盖可测", f"存活: {sorted(set(ids) - set(killed))}"
        else:
            state, note = "双红队皆盲", f"全部 {len(ids)} 种子存活"
        if ids and not w_d and not b_d and not killed:
            note += " | 双红队均未命中 → 雷区硬编码候选"
        marker = {"已覆盖": "✅", "未覆盖可测": "⚠️",
                  "未覆盖无oracle": "⛔", "双红队皆盲": "🚨"}[state]
        action = {"已覆盖": "自动流转", "未覆盖可测": "白盒任务(alive.md)",
                  "未覆盖无oracle": "**你来决策**", "双红队皆盲": "**你来决策(硬编码)**"}[state]
        rows.append(f"| {domain} | {marker} {note} | — | {action} |")
        json_rows.append({"domain": domain, "state": state, "note": note,
                          "seeded": len(ids), "killed": len(killed)})

    md = [
        f"# Round {args.round} 每周表 · {datetime.date.today().isoformat()}",
        "",
        f"- 种子捕获率: **{results['summary']['capture_rate_seeded']}** "
        f"({results['summary']['seeded_killed']}/{results['summary']['seeded_total']})",
        f"- 白盒击杀集 W = {sorted(W) or '∅'};黑盒击杀集 B = {sorted(B) or '∅'}",
        f"- 独立发现率 |W∩B|/|W∪B| = **{indep}**"
        f"{'' if indep is None else (' (⚠️ >0.7 → 换模型)' if indep > config['thresholds']['max_independence_overlap'] else ' (健康)')}",
        "",
        "| 物理域 | 状态 | 空白标记 | 动作 |",
        "|---|---|---|---|",
        *rows,
        "",
        "> 你只处理「未覆盖无oracle」「双红队皆盲」两行;其余自动流转。决策写 decisions.md。",
    ]
    (round_dir / "table.md").write_text("\n".join(md))
    (round_dir / "table.json").write_text(json.dumps(
        {"round": int(args.round), "independence": indep,
         "W": sorted(W), "B": sorted(B), "domains": json_rows}, indent=2))

    print(f"[table] round {args.round} -> {round_dir / 'table.md'}")
    print("\n".join(md[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
