#!/usr/bin/env python3
"""测试运行器(S2):击杀矩阵、捕获率、存活清单、状态机推进。

用法: run_tests.py --round N [--include baseline|baseline,blackbox|all]
判定: 测试文件在干净实现上必须全通过(E5),否则标记 invalid 并退出击杀统计;
      变异体被某测试文件杀死 = 该文件干净时通过、变异时失败。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
INCLUDES = ["baseline", "blackbox", "whitebox"]  # tests/ 下的来源子集


def test_files(tests_root: pathlib.Path, include: list[str]):
    files = []
    for prov in INCLUDES:
        if prov not in include:
            continue
        if prov == "baseline":
            files += [(p, "baseline") for p in sorted(tests_root.glob("test_*.py"))]
        else:
            files += [(p, prov) for p in sorted((tests_root / prov).glob("test_*.py"))]
    return files


def run_one(code_dir: str, test_file: str):
    proc = subprocess.run(
        [sys.executable, str(HERE / "run_single.py"), code_dir, test_file],
        capture_output=True, text=True, timeout=60,
    )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"file": test_file, "failures": ["<crash>"], "traceback": proc.stderr[-500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--src", default="src")
    ap.add_argument("--tests", default="tests")
    ap.add_argument("--rounds-dir", default="rounds")
    ap.add_argument("--include", default="all",
                    help="逗号分隔: baseline,blackbox,whitebox 或 all")
    args = ap.parse_args()
    round_dir = pathlib.Path(args.rounds_dir) / args.round
    mutants_root = round_dir / "mutants"

    include = INCLUDES if args.include == "all" else args.include.split(",")
    tf = test_files(pathlib.Path(args.tests), include)
    if not tf:
        print("[run] no test files selected", file=sys.stderr)
        return 1

    # 基线:干净实现必须全通过(E5)
    baseline, invalid = {}, []
    for p, prov in tf:
        r = run_one(str(pathlib.Path(args.src).resolve()), str(p.resolve()))
        rel = str(p.relative_to(pathlib.Path(args.tests)))
        baseline[rel] = r["failures"]
        if r["failures"]:
            invalid.append(rel)
    baseline_clean = not invalid

    # 击杀矩阵
    manifest = json.loads((mutants_root / "manifest.json").read_text())
    matrix, killed_by = {}, {}
    for m in manifest["mutants"]:
        mid = m["id"]
        if "error" in m:
            matrix[mid], killed_by[mid] = {}, []
            continue
        row, kb = {}, []
        for p, prov in tf:
            rel = str(p.relative_to(pathlib.Path(args.tests)))
            if rel in invalid:
                continue
            r = run_one(str((mutants_root / mid).resolve()), str(p.resolve()))
            kills = bool(r["failures"])
            row[rel] = kills
            if kills:
                kb.append(rel)
        matrix[mid] = row
        killed_by[mid] = kb

    # 汇总
    seeded_ids = [m["id"] for m in manifest["mutants"] if m.get("seeded") and "error" not in m]
    meta = {m["id"]: m for m in manifest["mutants"]}
    alive_seeded = [mid for mid in seeded_ids if not killed_by[mid]]
    seeded_killed = [mid for mid in seeded_ids if killed_by[mid]]
    all_ids = [m["id"] for m in manifest["mutants"] if "error" not in m]
    per_domain = {}
    for mid in seeded_ids:
        d = meta[mid]["domain"]
        s = per_domain.setdefault(d, {"seeded": 0, "killed": 0})
        s["seeded"] += 1
        s["killed"] += bool(killed_by[mid])

    results = {
        "round": int(args.round),
        "include": include,
        "baseline_clean": baseline_clean,
        "invalid_tests": invalid,
        "baseline_failures": baseline,
        "kill_matrix": matrix,
        "killed_by": killed_by,
        "per_domain": per_domain,
        "summary": {
            "total": len(all_ids),
            "seeded_total": len(seeded_ids),
            "seeded_killed": len(seeded_killed),
            "seeded_alive": len(alive_seeded),
            "capture_rate_seeded": round(len(seeded_killed) / len(seeded_ids), 3) if seeded_ids else None,
        },
        "alive_seeded": alive_seeded,
    }
    (round_dir / "results.json").write_text(json.dumps(results, indent=2))

    # 存活清单(白盒红队任务源)
    lines = [f"# Round {args.round} — 存活种子变异体(白盒红队任务)", ""]
    for mid in alive_seeded:
        m = meta[mid]
        lines += [f"## {mid} · {m['domain']}", f"- {m['desc']}",
                  f"- 变异位置: {m['file']} (sha {m['after_sha']})", ""]
    (round_dir / "alive.md").write_text("\n".join(lines))

    # 状态机推进
    state_path = pathlib.Path("state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state["round"] = int(args.round)
    state["state"] = "white_round" if alive_seeded else "idle"
    state_path.write_text(json.dumps(state, indent=2))

    s = results["summary"]
    print(f"[run] round {args.round} include={include} baseline_clean={baseline_clean} | "
          f"seeded capture {s['seeded_killed']}/{s['seeded_total']}"
          f" ({s['capture_rate_seeded']}) | alive: {alive_seeded or '—'}"
          f" | state -> {state['state']}")
    if not baseline_clean:
        print(f"[run] E5 VIOLATION — invalid tests: {invalid}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
