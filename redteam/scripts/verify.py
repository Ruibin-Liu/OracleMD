#!/usr/bin/env python3
"""约束校验器:E1–E6。任一硬违例非零退出——工作流"被遵守"由它定义(WORKFLOW.md §9)。"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

OK, BAD = "[verify] PASS", "[verify] FAIL"
fails = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"{OK if ok else BAD}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def e1_routes(config):
    r = config["routes"]
    w, b, blue = (r[k]["model"] for k in ("whitebox", "blackbox", "blue"))
    if not (w and b):
        check("E1 双红队模型不同", True, f"DEMO 模式未填路由(w={w or '∅'}, b={b or '∅'}),接入前必填")
        return
    check("E1a 白盒≠黑盒模型", w != b, f"{w} vs {b}")
    check("E1b 至少一个红队≠蓝队", (w != blue) or (b != blue))


def e2_pack_purity(rounds_dir):
    packs = sorted(rounds_dir.glob("*/packs/blackbox"))
    if not packs:
        check("E2 黑盒包纯度", True, "尚无黑盒包")
        return
    forbidden = ("src/", "tests/", "rounds/", "diff")
    for pack in packs:
        mf = json.loads((pack / "manifest.json").read_text())
        bad = [f["path"] for f in mf["files"]
               if not f["path"].startswith(tuple(mf["allowed_prefixes"]))
               or any(x in f["path"] for x in forbidden)]
        recheck = all(
            hashlib.sha256((pack / f["path"]).read_bytes()).hexdigest() == f["sha256"]
            for f in mf["files"])
        check(f"E2 {pack.parent.parent.name}/packs/blackbox 纯度", not bad and recheck,
              f"{mf['n_files']} files" + (f" 违规: {bad}" if bad else ""))


def e4_blackbox_schema(rounds_dir):
    dirs = sorted(rounds_dir.glob("*/blackbox"))
    if not dirs:
        check("E4 黑盒产出 schema", True, "尚无黑盒报告")
        return
    for d in dirs:
        stray = [str(p.relative_to(d)) for p in d.rglob("*")
                 if p.is_file() and not (str(p.relative_to(d)).startswith("tests/")
                                         and p.suffix == ".py")
                 and not (str(p.relative_to(d)).startswith("gaps/") and p.suffix == ".md")]
        check(f"E4 {d.parent.name}/blackbox 仅 tests/*.py + gaps/*.md", not stray,
              f"多余产物: {stray}" if stray else "")


def e5_black_tests_valid(results):
    invalid = [t for t in results.get("invalid_tests", []) if t.startswith("blackbox/")]
    check("E5 黑盒测试在干净实现上通过", not invalid, f"无效: {invalid}" if invalid else "")


def e6_manifest_complete(rounds_dir):
    for mf in sorted(rounds_dir.glob("*/mutants/manifest.json")):
        m = json.loads(mf.read_text())
        errored = [x for x in m["mutants"] if "error" in x]
        seeded = [x for x in m["mutants"] if x.get("seeded")]
        check(f"E6 {mf.parent.parent.name} 种子完整性",
              not errored and len(seeded) == m["n_seeded_in_lib"],
              f"{len(seeded)}/{m['n_seeded_in_lib']}" +
              (f" 未命中: {[x['id'] for x in errored]}" if errored else ""))


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    config = json.loads((root / "config.json").read_text())
    rounds_dir = root / "rounds"
    e1_routes(config)
    e2_pack_purity(rounds_dir)
    e4_blackbox_schema(rounds_dir)
    for res in sorted(rounds_dir.glob("*/results.json")):
        e5_black_tests_valid(json.loads(res.read_text()))
    e6_manifest_complete(rounds_dir)
    print(f"\n[verify] {'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
