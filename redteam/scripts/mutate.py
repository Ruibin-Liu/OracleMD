#!/usr/bin/env python3
"""变异注入器(S1)。种子变异来自雷区库 `## mutation` 块;另有少量通用规则。

用法: mutate.py --round N [--src src] [--minefield lib/minefield] [--out rounds/N/mutants]
产出: rounds/N/mutants/mXXX/{源码树, meta.json} + manifest.json
约束: 任一种子块匹配失败 → 非零退出(E6 完整性)。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import shutil

# 通用(非种子)变异规则: (file_suffix, description, old, new) —— 演示规模,接真实
# 代码时换 mutmut/cosmic-ray 之类工具,但 manifest 契约保持不变。
GENERIC_MUTATIONS = [
    ("kernels.py", "generic: off-by-one on rigid-body subtraction (-6 -> -7)",
     "3 * n_atoms - 6", "3 * n_atoms - 7"),
    ("kernels.py", "generic: virial sign flip (K - W -> K + W)",
     "kinetic - virial", "kinetic + virial"),
]


def parse_mutation_blocks(text: str):
    """解析雷区条目里的 `## mutation` 块。"""
    blocks, cur = [], None
    for line in text.splitlines():
        if line.startswith("## mutation"):
            cur = {"before": [], "after": []}
            blocks.append(cur)
        elif cur is not None:
            if line.startswith("file:"):
                cur["file"] = line.split(":", 1)[1].strip()
            elif line.startswith("domain:"):
                cur["domain"] = line.split(":", 1)[1].strip()
            elif line.startswith("desc:"):
                cur["desc"] = line.split(":", 1)[1].strip()
            elif line.startswith("before:"):
                cur["_mode"] = "before"
            elif line.startswith("after:"):
                cur["_mode"] = "after"
            elif line.startswith("##"):          # 下一个小节,块结束
                cur = None
            elif line.startswith("    ") and line.strip():
                cur[cur.get("_mode", "before")].append(line[4:])
    out = []
    for b in blocks:
        if b.get("file") and b["before"] and b["after"]:
            out.append({
                "file": b["file"], "domain": b.get("domain", "unknown"),
                "desc": b.get("desc", ""), "seeded": True,
                "before": "\n".join(b["before"]) + "\n",
                "after": "\n".join(b["after"]) + "\n",
            })
    return out


def apply_one(src_root: pathlib.Path, mutation: dict, mid: str, out_root: pathlib.Path):
    """生成单个变异体目录。返回 meta 或 None(未命中)。"""
    target = src_root / mutation["file"]
    text = target.read_text()
    if mutation["before"] not in text:
        return None
    patched = text.replace(mutation["before"], mutation["after"], 1)
    if patched == text or not _parses(patched):
        return None
    mdir = out_root / mid
    if mdir.exists():
        shutil.rmtree(mdir)
    shutil.copytree(src_root, mdir)
    (mdir / mutation["file"]).write_text(patched)
    meta = {
        "id": mid, "file": mutation["file"], "domain": mutation["domain"],
        "seeded": bool(mutation.get("seeded")), "desc": mutation["desc"],
        "before_sha": hashlib.sha256(text.encode()).hexdigest()[:12],
        "after_sha": hashlib.sha256(patched.encode()).hexdigest()[:12],
    }
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src")
    ap.add_argument("--minefield", default="lib/minefield")
    ap.add_argument("--round", required=True)
    ap.add_argument("--rounds-dir", default="rounds")
    args = ap.parse_args()

    src_root = pathlib.Path(args.src)
    out_root = pathlib.Path(args.rounds_dir) / args.round / "mutants"
    out_root.mkdir(parents=True, exist_ok=True)

    mutations = []
    for mf in sorted(pathlib.Path(args.minefield).glob("*.md")):
        if mf.name == "README.md":
            continue
        mutations.extend(parse_mutation_blocks(mf.read_text()))
    n_seeded_in_lib = len(mutations)
    mutations.extend([
        {"file": f, "domain": "generic", "desc": d, "seeded": False,
         "before": old, "after": new}
        for (f, d, old, new) in GENERIC_MUTATIONS
    ])

    metas, missing = [], []
    for i, m in enumerate(mutations, 1):
        mid = f"m{i:03d}"
        meta = apply_one(src_root, m, mid, out_root)
        (metas if meta else missing).append(meta or {
            "id": mid, "file": m["file"], "domain": m["domain"],
            "seeded": m["seeded"], "desc": m["desc"], "error": "before-block not found",
        })

    manifest = {"round": int(args.round), "n_seeded_in_lib": n_seeded_in_lib,
                "n_mutants": len(metas), "mutants": metas}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    seeded_alive_entries = [m for m in metas if m["seeded"]]
    print(f"[mutate] round {args.round}: {len(metas)} mutants "
          f"({len(seeded_alive_entries)} seeded) -> {out_root}")
    if missing:
        print("[mutate] E6 VIOLATION — mutation blocks that did not match:")
        for m in missing:
            print(f"         {m['id']} {m['file']} :: {m['desc']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
