#!/usr/bin/env python3
"""组包器(S3/S4 前置):实现信任边界的物理层。

  pack.py whitebox N  -> rounds/N/packs/whitebox/  (src+tests+存活清单+diff+任务书)
  pack.py blackbox N  -> rounds/N/packs/blackbox/  (仅 lib/ + 轮次信息 —— E2 由构造保证)
manifest 记录每个文件 sha256,供 verify.py 复核。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys


def manifest_for(pack_dir: pathlib.Path, allowed_prefixes: list[str] | None):
    files = []
    for p in sorted(pack_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files.append({"path": str(p.relative_to(pack_dir)),
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                          "bytes": p.stat().st_size})
    purity = True if allowed_prefixes is None else all(
        f["path"].startswith(tuple(allowed_prefixes)) for f in files)
    return {
        "allowed_prefixes": allowed_prefixes,
        "n_files": len(files),
        "files": files,
        "purity": purity,
    }


def copy_tree(src: pathlib.Path, dst: pathlib.Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.rglob("*")):
        if p.is_file():
            rel = p.relative_to(src)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)


def main() -> int:
    kind, round_no = sys.argv[1], sys.argv[2]
    round_dir = pathlib.Path("rounds") / round_no
    lib = pathlib.Path("lib")

    if kind == "whitebox":
        pack = round_dir / "packs" / "whitebox"
        if pack.exists():
            shutil.rmtree(pack)
        copy_tree(pathlib.Path("src"), pack / "src")
        copy_tree(pathlib.Path("tests"), pack / "tests")
        for f in ["alive.md", "results.json"]:
            src = round_dir / f
            if src.exists():
                shutil.copy2(src, pack / f)
        shutil.copy2(lib / "brief_whitebox.md", pack / "BRIEF.md")
        shutil.copy2(lib / "spec.md", pack / "spec.md")
        # diff(有 git 用 git,否则标注快照)
        g = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True)
        (pack / "diff.txt").write_text(
            g.stdout if g.returncode == 0 and g.stdout.strip()
            else "(no git diff — full snapshot under src/)")
        m = manifest_for(pack, None)   # 白盒包本来就含 src/tests,E2 不适用
        m["route_required"] = "config.routes.whitebox"
    elif kind == "blackbox":
        pack = round_dir / "packs" / "blackbox"
        if pack.exists():
            shutil.rmtree(pack)
        pack.mkdir(parents=True)
        shutil.copy2(lib / "spec.md", pack / "spec.md")
        shutil.copy2(lib / "interfaces.md", pack / "interfaces.md")
        shutil.copy2(lib / "brief_blackbox.md", pack / "BRIEF.md")
        copy_tree(lib / "minefield", pack / "minefield")
        (pack / "round_info.json").write_text(json.dumps(
            {"round": int(round_no), "kind": "blackbox", "cold_start": True}, indent=2))
        # E2:只允许 lib 派生文件 —— 不 copy src/、tests/、rounds/
        m = manifest_for(pack, ["spec.md", "interfaces.md", "BRIEF.md",
                                "minefield/", "round_info.json"])
        m["route_required"] = "config.routes.blackbox"
    else:
        print("usage: pack.py whitebox|blackbox N", file=sys.stderr)
        return 2

    (pack / "manifest.json").write_text(json.dumps(m, indent=2))
    status = "PURE" if m["purity"] else "CONTAMINATED"
    print(f"[pack] {kind} r{round_no}: {m['n_files']} files -> {pack} [{status}]")
    return 0 if m["purity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
