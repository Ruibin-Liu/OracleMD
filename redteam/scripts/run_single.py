#!/usr/bin/env python3
"""在单个代码目录上运行单个测试文件(run_tests.py 的子进程单元)。"""
from __future__ import annotations

import json
import pathlib
import sys
import traceback


def main() -> int:
    code_dir, test_file = sys.argv[1], sys.argv[2]
    sys.path.insert(0, code_dir)
    ns: dict = {}
    try:
        exec(compile(pathlib.Path(test_file).read_text(), test_file, "exec"), ns)
    except Exception:
        print(json.dumps({"file": test_file, "failures": ["<import>"],
                          "traceback": traceback.format_exc(limit=3)}))
        return 1
    failures = []
    for name in sorted(ns):
        if name.startswith("test_") and callable(ns[name]):
            try:
                ns[name]()
            except Exception:
                failures.append(name)
    print(json.dumps({"file": test_file, "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
