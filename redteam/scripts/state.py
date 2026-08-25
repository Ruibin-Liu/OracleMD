#!/usr/bin/env python3
"""state.json / config.json 轮次号维护。

  state.py get                 # 打印状态
  state.py set <state>         # 设置状态(idle|white_round|black_round|user_decision)
  state.py next-round          # 轮次 +1(写 config.json 与 state.json),打印新轮号
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATES = {"idle", "white_round", "black_round", "user_decision"}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "get"
    cfg = json.loads((ROOT / "config.json").read_text())
    st_path = ROOT / "state.json"
    st = json.loads(st_path.read_text()) if st_path.exists() else {"round": cfg["round"], "state": "idle"}

    if cmd == "get":
        print(json.dumps(st))
    elif cmd == "set":
        val = sys.argv[2]
        assert val in STATES, f"unknown state {val}; allowed {STATES}"
        st["state"] = val
        st_path.write_text(json.dumps(st, indent=2))
        print(f"[state] -> {val}")
    elif cmd == "next-round":
        cfg["round"] += 1
        (ROOT / "config.json").write_text(json.dumps(cfg, indent=2))
        st["round"] = cfg["round"]
        st["state"] = "idle"
        st_path.write_text(json.dumps(st, indent=2))
        print(cfg["round"])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
