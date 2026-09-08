#!/usr/bin/env python3
"""开放项 7(a) 落地前置:对间距收缩与列表失效事件实测(v2)。

关键修正(v1 只追带内对):列表失效的判据是 **ref > rc+skin 的对在窗口内
收缩进 rc**。故分两带跟踪:
  - 带内 (rc, rc+skin]:margin 消耗(shrink 分布)
  - 带外 (rc+skin, rc+0.45]:失效事件(shrink > ref − rc 即漏对)

附带 H 摆幅(disp_H − disp_O,自身 O 参照)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import openmm  # noqa: E402
from openmm import unit as u  # noqa: E402

from run_q006 import build_water, hmr  # noqa: E402

R = Path(__file__).resolve().parent / "results"
RC, SKIN = 1.0, 0.28
OUTER = RC + 0.45


def run(box=2.2, dt_fs=4.0, meas_ps=100.0, window_fs=100.0, seed=7):
    sys_, pos, grid = build_water(box)
    hmr(sys_)
    integ = openmm.LangevinMiddleIntegrator(300 * u.kelvin, 1.0 / u.picosecond,
                                            dt_fs * u.femtosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(sys_, integ, openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(pos * u.nanometer)
    openmm.LocalEnergyMinimizer.minimize(ctx, 1.0, 200)
    ctx.setVelocitiesToTemperature(300 * u.kelvin)
    integ.setStepSize(2.0 * u.femtosecond)
    integ.step(5000)
    integ.setStepSize(dt_fs * u.femtosecond)
    integ.step(7500)
    masses = np.array([sys_.getParticleMass(i).value_in_unit(u.dalton)
                       for i in range(sys_.getNumParticles())])
    O_idx = np.where(masses > 3.5)[0]
    H_of_O = {3 * w: (3 * w + 1, 3 * w + 2) for w in range(len(masses) // 3)}
    L = box
    dt = dt_fs * 1e-3
    w_len = int(round(window_fs * 1e-3 / dt))   # window_fs(fs) → ps 再除 dt(ps)
    n_win = int(round(meas_ps / (window_fs * 1e-3)))
    inb, outb, invalid, swings, dispH = [], [], [], [], []
    t0 = time.time()
    print("setup done, entering window loop", flush=True)
    for w in range(n_win):
        _t = {}
        _ts = time.time()
        x0 = ctx.getState(getPositions=True).getPositions(asNumpy=True) \
            .value_in_unit(u.nanometer).copy()
        d = x0[:, None, :] - x0[None, :, :]
        d -= np.round(d / L) * L
        dist = np.sqrt((d * d).sum(-1))
        iu, ju = np.triu_indices(len(x0), k=1)
        dd = dist[iu, ju]
        mi = (dd > RC) & (dd <= RC + SKIN)
        mo = (dd > RC + SKIN) & (dd <= OUTER)
        bi, bj = iu[mi], ju[mi]
        oi, oj = iu[mo], ju[mo]
        _t['list'] = time.time() - _ts
        _ts = time.time()
        for s in range(w_len):
            integ.step(1)
        _t['steps'] = time.time() - _ts
        _ts = time.time()
        x1 = ctx.getState(getPositions=True).getPositions(asNumpy=True) \
            .value_in_unit(u.nanometer)


        def mind(ii, jj):
            d1 = x1[ii] - x1[jj]
            d1 -= np.round(d1 / L) * L
            return np.sqrt((d1 * d1).sum(-1))


        si = (dd[mi] - mind(bi, bj))          # 带内 shrink
        so = (dd[mo] - mind(oi, oj))          # 带外 shrink
        inb.append(si.max() if len(si) else 0.0)
        if len(so):
            outb.append(so.max())
            invalid.append(int((so > dd[mo] - RC).sum()))  # 漏对事件
        disp = x1 - x0
        disp -= np.round(disp / L) * L
        _t['track'] = time.time() - _ts
        _ts = time.time()
        sw = 0.0
        for o, (h1, h2) in H_of_O.items():
            for h in (h1, h2):
                s_h = np.linalg.norm(disp[h] - disp[o])
                sw = max(sw, s_h)
        swings.append(sw)
        dispH.append(np.sqrt((disp[O_idx] ** 2).sum(-1)).max())
        if (w + 1) % 200 == 0 or w < 5:
            print(f"  win {w+1}/{n_win}  " + " ".join(f"{k}={v:.2f}s" for k, v in _t.items()), flush=True)
    def pct(v):
        v = np.asarray(v)
        return {f"p{p}": float(np.percentile(v, p)) for p in (50, 95, 99.9)} | \
            {"max": float(v.max())}

    out = dict(
        box=box, n_win=n_win, window_fs=window_fs, meas_ps=meas_ps,
        shrink_inband=pct(inb), shrink_outband=pct(outb),
        missed_pairs_total=int(np.sum(invalid)),
        windows_with_miss=int(np.sum(np.array(invalid) > 0)),
        swing=pct(swings), disp_heavy=pct(dispH),
    )
    return out


if __name__ == "__main__":
    print("pair_shrink v2 starting", flush=True)
    res = run()
    print(json.dumps(res, indent=1))
    (R / "pair_shrink.json").write_text(json.dumps(res, indent=1))
