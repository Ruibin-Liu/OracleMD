#!/usr/bin/env python3
"""Q-006 事后分析:renewal 间隔 / 窗口越界概率 / D / 步长与尺寸标度。

数据源:results/*.npz(见 run_q006.py 的记录语义)。
- renewal:renew_steps / renew_heavy_steps(内联,越过 0.05 即回置)
- 生产窗口位移:max_d(episode 原点,2 ps 不回置)按 lag 对齐取窗口末值
  ⇒ 任意窗口长 ≤ 2 ps 的"自窗口起点位移"分布(K 窗口哨兵)
- D:msd(同 episode 原点)按 lag 平均,线性段拟合 slope/6
"""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
BOOT = 10000


def renew_stats(z, key):
    rs = z[key]
    dt = float(z["dt_fs"])
    if len(rs) == 0:
        return dict(n=0)
    rng = np.random.default_rng(0)
    bs = rs[rng.integers(0, len(rs), (BOOT, len(rs)))].mean(1) * dt
    return dict(n=int(len(rs)), mean_fs=float(rs.mean() * dt),
                median_fs=float(np.median(rs) * dt),
                p90_fs=float(np.percentile(rs, 90) * dt),
                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                per_ns=1000.0 / (rs.mean() * dt),
                per_step_prob=float(len(rs) / z["mw"].shape[0]))


def window_disp(z, lag_fs, key="max_d"):
    """max displacement at `lag_fs` since episode origin (boundary-only resync)"""
    dt = float(z["dt_fs"])
    lag = int(round(lag_fs / dt))
    md = z[key]
    ep = int(z["ep_len"])
    n_ep = len(md) // ep
    idx = np.arange(n_ep) * ep + (lag - 1)
    idx = idx[idx < len(md)]
    return md[idx]


def diffusion(z, arr_key="msd", fit_ps=(2.0, 4.5)):
    dt = float(z["dt_fs"]) * 1e-3  # ps
    msd, ep = z[arr_key], int(z["ep_len"])
    n_ep = len(msd) // ep
    m = msd[:n_ep * ep].reshape(n_ep, ep)
    lags = np.arange(ep) * dt
    prof = m.mean(0)
    sel = (lags >= fit_ps[0]) & (lags <= fit_ps[1])
    if sel.sum() < 5:
        sel = lags > lags.max() * 0.5
    slope = np.polyfit(lags[sel], prof[sel], 1)[0]
    return dict(nm2_per_ps=float(slope / 6.0), n_ep=int(n_ep),
                fit=str(fit_ps),
                msd_at_1ps=float(np.interp(1.0, lags, prof)))


def analyze(tag):
    z = np.load(R / f"{tag}.npz")
    dt = float(z["dt_fs"])
    out = dict(tag=tag, n_atoms=int(z["n_atoms"]), n_heavy=int(z["n_heavy"]),
               dt_fs=dt, hmr=bool(z["hmr"]), meas_ps=len(z["mw"]) * dt * 1e-3,
               renew_all=renew_stats(z, "renew_steps"),
               renew_heavy=renew_stats(z, "renew_heavy_steps"),
               D_all=diffusion(z, "msd"),
               D_heavy=diffusion(z, "msd_heavy"))
    # 生产窗口(K=25 ⇒ lag 100 fs;K=50 ⇒ 200 fs)
    for lag, name in ((100, "w100"), (200, "w200")):
        for key, suffix in (("max_d", ""), ("mde_heavy", "_heavy")):
            w = window_disp(z, lag, key)
            if len(w):
                out[name + suffix] = dict(n=len(w), p50=float(np.percentile(w, 50)),
                                          p95=float(np.percentile(w, 95)),
                                          p999=float(np.percentile(w, 99.9)),
                                          mx=float(w.max()),
                                          p_gt_d=float((w > 0.05).mean()),
                                          p_gt_swin=float((w > 0.12).mean()))
    return out


if __name__ == "__main__":
    summaries = []
    for f in sorted(R.glob("b*.npz")):
        s = analyze(f.stem)
        summaries.append(s)
        print(json.dumps(s))
        print("---", flush=True)
    (R / "summary.json").write_text(json.dumps(summaries, indent=1))
    print(f"wrote {R/'summary.json'}")
