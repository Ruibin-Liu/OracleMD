#!/usr/bin/env python3
"""E0g: 直空间内核 ILP/算术优化阶梯(M2 tile 前置;Q-002 从 21% peak 上探)。

变体(全部 f64 + 3 段 poly-erfc,同 e0f):
  K0 = e0f poly 基线(逐字)
  K1 = 算术重构:r = sqrt(r2); invr = r*invr2; x = alpha*r(每对省 1 除)
  K2 = K1 + sqrt(eps) 预计算(省 1 sqrt/对)+ qi*KE 端点折叠(省 2 乘/对)
  K3 = K2 + k 展开 x4,四组独立 (fx,fy,fz) 累加器(断依赖链,主 ILP)
数据:物理几何(抖动格子水密度,真实 1.35 nm 邻居表,α=3.5,rc=1.0 mask)
⇒ 分支混合与生产一致(W0/W1/W2 按体积加权)。
交叉验证:各变体 vs K0,max_rel 预期 ~1e-14(重结合轮次差)。
GPU util > 20% 拒跑。
"""
import sys, time, subprocess
import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if util > 20:
    sys.exit(f"GPU busy ({util}%), timing invalid.")

import cupy as cp

SRC = open(__import__("pathlib").Path(__file__).with_name(
    "e0g_kernels.cu"), encoding="ascii").read()

import os
FAST = os.environ.get("FAST_MATH", "0") == "1"
OPTS = ("-prec-div=false,-prec-sqrt=true",) if False else (
    ("--std=c++17", "-prec-div=false", "-prec-sqrt=false") if FAST else ("--std=c++17",))
mod = cp.RawModule(code=SRC, options=OPTS)
print("NVRTC options:", OPTS)
KER = {n: mod.get_function(n) for n in
       ("k0_base", "k1_arith", "k2_prefold", "k3_unroll", "k4_compact", "k5_compact_masked")}

# ---------------- 物理数据:抖动格子 + 真实邻居表 ----------------
N = 60000
RHO = 100.0          # atoms/nm^3
RC, SKIN, ALPHA = 1.0, 0.35, 3.5
LR = RC + SKIN

rng = np.random.default_rng(7)
L = (N / RHO) ** (1 / 3)
m = int(np.ceil(L / 0.215))
assert m ** 3 >= N
pos = np.stack(np.unravel_index(np.arange(N), (m, m, m)), 1).astype(float) * (L / m)
pos += rng.uniform(0, 0.06, pos.shape)      # 抖动
pos %= L

# cell-list 邻居(半列表, j > i, dist <= LR)
cell = np.floor(pos / LR).astype(int)
ncell = int(np.ceil(L / LR))
cellid = (cell[:, 0] * ncell + cell[:, 1]) * ncell + cell[:, 2]
order = np.argsort(cellid)
sorted_cid = cellid[order]
cell_start = np.searchsorted(sorted_cid, np.arange(ncell**3))
cell_end = np.searchsorted(sorted_cid, np.arange(ncell**3) + 1)
idx_sorted = order

maxnb = 0
lists = np.zeros((N, 1200), dtype=np.int32)
ncnt = np.zeros(N, dtype=np.int32)
clists = np.zeros((N, 700), dtype=np.int32)   # compact in-rc list
cncnt = np.zeros(N, dtype=np.int32)
ninrc = 0
CH = 2000
for a0 in range(0, N, CH):
    a1 = min(N, a0 + CH)
    cand, masks = [], []
    cx, cy, cz = cell[a0:a1, 0], cell[a0:a1, 1], cell[a0:a1, 2]
    for dx_ in (-1, 0, 1):
        for dy_ in (-1, 0, 1):
            for dz_ in (-1, 0, 1):
                gx = (cx + dx_) % ncell
                gy = (cy + dy_) % ncell
                gz = (cz + dz_) % ncell
                cid = (gx * ncell + gy) * ncell + gz
                s, e = cell_start[cid], cell_end[cid]
                cnt = e - s
                mc = int(cnt.max()) if len(cnt) else 0
                if mc == 0:
                    continue
                ar = np.arange(mc)
                offs = s[:, None] + ar[None, :]          # (chunk, mc)
                mk = ar[None, :] < cnt[:, None]
                offs = np.where(mk, offs, 0)             # pad with 0, masked later
                cand.append(idx_sorted[offs])
                masks.append(mk)
    C = np.concatenate(cand, 1)                       # (chunk, ncand)
    M = np.concatenate(masks, 1)
    d = pos[C] - pos[a0:a1, None, :]
    d -= np.round(d / L) * L
    r2 = (d * d).sum(-1)
    keep = M & (r2 <= LR * LR) & (C > a0 + np.arange(a1 - a0)[:, None])
    keepc = M & (r2 <= RC * RC) & (C > a0 + np.arange(a1 - a0)[:, None])
    for a in range(a1 - a0):
        js = C[a][keep[a]]
        n = len(js)
        lists[a0 + a, :n] = js
        ncnt[a0 + a] = n
        maxnb = max(maxnb, n)
        jc = C[a][keepc[a]]
        cncnt[a0 + a] = len(jc)
        clists[a0 + a, :len(jc)] = jc
        ninrc += len(jc)
maxnb = int(maxnb)
print(f"N={N} box={L:.2f} nm  maxnb={maxnb}  avg nb={ncnt.mean():.0f}")
assert maxnb <= 1200
cmaxnb = int(cncnt.max())
print(f"in-rc pairs/atom avg = {cncnt.mean():.0f} (mask fraction {ninrc/ncnt.sum():.3f})")

q = rng.uniform(-1, 1, N)
sig = rng.uniform(0.3, 0.4, N)
eps = rng.uniform(0.3, 0.7, N)
sqrteps = np.sqrt(eps)

R = 48
x0 = np.repeat(pos[:, None, :], R, axis=1) + rng.normal(0, 0.01, (N, R, 3))
d_list = cp.asarray(lists[:, :maxnb].ravel())
d_ncnt = cp.asarray(ncnt)
d_clist = cp.asarray(clists[:, :cmaxnb].ravel())
d_cncnt = cp.asarray(cncnt)

KE = 138.935456  # kJ nm / mol e^2


def bench(kname, iters=10, block=256):
    x = cp.asarray(x0)
    F = cp.zeros(N * R * 3)
    dq, ds, de = cp.asarray(q), cp.asarray(sig), cp.asarray(eps)
    dse = cp.asarray(sqrteps)
    grid = ((N * R + block - 1) // block,)
    lst, cnt, mnb = (d_clist, d_cncnt, cmaxnb) if kname in ("k4_compact", "k5_compact_masked") else (d_list, d_ncnt, maxnb)
    args = (x, lst, cnt, dq, ds, de, dse, F, N, R, mnb,
            RC * RC, ALPHA)
    k = KER[kname]
    k(grid, (block,), args)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        k(grid, (block,), args)
    cp.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t0) / iters
    npairs = float(ninrc if kname in ("k4_compact", "k5_compact_masked") else ncnt.sum()) * R
    return dt, npairs / dt, F, (x, dq, ds, de, dse)


# ---- 交替 A/B: default vs -prec-div=false,-prec-sqrt=false ----
mod_fast = cp.RawModule(code=SRC, options=("--std=c++17", "-prec-div=false",
                                           "-prec-sqrt=false"))
KER_F = {n: mod_fast.get_function(n) for n in
         ("k2_prefold", "k4_compact", "k5_compact_masked")}


def bench2(kname, kers, iters=10, block=128):
    x = cp.asarray(x0)
    F = cp.zeros(N * R * 3)
    dq, ds_, de = cp.asarray(q), cp.asarray(sig), cp.asarray(eps)
    dse = cp.asarray(sqrteps)
    grid = ((N * R + block - 1) // block,)
    lst, cnt, mnb = (d_clist, d_cncnt, cmaxnb) \
        if kname in ("k4_compact", "k5_compact_masked") else (d_list, d_ncnt, maxnb)
    args = (x, lst, cnt, dq, ds_, de, dse, F, N, R, mnb, RC * RC, ALPHA)
    kers[kname](grid, (block,), args)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        kers[kname](grid, (block,), args)
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / iters, F


print("variant / block / default_ms / fast_ms / speedup")
from collections import defaultdict
ab = defaultdict(lambda: ([], []))
for rep in range(5):
    for kn in ("k2_prefold", "k4_compact", "k5_compact_masked"):
        d1, _ = bench2(kn, KER)
        d2, _ = bench2(kn, KER_F)
        ab[kn][0].append(d1)
        ab[kn][1].append(d2)
for kn, (ds_, fs_) in ab.items():
    d_med, f_med = np.median(ds_), np.median(fs_)
    print(f"{kn:>22} {128:>5} {d_med*1e3:>11.3f} {f_med*1e3:>9.3f} {d_med/f_med:>7.2f}x",
          flush=True)
_, Fd = bench2("k2_prefold", KER, iters=1)
_, Ff = bench2("k2_prefold", KER_F, iters=1)
rel = np.abs(Ff.get() - Fd.get()).max() / np.abs(Fd.get()).max()
print(f"k2 default-vs-fast force rel diff: {rel:.1e}  (A1 tol 1e-10)")
ninrc_pairs = float(ninrc) * R
d_med = np.median(ab["k2_prefold"][0])
f_med = np.median(ab["k2_prefold"][1])
print(f"\nQ-002 direct(k2, R=48): default {d_med*1e3:.2f} ms -> fast {f_med*1e3:.2f} ms"
      f"  (computed-pair {ninrc_pairs/d_med/1e9:.1f} -> {ninrc_pairs/f_med/1e9:.1f} Gpair/s)")
