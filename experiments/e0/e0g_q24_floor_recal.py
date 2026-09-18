#!/usr/bin/env python3
"""E0g-Q24.40 变体: Q24.40 形态直空间地板行重定标 (登记待办③, review E0p 复跑段).

问题: E0p 的 fp64/Q24.40 同列表比 (1.55-1.71x) 用的是随机全列表 + e0p 几何,
与地板行 16.0 ms 的 E0g 口径 (抖动格子 + 空间一致半列表 + 真实掩码分数)
不可直接比 -- Q24.40 的地板行至今没有合规测量.

台本: E0g 列表/密度口径复刻 (seed 7 抖动格子 N=60k/RHO=100/LR=1.35 cell
半列表 j>i, R=48), 同一列表跑三行:
  anchor  e0g k2_prefold (default opts, block=128) -- 系到 16.0 ms 地板行,
          兼作窗口质量锚 (anchor 明显偏离 16.0 => 本窗口作废)
  fp64    gpu/direct.py (生产 k2 形态, A1 对齐过) -- 生产 fp64 行
  Q24.40  gpu/force_q24.py (柱 1 生产形态) -- 重定标对象
推断: 重定标地板 = Q24.40_row x (16.0 / anchor_row) (窗口归一, 消窗间漂移).

anchor 语义勘误 (2026-09-18 单对隔离实测): e0g_kernels.cu 全家族无 MIC
(dx 直接相减, 无 round(dx/L) 回卷) —— 非周期时序内核, 跨边界对被 rc 掩码
丢弃 (这是 16.0 ms 血统口径的一部分, e0g 自身如此)。故:
  - anchor 行仅作窗口校准 (同内核同口径系到 16.0), 不与生产行比力值;
  - 生产行 (direct.py/force_q24 均有 in-kernel MIC) 互相同构, 行间比值
    = 纯算术形态比 (本台本交付物);
  - sanity = fp64 手搓 launch vs direct_forces wrapper (同内核同参数,
    必须机器精度一致 —— 转录检查), 非 fp64-vs-anchor。

包络偏离声明 (2026-09-18 冒烟实测, 非任意选择): e0g 原参数 (均匀随机
q/sig/eps + 抖动 0.06) 不可用于 Q24.40 形态 -- 非弛豫晶格把最小对压到
r~0.12-0.17, O-O LJ 排斥墙 sr12 项实测 |F|~1e8-1e9, 超 Q24.40 包络
(2^23~8.4e6) 两三个量级 => 饱和 + n_overflow~6e3, 固定点测不了。故:
参数改生产 TIP3P-FB 平铺 (电荷 ±0.734/0.367, sig 0.3186/0.0087,
eps 0.6502/0), 抖动 0.06 -> 0.01 (min r~0.186, LJ 墙 ~5.7e5, 14x 裕量;
累积器和 ~2.7e6 上界, 3x 裕量)。时序对参数值/抖动不敏感 (分支仅几何:
in-rc 掩码 + 列表统计不变), anchor 行仍系 16.0。

计时纪律: 独占窗口 (util > 5% 拒跑) + 逐迭代 util 采样 (>30% 丢弃该迭代);
行间 A/B/A/B 交替对消慢漂移; median 与 min 并报 (e0g 口径用 min).
 Sanity (非门, 仅打印): fp64 手搓 launch vs wrapper 逐位一致性 (转录检查);
 Q24.40 查溢出 (走官方 wrapper 无转录面)。fp64-vs-anchor 力对拍无意义:
 anchor 无 MIC (勘误见头部), 与生产行工作量不同。

RECAL_N 环境变量可缩小 N 做冒烟 (默认 60000 = 生产口径).
"""
import os
import subprocess
import sys
import time

import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu",
     "--format=csv,noheader,nounits"]).strip())
if util > 5:
    sys.exit(f"GPU busy ({util}%) -- exclusive window required for timing.")
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

N = int(os.environ.get("RECAL_N", "60000"))
assert N % 3 == 0, "water tiling requires N divisible by 3"
RHO = 100.0          # atoms/nm^3 (E0g 口径)
RC, SKIN, ALPHA = 1.0, 0.35, 3.5
LR = RC + SKIN
R = 48
FLOOR_K2_MS = 16.0   # 09-08 E0g 实测地板行 (k2 合规形态 @R48)
ITERS = 10
ROUNDS = 2

# ---------------- E0g 数据口径 (builder 逐字取自 e0g_ilp_bench.py) ---------
rng = np.random.default_rng(7)
L = (N / RHO) ** (1 / 3)
m = int(np.ceil(L / 0.215))
assert m ** 3 >= N
pos = np.stack(np.unravel_index(np.arange(N), (m, m, m)), 1).astype(float) * (L / m)
pos += rng.uniform(0, 0.01, pos.shape)      # 抖动 0.01 (包络安全: 0.06 会把
                                            # min r 压进 LJ 墙, 见头部声明)
pos %= L

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
                offs = s[:, None] + ar[None, :]
                mk = ar[None, :] < cnt[:, None]
                offs = np.where(mk, offs, 0)
                cand.append(idx_sorted[offs])
                masks.append(mk)
    C = np.concatenate(cand, 1)
    M = np.concatenate(masks, 1)
    d = pos[C] - pos[a0:a1, None, :]
    d -= np.round(d / L) * L
    r2 = (d * d).sum(-1)
    keep = M & (r2 <= LR * LR) & (C > a0 + np.arange(a1 - a0)[:, None])
    for a in range(a1 - a0):
        js = C[a][keep[a]]
        n = len(js)
        lists[a0 + a, :n] = js
        ncnt[a0 + a] = n
        maxnb = max(maxnb, n)
        ninrc += int((M[a] & (r2[a] <= RC * RC)
                      & (C[a] > a0 + a)).sum())
maxnb = int(maxnb)
print(f"N={N} box={L:.4f}  maxnb={maxnb}  avg nb(skin)={ncnt.mean():.0f}  "
      f"avg in-rc={ninrc / N:.0f}  mask fraction={ninrc / ncnt.sum():.3f}",
      flush=True)
assert maxnb <= 1200

q = np.tile(np.array([-0.734, 0.367, 0.367]), N // 3)      # 生产 TIP3P-FB
sig = np.tile(np.array([0.3186, 0.0087, 0.0087]), N // 3)
eps = np.tile(np.array([0.6502, 0.0, 0.0]), N // 3)
sqrteps = np.sqrt(eps)
x0 = np.repeat(pos[:, None, :], R, axis=1) + rng.normal(0, 0.01, (N, R, 3))

# ---------------- 三行内核装配 ----------------
import cupy as cp
from gpu import direct, force_q24

import pathlib
SRC = open(pathlib.Path(__file__).with_name("e0g_kernels.cu"),
           encoding="ascii").read()
mod = cp.RawModule(code=SRC, options=("--std=c++17",))
k2 = mod.get_function("k2_prefold")

d_x = cp.asarray(np.ascontiguousarray(x0))
d_list = cp.asarray(lists[:, :maxnb].ravel())      # 平铺: k2 / fp64 手发射
d_list2d = cp.asarray(np.ascontiguousarray(lists[:, :maxnb]))  # (N,maxnb): force_q24 wrapper 要二维
d_ncnt = cp.asarray(ncnt)
d_q, d_sig, d_eps = cp.asarray(q), cp.asarray(sig), cp.asarray(eps)
d_se = cp.asarray(sqrteps)
F2 = cp.zeros(N * R * 3)
grid = ((N * R + 127) // 128,)
BOX = np.diag([L, L, L])


def row_anchor():
    k2(grid, (128,), (d_x, d_list, d_ncnt, d_q, d_sig, d_eps, d_se, F2,
                      N, R, maxnb, RC * RC, ALPHA))
    return F2


# ---------------- fp64 行装配 (设备驻留, prime 一次) ----------------
# wrapper 每次调用从 host 上传 x/nlist (~270 MB PCIe), 会把行时序虚高
# 几十 ms (e0p 首轮形态比被这个不对称污染过: fp64 行含上传, Q24.40 行
# 不含)。prime 后取编译好的内核, 发射元组逐字镜像 gpu/direct.py::
# direct_forces; 转录风险由尾部 sanity (手搓 vs wrapper ~1e-15) 兜底。
direct.direct_forces(x0, q, sig, eps, lists[:, :maxnb], ncnt,
                     ALPHA, RC, box_diag=[L, L, L], block=128)  # prime
k_fp = direct._SRC_CACHE["k"]
F_fp_dev = cp.zeros(N * R * 3)


def row_fp64():
    k_fp(grid, (128,), (d_x, d_list, d_ncnt, d_q, d_sig, d_eps, d_se,
                        F_fp_dev, N, R, maxnb, RC * RC, ALPHA,
                        float(L), float(L), float(L)))


def row_q2440():
    return force_q24.direct_forces_q(d_x, d_q, d_sig, d_eps, d_list2d, d_ncnt,
                                     ALPHA, RC, box=BOX)


ROWS = [("anchor k2 (e0g)", row_anchor),
        ("fp64 gpu/direct", row_fp64),
        ("Q24.40 gpu/force_q24", row_q2440)]


def gpu_util():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu",
         "--format=csv,noheader,nounits"]).strip())


samples = {name: [] for name, _ in ROWS}
dropped = 0
for rnd in range(ROUNDS):
    for name, fn in ROWS:
        for _ in range(ITERS):
            u0 = gpu_util()
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            fn()
            cp.cuda.Stream.null.synchronize()
            dt = time.perf_counter() - t0
            u1 = gpu_util()
            if max(u0, u1) > 30:
                dropped += 1
                continue
            samples[name].append(dt)
            if dt > 0.2:
                dropped += 1

print(f"[util-guard: {dropped} polluted iters dropped]  "
      f"rounds={ROUNDS} iters={ITERS}", flush=True)
med, mn = {}, {}
for name, _ in ROWS:
    ts = np.array(samples[name]) * 1e3
    med[name] = float(np.median(ts))
    mn[name] = float(ts.min())
    npair_G = float(ncnt.sum()) * R / (ts.min() / 1e3) / 1e9
    print(f"{name:>22}: median {med[name]:8.2f} ms   min {mn[name]:8.2f} ms   "
          f"[{npair_G:.1f} G list-pairs/s @min]", flush=True)

r_q_fp = med["Q24.40 gpu/force_q24"] / med["fp64 gpu/direct"]
anchor_drift = med["anchor k2 (e0g)"] / FLOOR_K2_MS
recal_norm = med["Q24.40 gpu/force_q24"] * (FLOOR_K2_MS
                                            / med["anchor k2 (e0g)"])
print(f"\nQ24.40 / fp64 (同列表纯算术形态比, E0g 口径): {r_q_fp:.2f}x")
print(f"anchor vs 地板行 {FLOOR_K2_MS} ms: {anchor_drift:.3f}x  "
      f"({'窗口可信' if 0.9 <= anchor_drift <= 1.15 else '窗口异常 -- 本轮测量不可作门'})")
print(f"Q24.40 地板行重定标(窗口归一): {recal_norm:.2f} ms  "
      f"(raw {med['Q24.40 gpu/force_q24']:.2f} ms)", flush=True)

# ---------------- sanity (非门, 转录检查) ----------------
# fp64 手搓 launch vs wrapper: 同内核同参数逐位可重现 (rel ~1e-15), 抓的
# 是本台本发射元组的转录错。anchor 与生产行工作量不同 (无 MIC/边缘对),
# 力值不可比 (勘误见头部)。Q24.40 行走官方 wrapper 无转录面, 查溢出即可。
row_fp64()
F_wr = cp.asnumpy(direct.direct_forces(
    x0, q, sig, eps, lists[:, :maxnb], ncnt, ALPHA, RC,
    box_diag=[L, L, L], block=128)).reshape(N, R, 3)
F_fp = F_fp_dev.reshape(N, R, 3).get()
rel_launch = np.abs(F_fp - F_wr).max() / max(np.abs(F_wr).max(), 1e-30)
Fq, novf = row_q2440()
F_q = Fq.get() * (2.0 ** -40)
print(f"sanity: fp64 手搓launch vs wrapper max rel {rel_launch:.2e} "
      f"(同内核同参数, 期望 ~1e-15)   |F_fp|max={np.abs(F_fp).max():.3e}   "
      f"|F_q24|max={np.abs(F_q).max():.3e} (内折 KE)   "
      f"n_overflow={novf} (期望 0)", flush=True)
