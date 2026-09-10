#!/usr/bin/env python3
"""E0k: 直空间 kernel 消融阶梯(带宽墙 vs 计算/延迟墙的无 profiler 判定)。

同数据同布局同列表,逐级累加成本成分(每级差值 = 该成分真实成本):
  P1 纯访存:x[j] 载入 + r^2 + mask 内 r2 求和(防 DCE)——无除法/无 exp/无 erfc
  P2 +LJ(1 除 + ~10 flops)
  P3 +静电(intrinsic erfc + exp)
  k2 全量(poly erfc;基准 15.65 ms)
判定:P1 若 ≈ 15 ms => 带宽墙(架构重构对 此布局 判死);P1 若 ≪ 15 => 差值
是计算/发射,重构/ILP 有空间。占用守卫同 E0g 守卫版。
"""
import subprocess, time
import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if util > 20:
    sys.exit = None  # noqa -- 占用守卫在迭代级,入口不硬拒
def _exclusive():
    try:
        mem = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"]).strip())
        return mem < 100   # MiB: 无共租户 => util 采样只会看到我们自己
    except Exception:
        return False


EXCLUSIVE = _exclusive()



import cupy as cp

SRC = open("/root/e0g_kernels.cu", encoding="ascii").read()

SRC += r"""
// P1: loads + r2 only
extern "C" __global__ void p1_load(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    double* __restrict__ F, int N, int R, int maxnb, double rc2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double acc = 0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        acc += (r2 < rc2) ? r2 : 0.0;
    }
    F[a*R + r] = acc;
}

// P2: + LJ (1 div)
extern "C" __global__ void p2_lj(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ sig, const double* __restrict__ eps,
    double* __restrict__ F, int N, int R, int maxnb, double rc2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double si = sig[a], ei = eps[a];
    double fx=0., fy=0., fz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            double invr2 = 1.0/r2;
            double s = 0.5*(si+sig[j]), e = sqrt(ei*eps[j]);
            double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            double f = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
            fx += f*dx; fy += f*dy; fz += f*dz;
        }
    }
    F[(a*R+r)*3] = fx; F[(a*R+r)*3+1] = fy; F[(a*R+r)*3+2] = fz;
}

// P3: + electrostatics (intrinsic erfc + exp)
extern "C" __global__ void p3_full(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    double* __restrict__ F, int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], ei = eps[a];
    double fx=0., fy=0., fz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            double invr2 = 1.0/r2;
            double rr = sqrt(r2);
            double invr = rr*invr2;
            double s = 0.5*(si+sig[j]), e = sqrt(ei*eps[j]);
            double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
            double emx2 = exp(-alpha*alpha*r2);
            double ec = erfc(alpha*rr);
            double fc = qi*q[j]*(ec*invr2 + 2.0*alpha*0.5641895835477563*emx2*invr);
            double f = flj + fc;
            fx += f*dx; fy += f*dy; fz += f*dz;
        }
    }
    F[(a*R+r)*3] = fx; F[(a*R+r)*3+1] = fy; F[(a*R+r)*3+2] = fz;
}

// P3b: P3 without exp (cheap stand-in) - isolates exp cost
extern "C" __global__ void p3b_noexp(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    double* __restrict__ F, int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], ei = eps[a];
    double fx=0., fy=0., fz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            double invr2 = 1.0/r2;
            double rr = sqrt(r2);
            double invr = rr*invr2;
            double s = 0.5*(si+sig[j]), e = sqrt(ei*eps[j]);
            double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
            double emx2 = 1.0 - alpha*alpha*r2*1e-9;   // cheap stand-in (cost probe)
            double ec = erfc(alpha*rr);
            double fc = qi*q[j]*(ec*invr2 + 2.0*alpha*0.5641895835477563*emx2*invr);
            double f = flj + fc;
            fx += f*dx; fy += f*dy; fz += f*dz;
        }
    }
    F[(a*R+r)*3] = fx; F[(a*R+r)*3+1] = fy; F[(a*R+r)*3+2] = fz;
}


// P4: k2 with poly-exp (e^-y = e^-k * poly(frac), y in [0,22]; rel 1.5e-15)
__device__ __forceinline__ double epoly_frac(double t) {
    return ((((((((((((+1.26981333521958983e-09*t-2.26359758176243772e-08)*t+2.70644991265625131e-07)*t-2.74849365234836673e-06)*t+2.47940001576616951e-05)*t-1.98407135174930412e-04)*t+1.38888610078410392e-03)*t-8.33333240829249068e-03)*t+4.16666664735225387e-02)*t-1.66666666643370764e-01)*t+4.99999999998610889e-01)*t-9.99999999999971134e-01)*t+1.00000000000000044e+00);
}
extern "C" __global__ void p4_polyexp(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    const double* __restrict__ se, const double* __restrict__ ek,
    double* __restrict__ F, int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], sei = se[a];
    double lx=0., ly=0., lz=0., cx=0., cy=0., cz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            double invr2 = 1.0/r2;
            double rr = sqrt(r2);
            double invr = rr*invr2;
            double s = 0.5*(si+sig[j]), e = se[j];
            double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
            double y = alpha*alpha*r2;
            int kk = (int)y;
            double emx2 = ek[kk] * epoly_frac(y - kk);
            double ec = erfc_poly(alpha*rr, emx2);
            double fc = q[j]*(ec*invr2 + 2.0*alpha*0.5641895835477563*emx2*invr);
            lx += flj*dx; ly += flj*dy; lz += flj*dz;
            cx += fc*dx; cy += fc*dy; cz += fc*dz;
        }
    }
    F[(a*R+r)*3]   = sei*lx + qi*cx;
    F[(a*R+r)*3+1] = sei*ly + qi*cy;
    F[(a*R+r)*3+2] = sei*lz + qi*cz;
}
"""


mod = cp.RawModule(code=SRC)
K = {n: mod.get_function(n) for n in ("k2_prefold", "p1_load", "p2_lj", "p3_full", "p3b_noexp", "p4_polyexp")}

# ---- 数据(e0g 同法) ----
N, R = 60000, 48
RHO, RC, SKIN, ALPHA = 100.0, 1.0, 0.35, 3.5
LR = RC + SKIN
rng = np.random.default_rng(7)
L = (N / RHO) ** (1 / 3)
m = int(np.ceil(L / 0.215))
pos = np.stack(np.unravel_index(np.arange(N), (m, m, m)), 1).astype(float) * (L / m)
pos += rng.uniform(0, 0.06, pos.shape)
pos %= L
cell = np.floor(pos / LR).astype(int)
ncell = int(np.ceil(L / LR))
cellid = (cell[:, 0] * ncell + cell[:, 1]) * ncell + cell[:, 2]
order = np.argsort(cellid)
sc = cellid[order]
cs = np.searchsorted(sc, np.arange(ncell ** 3))
ce = np.searchsorted(sc, np.arange(ncell ** 3) + 1)
lists = np.zeros((N, 1200), dtype=np.int32)
ncnt = np.zeros(N, dtype=np.int32)
maxnb = 0
for a0 in range(0, N, 2000):
    a1 = min(N, a0 + 2000)
    cand, masks = [], []
    cx, cy, cz = cell[a0:a1, 0], cell[a0:a1, 1], cell[a0:a1, 2]
    for dx_ in (-1, 0, 1):
        for dy_ in (-1, 0, 1):
            for dz_ in (-1, 0, 1):
                cid = (((cx+dx_) % ncell)*ncell + (cy+dy_) % ncell)*ncell + (cz+dz_) % ncell
                s, e = cs[cid], ce[cid]
                cnt = e - s
                mc = int(cnt.max()) if len(cnt) else 0
                if mc == 0:
                    continue
                ar = np.arange(mc)
                offs = s[:, None] + ar[None, :]
                mk = ar[None, :] < cnt[:, None]
                cand.append(order[np.where(mk, offs, 0)])
                masks.append(mk)
    C = np.concatenate(cand, 1)
    M = np.concatenate(masks, 1)
    d = pos[C] - pos[a0:a1, None, :]
    d -= np.round(d / L) * L
    r2 = (d * d).sum(-1)
    keep = M & (r2 <= LR * LR) & (C > a0 + np.arange(a1 - a0)[:, None])
    for a in range(a1 - a0):
        js = C[a][keep[a]]
        lists[a0 + a, :len(js)] = js
        ncnt[a0 + a] = len(js)
        maxnb = max(maxnb, len(js))

q = rng.uniform(-1, 1, N)
sig = rng.uniform(0.3, 0.4, N)
eps = rng.uniform(0.3, 0.7, N)
x0 = np.repeat(pos[:, None, :], R, axis=1) + rng.normal(0, 0.01, (N, R, 3))
x0 %= L
d_x = cp.asarray(x0)
d_list = cp.asarray(lists[:, :maxnb].ravel())
d_ncnt = cp.asarray(ncnt)
d_q, d_s, d_e = cp.asarray(q), cp.asarray(sig), cp.asarray(eps)
d_se = cp.asarray(np.sqrt(eps))
d_ek = cp.asarray(np.exp(-np.arange(23)))   # e^{-k} table for poly-exp
F = cp.zeros(N * R * 3)


def gpu_util():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"]).strip())
    except Exception:
        return -1


def bench(kname, iters=30, block=128):
    grid = ((N * R + block - 1) // block,)
    if kname == "p4_polyexp":
        args = (d_x, d_list, d_ncnt, d_q, d_s, d_e, d_se, d_ek, F,
                N, R, maxnb, RC * RC, ALPHA)
    elif kname == "k2_prefold":
        args = (d_x, d_list, d_ncnt, d_q, d_s, d_e, cp.asarray(np.sqrt(eps)), F,
                N, R, maxnb, RC * RC, ALPHA)
    elif kname == "p1_load":
        args = (d_x, d_list, d_ncnt, F, N, R, maxnb, RC * RC)
    elif kname == "p2_lj":
        args = (d_x, d_list, d_ncnt, d_s, d_e, F, N, R, maxnb, RC * RC)
    else:
        args = (d_x, d_list, d_ncnt, d_q, d_s, d_e, F, N, R, maxnb, RC * RC, ALPHA)
    K[kname](grid, (block,), args)
    cp.cuda.Stream.null.synchronize()
    clean = []
    for _ in range(iters):
        u0 = gpu_util()
        t0 = time.perf_counter()
        K[kname](grid, (block,), args)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        if EXCLUSIVE:
            if dt < 0.2:
                clean.append(dt)
        elif u0 <= 20 and gpu_util() <= 20 and dt < 0.2:
            clean.append(dt)
    return min(clean) if clean else float("nan"), len(clean)


print(f"N={N} maxnb={maxnb} avg={ncnt.mean():.0f} (exclusive={EXCLUSIVE}: 守卫自动切换)")
import time as _time
res, attempt = {}, 0
ORDER = ("p4_polyexp", "k2_prefold", "p3b_noexp", "p3_full", "p2_lj", "p1_load")
while attempt < 10 and any(k not in res for k in ORDER):
    attempt += 1
    for kn in ORDER:
        if kn in res:
            continue
        t, n = bench(kn, iters=8)
        if n >= 3:
            res[kn] = t
            print(f"[try{attempt}] {kn:>10}: {t*1e3:8.3f} ms (n_clean={n})", flush=True)
        else:
            print(f"[try{attempt}] {kn:>10}: retry (n_clean={n})", flush=True)
    if any(k not in res for k in ORDER):
        _time.sleep(120)
if len(res) < len(ORDER):
    print("INCOMPLETE:", {k: f"{res[k]*1e3:.2f}" for k in res})

t1, t2, t3, tk = res["p1_load"], res["p2_lj"], res["p3_full"], res["k2_prefold"]
if "p4_polyexp" in res and "k2_prefold" in res:
    print(f"p4(poly-exp) vs k2(intrinsic-exp): {res['p4_polyexp']*1e3:.2f} vs "
          f"{res['k2_prefold']*1e3:.2f} ms")
print(f"\n消融: 纯访存 {t1*1e3:.2f} | +LJ(1除) {t2*1e3:.2f} (Δ{1e3*(t2-t1):.2f})"
      f" | +静电(erfc+exp) {t3*1e3:.2f} (Δ{1e3*(t3-t2):.2f})"
      f" | k2(poly) {tk*1e3:.2f} (Δ{1e3*(tk-t3):.2f})")
frac = t1 / tk
print(f"纯访存占比 = {frac*100:.0f}%  => "
      + ("带宽墙主导" if frac > 0.75 else "计算/延迟占比 %.0f%% —— 重构有空间" % (100*(1-frac))))
