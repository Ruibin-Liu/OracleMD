#!/usr/bin/env python3
"""E0p: M2 component timing at production scale (N=60k / R=48 / NG=128 /
dt=4 fs) -- the hybrid-gate floor measurements for the COLLECTED kernels.

Timing discipline: exclusive window (util guard below); median of runs
after warmup; same synthetic lists for the fp64-k2-shaped kernel
(gpu/direct.py, the E0g floor baseline form) and the Q24.40 kernel
(gpu/force_q24.py) so the ratio is a pure arithmetic-form comparison.

Rows (floor table, v1.1.7): direct 16.0 / FFT 13.3 / spread 4.7 /
interp 0.9 / constrain+integrate 1.5 ms/step, kappa = 1.5.
NOTE the Q24.40 direct form and the REAL RNG integrate are new arithmetic
semantics -- these timings are the re-baselining data per the
baseline-source discipline (spec 10.3).
"""
import sys
import time
import subprocess

import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu",
     "--format=csv,noheader,nounits"]).strip())
if util > 5:
    sys.exit(f"GPU busy ({util}%) -- exclusive window required for timing.")
sys.path.insert(0, "/root")

N, R, NG = 60000, 48, 128
L = 12.15
NCNT = 474        # per-atom list entries (E0g k2 convention)
MAXNB = 512
DT = 4.0e-3


def bench(fn, iters=10, warmup=3):
    import cupy as cp
    import subprocess as _sp

    def _util():
        return int(_sp.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"]).strip())
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    ts = []
    for _ in range(iters):
        cp.cuda.Stream.null.synchronize()
        u0 = _util()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        ts.append(time.perf_counter() - t0)
        u1 = _util()
        if max(u0, u1) > 30:
            print(f"  [WARN] util {max(u0, u1)}% during bench "
                  f"(co-tenant) -- USABLE 级存疑", flush=True)
    return float(np.median(ts))


def main():
    import cupy as cp
    from gpu import direct, force_q24, integrate, pme, step as step_mod
    rng = np.random.default_rng(0)

    # synthetic production-scale system
    L = 12.15
    x = rng.uniform(0, L, (N, R, 3))
    nw = N // 3
    q = np.tile(np.array([-0.734, 0.367, 0.367]), nw)
    sig = np.tile(np.array([0.3186, 0.0087, 0.0087]), nw)
    eps = np.tile(np.array([0.6502, 0.0, 0.0]), nw)
    m = np.tile(np.array([15.999, 1.008, 1.008]), nw)
    box = np.diag([L, L, L])
    # synthetic full lists (timing-only; same arrays for both kernels)
    nlist = np.zeros((N, MAXNB), dtype=np.int32)
    ncnt = np.full(N, NCNT, dtype=np.int32)
    for a in range(N):
        nlist[a] = rng.integers(0, N, size=MAXNB)
    print(f"system: N={N} R={R} NG={NG} list pairs/step "
          f"~{N * NCNT * R / 2 / 1e9:.2f} G", flush=True)

    x_dev = cp.asarray(x)
    d_q = cp.asarray(q)
    d_sig = cp.asarray(sig)
    d_eps = cp.asarray(eps)
    d_nlist = cp.asarray(nlist)
    d_ncnt = cp.asarray(ncnt)

    # 1) direct fp64 (E0g k2 baseline form)
    t_k2 = bench(lambda: direct.direct_forces(
        x, q, sig, eps, nlist, ncnt, 3.5, 0.9,
        box_diag=[L, L, L]))
    print(f"direct fp64 (k2 form)      : {t_k2 * 1e3:8.2f} ms   "
          f"[floor row 16.0]", flush=True)

    # 2) direct Q24.40 (pillar-1 production form)
    t_q = bench(lambda: force_q24.direct_forces_q(
        x_dev, d_q, d_sig, d_eps, d_nlist, d_ncnt, 3.5, 0.9,
        box=np.diag([L, L, L])))
    print(f"direct Q24.40 (pillar 1)   : {t_q * 1e3:8.2f} ms   "
          f"ratio vs k2 {t_q / t_k2:.2f}x", flush=True)

    # 3) integrate + shake (REAL RNG) at production scale
    v_dev = cp.asarray(rng.normal(0, 0.15, (N * R * 3)))
    f_dev = cp.asarray(rng.normal(0, 1e3, (N * R * 3)))
    mass = cp.asarray(m)
    invm = cp.asarray(1.0 / m)

    def int_step():
        integrate.baoab_step(x_dev, v_dev, f_dev, mass, invm, N, R,
                             dt=DT, gamma=1.0,
                             kT=0.00831446261815324 * 300.0,
                             step=1, seed=77, first=False,
                             r_oh=0.09572, r_hh=0.15139)
    t_int = bench(int_step)
    print(f"integrate+shake+RNG (real) : {t_int * 1e3:8.2f} ms   "
          f"[floor row 1.5 (placeholder RNG)]", flush=True)

    # 3) spread tile: KERNEL-ONLY (floor convention); host cell_sort at
    # 60k is a python loop -- measured once separately, registered as a
    # production work item (vectorize).
    # e0h2 bench convention: positions off the seam -> zero shifts;
    # vectorized cell build (the production cell_sort python loop is the
    # registered vectorize work item)
    x_sp_base = rng.uniform(0.2, L - 0.2, (N, 1, 3))
    x_sp = (x_sp_base + rng.normal(0, 0.01, (N, R, 3))) % L
    t0 = time.perf_counter()
    inv_h = NG / L
    gx = x_sp * inv_h
    TC = 8
    C = -(-NG // TC)
    kmin = np.floor(gx.min(axis=1)).astype(np.int64) - 1
    kmax = np.floor(gx.max(axis=1)).astype(np.int64) + 2
    entries = []
    from itertools import product as _prod
    for a in range(N):
        sets = [sorted(set(range(int(kmin[a, d] // TC),
                                 int(kmax[a, d] // TC) + 1)))
                for d in range(3)]
        for bx, by, bz in _prod(*sets):
            entries.append((((bx % C) * C + (by % C)) * C + (bz % C),
                            bx * TC, by * TC, bz * TC, a))
    ent = np.array(entries, dtype=np.int64)
    ent = ent[np.argsort(ent[:, 0], kind="stable")]
    cid_e = ent[:, 0]
    used_ids = np.unique(cid_e)
    cs = np.searchsorted(cid_e, used_ids, "left").astype(np.int32)
    ce = np.searchsorted(cid_e, used_ids, "right").astype(np.int32)
    org = ent[cs, 1:4].astype(np.int32)
    atom_e = ent[:, 4].astype(np.int64)
    ncell = int(len(used_ids))
    shift = np.zeros((len(atom_e), R, 3), dtype=np.int8)
    t_sort = time.perf_counter() - t0
    xs_s = np.ascontiguousarray(x.reshape(N, R, 3)[atom_e])
    qs_s = np.ascontiguousarray(q[atom_e])
    mod, k_tile = pme._module(NG, 8)[0], pme._module(NG, 8)[2]
    grid = cp.zeros(R * NG ** 3, dtype=cp.int64)
    inv = np.linalg.inv(np.diag([L, L, L]))
    args = (cp.asarray(xs_s), cp.asarray(qs_s), cp.asarray(shift),
            cp.asarray(cs), cp.asarray(ce), cp.asarray(org.reshape(-1)),
            grid, np.int32(ncell), np.int32(R),
            float(inv[0, 0]), float(inv[1, 1]), float(inv[2, 2]),
            float(pme.SCALE))

    def spread_kernel():
        grid.fill(0)
        k_tile((max(ncell, 1) * R,), (128,), args)
    t_sp = bench(spread_kernel)
    print(f"spread tile (kernel only)  : {t_sp * 1e3:8.2f} ms   "
          f"[floor row 4.7]", flush=True)
    print(f"  host cell_sort (once/    : {t_sort:8.2f} s    "
          f"[vectorize = production work item]", flush=True)
    print("  window, amortized /48)  :", flush=True)

    # 4) FFT chain: rfftn + c(m) + irfftn (48x128^3)
    grid = cp.asarray(np.zeros((R, NG, NG, NG), dtype=np.int64))
    ctx_x = x_dev.reshape(N, R, 3)
    from gpu import rand as rand_mod
    gf = grid.astype(np.float64) * (2.0 ** -48)
    ch = np.zeros((NG, NG, NG // 2 + 1), dtype=np.complex128)
    ch_dev = cp.asarray(ch)

    SUB = 1  # E0e: per-transform time batch-independent (48 x 0.28 ms); batch=1 minimizes cuFFT plan work areas under a co-tenant
    def fft_chain():
        # keeps peak memory low under a co-tenant; the total over all
        # sub-batches remains floor-comparable (E0e batch-invariance)
        pot = cp.empty((R, NG, NG, NG), dtype=np.float64)
        for b0 in range(0, R, SUB):
            gf2 = grid[b0:b0 + SUB].astype(np.float64) * (2.0 ** -48)
            G = cp.fft.rfftn(gf2, axes=(1, 2, 3))
            G *= ch_dev
            pot[b0:b0 + SUB] = cp.fft.irfftn(
                G, s=(NG, NG, NG), axes=(1, 2, 3)).real
            del G, gf2
            cp.get_default_memory_pool().free_all_blocks()
        return pot
    t_fft = bench(fft_chain)
    cp.get_default_memory_pool().free_all_blocks()
    print(f"FFT chain r2c+c(m)+c2r     : {t_fft * 1e3:8.2f} ms   "
          f"[floor row 13.3 (c2c)]", flush=True)

    # 5) interp
    pot = fft_chain()
    t_in = bench(lambda: pme.interp_forces(x_dev, d_q,
                                           pot.reshape(R, -1), box, ng=NG))
    print(f"interp (adjoint gather)    : {t_in * 1e3:8.2f} ms   "
          f"[floor row 0.9]", flush=True)

    tot = t_sp + t_fft + t_in + t_q + t_int
    print(f"\ncomponent total (Q24.40 form): {tot * 1e3:.2f} ms/step"
          f"  -> {86400 * 0.004 / tot:.0f} ns/day @48 x 4 fs", flush=True)


if __name__ == "__main__":
    main()
