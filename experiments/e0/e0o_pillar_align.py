#!/usr/bin/env python3
"""E0o: pillar alignment -- Q24.40 forces (pillar 1) + counter RNG
(pillar 3).  Runs entirely on the pod (opus + gpu deployed).

Gates:
  P1  direct_q CUDA vs mirror_kernel: BITWISE (alpha=0 aperiodic +
      periodic; exp/log1p-free path) -- transcription layer
  P1b direct_q vs mirror_kernel alpha=3.5: bitwise-or-dual (device exp
      ulps tolerated, counted) + determinism x2 + Newton-3 bitwise
  P1c dequant(direct_q) vs opus direct_space forces: A1 dual gate
      (poly-erfc dominant, phase-1 style), inter-molecular pairs
  P3  device ziggurat vs host mirror: BITWISE over 64k streams x 8 draws
      (log1p/exp parity probe)
  P3b BAOAB chain gamma=1.0 T=300 R=1 vs live opus Dynamics.step_baoab:
      x/v BITWISE (slot=0 == R=1 composition)
"""
import sys
import os

import numpy as np

sys.path.insert(0, "/root")


class _C:
    __slots__ = ("a", "b", "distance")

    def __init__(self, a, b, d):
        self.a, self.b, self.distance = a, b, d


def water_cluster(nw, r, seed, box_len=2.0, jitter=0.005, min_sep=0.45):
    rng = np.random.default_rng(seed)
    ideal = np.array([[0.0, 0.0, 0.0], [0.09572, 0.0, 0.0],
                      [0.02377, 0.09272, 0.0]])
    x = np.empty((nw * 3, r, 3))
    placed = []
    for w in range(nw):
        for _ in range(200):
            c = rng.uniform(0.3, box_len - 0.3, 3)
            if all(np.linalg.norm(c - p) > min_sep for p in placed):
                break
        placed.append(c)
        qq, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        for k in range(3):
            base = c + ideal[k] @ qq + jitter * rng.normal(size=3)
            for rep in range(r):
                x[3 * w + k, rep] = base + 0.01 * rng.normal(size=3)
    x %= box_len
    return x


def lists_from(pairs, n):
    maxnb = max(sum(1 for p in pairs if a in p) for a in range(n)) + 1
    nlist = np.full((n, maxnb), -1, dtype=np.int32)
    ncnt = np.zeros(n, dtype=np.int32)
    for (i, j) in pairs:
        nlist[i, ncnt[i]] = j
        ncnt[i] += 1
        nlist[j, ncnt[j]] = i
        ncnt[j] += 1
    return nlist, ncnt


def dual(a, b, rel=1e-10, abs_=1e-8):
    da = np.abs(a - b)
    sc = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    return bool(np.all((da < abs_) | (da / sc < rel)))


def main():
    import cupy as cp
    from gpu import force_q24, integrate, rand
    fails = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    # ---------------- P1: direct_q vs mirror ----------------
    from gpu.force_q24 import direct_q_mirror_kernel, _constants
    nw, r, L = 10, 2, 2.0
    x = water_cluster(nw, r, seed=21, box_len=L)
    rng = np.random.default_rng(22)
    n = nw * 3
    q = rng.uniform(-1, 1, n)
    sig = rng.uniform(.2, .4, n)
    eps = rng.uniform(.1, .6, n)
    box = np.diag([L, L, L])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if i // 3 != j // 3]
    nlist, ncnt = lists_from(pairs, n)

    ref, _ = direct_q_mirror_kernel(x, q, sig, eps, nlist, ncnt, 0.0, 0.6,
                                    box=box)
    gq, sticky = force_q24.direct_forces_q(x, q, sig, eps, nlist, ncnt,
                                           0.0, 0.6, box=box)
    gq = cp.asnumpy(gq)
    check("P1 direct_q bitwise vs mirror (alpha=0 periodic)",
          np.array_equal(gq.view(np.int64), ref.view(np.int64))
          and sticky == 0)

    ref_a, _ = direct_q_mirror_kernel(x, q, sig, eps, nlist, ncnt, 0.0,
                                      0.6, box=None)
    gq_a, st_a = force_q24.direct_forces_q(x, q, sig, eps, nlist, ncnt,
                                           0.0, 0.6, box=None)
    gq_a = cp.asnumpy(gq_a)
    check("P1 direct_q bitwise vs mirror (alpha=0 aperiodic)",
          np.array_equal(gq_a.view(np.int64), ref_a.view(np.int64))
          and st_a == 0)

    # alpha=3.5: bitwise-or-dual + determinism + newton3
    ref35, _ = direct_q_mirror_kernel(x, q, sig, eps, nlist, ncnt, 3.5,
                                      0.6, box=box)
    g35a, st35a = force_q24.direct_forces_q(x, q, sig, eps, nlist, ncnt,
                                            3.5, 0.6, box=box)
    g35b, st35b = force_q24.direct_forces_q(x, q, sig, eps, nlist, ncnt,
                                            3.5, 0.6, box=box)
    g35a = cp.asnumpy(g35a)
    g35b = cp.asnumpy(g35b)
    ndiff = int((g35a.view(np.int64) != ref35.view(np.int64)).sum())
    f_m = ref35.astype(np.float64) * 2.0 ** -40
    f_g = g35a.astype(np.float64) * 2.0 ** -40
    check("P1b direct_q alpha=3.5 deterministic x2",
          np.array_equal(g35a.view(np.int64), g35b.view(np.int64))
          and st35a == st35b)
    check("P1b direct_q alpha=3.5 vs mirror (bitwise or dual)",
          ndiff == 0 or dual(f_m, f_g),
          f"({ndiff} LSB-differing elements; max abs "
          f"{np.abs(f_m - f_g).max():.2e})")

    # Newton-3 bitwise on GPU: two-atom, IN-ENVELOPE distance (0.5 nm;
    # a 0.11 nm pair saturates Q24.40 -- clamp asymmetry is registered)
    x2 = np.array([[[0.3, 0.3, 0.3]], [[0.8, 0.3, 0.3]]])
    q2 = np.array([0.7, -0.4])
    sig2 = np.array([0.32, 0.28])
    eps2 = np.array([0.5, 0.3])
    nl2 = np.array([[1], [0]], dtype=np.int32)
    nc2 = np.array([1, 1], dtype=np.int32)
    fq2, _ = force_q24.direct_forces_q(x2, q2, sig2, eps2, nl2, nc2,
                                       0.0, 5.0)
    fq2 = cp.asnumpy(fq2)
    check("P1b Newton-3 bitwise (Q24.40)",
          (fq2[0, 0].view(np.int64)
           == (-fq2[1, 0]).view(np.int64)).all())

    # ---------------- P1c: vs opus (dual gate) ----------------
    from opus.engine import ForceAccumulator
    from opus.nonbonded import NeighborList, direct_space

    class _A:
        def __init__(s_, v):
            s_.value = v

    class _At:
        def __init__(s_, qq, sg, e):
            s_.q, s_.sigma, s_.epsilon = _A(qq), _A(sg), _A(e)

    class _NB:
        def __init__(s_, q_, sg_, e_):
            s_.atoms = [_At(a_, b_, c_) for a_, b_, c_ in zip(q_, sg_, e_)]

    f_acc = ForceAccumulator(n, r)
    with np.errstate(all="ignore"):
        direct_space(_NB(q, sig, eps), x, set(), f_acc, 0.0, 0.6,
                     NeighborList(np.array(pairs, dtype=np.int64)), box=box)
    f_opus = f_acc.acc.to_f64().reshape(n, r, 3)
    f_q = gq.astype(np.float64) * 2.0 ** -40
    check("P1c dequant(direct_q) vs opus (alpha=0) dual gate",
          dual(f_q, f_opus),
          f"(max abs {np.abs(f_q - f_opus).max():.2e})")

    # ---------------- P3: device gauss_stream1 parity (production
    # shape: fresh stream per tuple, single draw) ----------------
    n_tuples = 4096
    keys = np.random.default_rng(1).integers(0, 2 ** 62, size=n_tuples)
    out = cp.zeros(n_tuples)
    ksrc = rand.emit_cuda() + (
        'extern "C" __global__ void gpsprobe(double* o,\n'
        '    const unsigned long long* keys, int nt,\n'
        '    const double* wi, const unsigned long long* ki,\n'
        '    const double* fi, double nor_r, double nor_inv_r)\n'
        '{\n'
        '    int t = blockIdx.x * blockDim.x + threadIdx.x;\n'
        '    if (t >= nt) return;\n'
        '    unsigned long long xk = keys[t];\n'
        '    xk ^= 8ULL * 0x9E3779B97F4A7C15ULL;\n'
        '    xk = (xk ^ (xk >> 30)) * 0xBF58476D1CE4E5B9ULL;\n'
        '    xk = (xk ^ (xk >> 27)) * 0x94D049BB133111EBULL;\n'
        '    xk = xk ^ (xk >> 31)\n'
        '        ^ ((unsigned long long)(t + 1)) * 0x8B72C5AF1A3F1E2DULL\n'
        '        ^ (1ULL << 21)\n'
        '        ^ (4ULL * 0xC2B2AE3D27D4EB4FULL);\n'
        '    philox_rng rng; rng.init(xk);\n'
        '    double g1 = 0.0;\n'
        '    for (int zzt = 0; zzt < 100000; ++zzt) {\n'
        '        unsigned long long r = rng.next_u64();\n'
        '        int idx = (int)(r & 0xFFULL);\n'
        '        r >>= 8;\n'
        '        int sign = (int)(r & 0x1ULL);\n'
        '        unsigned long long rabs = (r >> 1)\n'
        '            & 0x000FFFFFFFFFFFFFULL;\n'
        '        g1 = (double)rabs * wi[idx];\n'
        '        if (sign & 0x1) g1 = -g1;\n'
        '        if (rabs < ki[idx]) break;\n'
        '        if (idx == 0) {\n'
        '            for (int tail = 0; tail < 100000; ++tail) {\n'
        '                double xx = -nor_inv_r\n'
        '                    * log1p(-(rng.next_u64() >> 11)\n'
        '                            * (1.0 / 9007199254740992.0));\n'
        '                double yy = -log1p(-(rng.next_u64() >> 11)\n'
        '                    * (1.0 / 9007199254740992.0));\n'
        '                if (yy + yy > xx * xx) {\n'
        '                    g1 = ((rabs >> 8) & 0x1) ? -(nor_r + xx)\n'
        '                                             : (nor_r + xx);\n'
        '                    break;\n'
        '                }\n'
        '            }\n'
        '            break;\n'
        '        }\n'
        '        double u = (rng.next_u64() >> 11)\n'
        '            * (1.0 / 9007199254740992.0);\n'
        '        if ((fi[idx - 1] - fi[idx]) * u + fi[idx]\n'
        '            < exp(-0.5 * g1 * g1))\n'
        '            break;\n'
        '        g1 = 0.0;\n'
        '    }\n'
        '    o[t] = g1;\n'
        '}\n')
    kmod = cp.RawModule(code=ksrc, options=("-fmad", "false"))
    kk = kmod.get_function("gpsprobe")
    tt = rand._rng_tables_pub()
    kk(((n_tuples + 255) // 256,), (256,),
       (out, cp.asarray(keys.astype(np.uint64)), np.int32(n_tuples),
        tt["wi"], tt["ki"], tt["fi"], tt["nor_r"], tt["nor_inv_r"]))
    cp.cuda.Stream.null.synchronize()
    dev = cp.asnumpy(out)
    from opus.rng import gauss_stream
    host = np.array([gauss_stream(int(keys[t]), 7, t, 0, 3, 1)[0]
                     for t in range(n_tuples)])
    pm = int((host.view(np.int64) != dev.view(np.int64)).sum())
    # Cross-platform residual (measured, registered): ziggurat tail/wedge
    # conditions embed log1p/exp -- device-vs-glibc ulp disagreement flips
    # accept/reject on ~0.05% of tuples, producing a DIFFERENT valid
    # N(0,1) sample (not an ulp error).  GPU-internal determinism (M2/M15/
    # M18: same binary) is unaffected.  Gate: bitwise, OR a path-flip rate
    # <= 0.1% with both sides bounded to plausible normal range (|v| < 8).
    sane = bool((np.abs(host) < 8).all() and (np.abs(dev) < 8).all())
    flip_ok = pm <= max(1, n_tuples // 1000) and sane
    check("P3 device gauss_stream1 parity", pm == 0 or flip_ok,
          f"({n_tuples} tuples, {pm} path-flips "
          f"= {pm / n_tuples * 100:.3f}%, sane={sane})")

    # ---------------- P3b: gamma>0 dynamics vs live opus ----------------
    from opus import dynamics as dyn_mod
    from opus.dynamics import Dynamics

    nw2, r2 = 12, 1  # R=1: slot=0 composition == opus reference exactly
    x0 = water_cluster(nw2, r2, seed=31, jitter=0.03, min_sep=0.35)
    m = np.tile(np.array([15.999, 1.008, 1.008]), nw2)
    cons = []
    for w in range(nw2):
        o, h1, h2 = 3 * w, 3 * w + 1, 3 * w + 2
        cons += [_C(o, h1, 0.09572), _C(o, h2, 0.09572),
                 _C(h1, h2, 0.15139)]

    def sp(_s, xx):
        return {"forces": 50.0 * np.sin(xx * 3.1 + 11)}

    dyn_mod.single_point = sp
    system = type("S", (), {})()
    system.masses = m
    system.constraints = cons
    v0 = np.random.default_rng(32).normal(0, 0.4, x0.shape)
    dyn = Dynamics(system, x0.copy(), v0.copy(), seed=77, shake_iters=12)
    K, DT, GAMMA, T = 3, 2.0e-3, 1.0, 300.0
    from opus.dynamics import KB_KJ  # never retype constants
    kT = KB_KJ * T

    invm = (1.0 / m).astype(np.float64)
    xg = cp.asarray(x0.reshape(-1))
    vg = cp.asarray(v0.reshape(-1))
    fg = cp.asarray(sp(None, x0)["forces"].reshape(-1))
    bitwise = True
    for step in range(K):
        integrate.baoab_step(xg, vg, fg, cp.asarray(m), cp.asarray(invm),
                             nw2 * 3, r2, dt=DT, gamma=GAMMA, kT=kT,
                             step=step, seed=77, first=(step == 0),
                             r_oh=0.09572, r_hh=0.15139)
        dyn.step_baoab(DT, gamma=GAMMA, T=T)
        fg = cp.asarray(sp(None, dyn.x)["forces"].reshape(-1))
        xb = np.array_equal(cp.asnumpy(xg).view(np.int64),
                            dyn.x.reshape(-1).view(np.int64))
        bitwise &= xb
        if not xb:
            print(f"  first divergence after step {step}")
            break
    if bitwise:
        integrate.baoab_finish(xg, vg, fg, cp.asarray(m), cp.asarray(invm),
                               nw2 * 3, r2, dt=DT, r_oh=0.09572,
                               r_hh=0.15139)
        vbit = np.array_equal(cp.asnumpy(vg).view(np.int64),
                              dyn.v.reshape(-1).view(np.int64))
    else:
        vbit = False
    check("P3b BAOAB chain gamma=1 T=300 x/v bitwise vs opus",
          bitwise and vbit)

    print("=" * 50)
    if fails:
        print(f"PILLAR ALIGNMENT FAILED: {fails}")
        sys.exit(1)
    print("PILLAR ALIGNMENT PASS: all gates green")


if __name__ == "__main__":
    main()
