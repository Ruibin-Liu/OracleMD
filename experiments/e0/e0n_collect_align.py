#!/usr/bin/env python3
"""E0n: production-collection alignment bench (gpu/pme, constrain, integrate).

A1-alignment for the M2 kernel collection, E0m-pattern:
local opus oracle -> npz -> pod GPU comparison.

  local (no CUDA):  python e0n_collect_align.py gen <out-dir>
  pod (A100/cupy):  python e0n_collect_align.py run <out-dir>

Gates:
  - spread grid (int64 Q16.48): BITWISE vs opus.pme.spread + fxp
    (both v1 global-atomic oracle path and tile production path; tile ==
    v1 bitwise is the E0h2 precedent).  Includes seam-stressed atoms.
  - interp forces: A1 dual gate (rel < 1e-10 or abs < 1e-8) vs opus
    reciprocal_energy forces, with the SAME phi grid fed to both sides
    (numpy FFT chain; isolates the gather; FFT bitwise work is E0e).
    Not bitwise by design: numpy pairwise 64-point sum vs sequential.
  - shake + rattle: BITWISE vs opus.dynamics (12 fixed iters, Q-016).
  - streaming baoab chain (gamma=0): BITWISE x, v after K steps + tail
    vs a real opus Dynamics.step_baoab chain with the same exogenous
    force sequence.  gamma > 0 NOT aligned (pillar-3 RNG pending).
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

NG, TC, L, ALPHA = 32, 8, 2.0, 2.0
DT, KSTEPS = 2.0e-3, 3
R_OH, R_HH = 0.09572, 0.15139


class _C:
    __slots__ = ("a", "b", "distance")

    def __init__(self, a, b, d):
        self.a, self.b, self.distance = a, b, d


def water_constraints(nw):
    out = []
    for w in range(nw):
        o, h1, h2 = 3 * w, 3 * w + 1, 3 * w + 2
        out += [_C(o, h1, R_OH), _C(o, h2, R_OH), _C(h1, h2, R_HH)]
    return out


def water_cluster(nw, r, seed, jitter=0.05, seam=False):
    rng = np.random.default_rng(seed)
    ideal = np.array([[0.0, 0.0, 0.0],
                      [R_OH, 0.0, 0.0],
                      [0.02377, 0.09272, 0.0]])
    x = np.empty((nw * 3, r, 3))
    for w in range(nw):
        c = rng.uniform(0.3, L - 0.3, 3)
        qq, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        for k in range(3):
            base = c + ideal[k] @ qq + jitter * rng.normal(size=3)
            for rep in range(r):
                x[3 * w + k, rep] = base + 0.01 * rng.normal(size=3)
    x %= L
    if seam:  # park some atoms across the ng seam (both edges)
        x[::5] = rng.uniform(0.0, 0.02, x[::5].shape)
        x[2::7] = rng.uniform(L - 0.02, L - 1e-12, x[2::7].shape)
    return x


def forces_field(x, seed=0):
    return {"forces": 50.0 * np.sin(x * 3.1 + seed)}


# ------------------------------------------------------------------ gen

def gen(outdir):
    from opus.dynamics import Dynamics, project_velocities, shake_positions
    import opus.dynamics as dyn_mod
    from opus.fxp import FixedPointAccumulator
    from opus.pme import PmeGrid, reciprocal_energy, spread
    os.makedirs(outdir, exist_ok=True)
    box = np.diag([L, L, L])

    # --- PME oracle (spread grid + phi + gather forces) ---
    nw, r = 30, 2
    x = water_cluster(nw, r, seed=3, seam=True)
    q = np.random.default_rng(4).uniform(-1, 1, nw * 3)
    g = PmeGrid(box, (NG, NG, NG), alpha=ALPHA, order=4)
    inv_box = g.inv_box
    grid_ref = np.zeros((r, NG, NG, NG), dtype=np.int64)
    phi = np.zeros((r, NG ** 3))
    f_ref = np.zeros((nw * 3, r, 3))
    for rep in range(r):
        frac = x[:, rep, :] @ inv_box
        acc = FixedPointAccumulator((NG, NG, NG), int_bits=16, frac_bits=48)
        spread(frac, q, g, acc)
        grid_ref[rep] = acc.acc.reshape(NG, NG, NG)
        qg = acc.to_f64().reshape(NG, NG, NG)
        qhat = np.fft.fftn(qg)
        phi[rep] = (g.n_grid_total * np.fft.ifftn(qhat * g.c).real).ravel()
        _, f_rep = reciprocal_energy(g, q, frac, acc)
        f_ref[:, rep, :] = f_rep
    np.savez_compressed(
        os.path.join(outdir, "pme.npz"),
        x=x, q=q, box=box, grid_ref=grid_ref.reshape(r, -1), phi=phi,
        f_ref=f_ref, ng=NG, alpha=ALPHA)
    print(f"pme.npz: x{x.shape} grid{grid_ref.shape} "
          f"charge_r0={grid_ref[0].sum() / float(1 << 48):.6f} "
          f"(q.sum={q.sum():.6f})")

    # --- SHAKE/RATTLE oracle ---
    nw, r = 25, 2
    x0 = water_cluster(nw, r, seed=7, jitter=0.08, seam=True)
    m = np.tile(np.array([15.999, 1.008, 1.008]), nw)
    invm = 1.0 / m
    cons = water_constraints(nw)
    xs = shake_positions(x0.copy(), x0.copy(), cons, invm, n_iter=12)
    rng = np.random.default_rng(8)
    v0 = rng.normal(0, 0.6, x0.shape)
    vs = project_velocities(v0.copy(), x0, cons, invm, n_iter=12)
    np.savez_compressed(
        os.path.join(outdir, "shake.npz"),
        x0=x0, v0=v0, m=m, x_ref=xs, v_ref=vs, iters=12,
        r_oh=R_OH, r_hh=R_HH)
    print(f"shake.npz: x{x0.shape}")

    # --- streaming baoab chain oracle (gamma=0, real Dynamics chain) ---
    nw, r = 12, 2
    x0 = water_cluster(nw, r, seed=11, jitter=0.05)
    m = np.tile(np.array([15.999, 1.008, 1.008]), nw)
    cons = water_constraints(nw)

    def sp(_sys, xx):
        return forces_field(xx, seed=11)

    dyn_mod.single_point = sp
    system = type("S", (), {})()
    system.masses = m
    system.constraints = cons
    rng = np.random.default_rng(12)
    v0 = rng.normal(0, 0.4, x0.shape)
    dyn = Dynamics(system, x0.copy(), v0.copy(), seed=0, shake_iters=12)
    f_seq = [forces_field(x0, seed=11)["forces"]]
    for _ in range(KSTEPS):
        dyn.step_baoab(DT, gamma=0.0)
        f_seq.append(forces_field(dyn.x, seed=11)["forces"])
    np.savez_compressed(
        os.path.join(outdir, "baoab.npz"),
        x0=x0, v0=v0, m=m, dt=DT, ksteps=KSTEPS,
        f=np.stack(f_seq),  # (K+1, N, R, 3): held force per chain position
        x_ref=dyn.x, v_ref=dyn.v, r_oh=R_OH, r_hh=R_HH)
    print(f"baoab.npz: chain K={KSTEPS} x{x0.shape}")


# ------------------------------------------------------------------ run

def dual_gate(a, b, rel=1e-10, abs_=1e-8):
    """A1 dual gate, per-element (spec 11.1): pass if EVERY element is
    abs-small OR rel-small."""
    da = np.abs(a - b)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    ok = bool(np.all((da < abs_) | (da / scale < rel)))
    return ok, float(da.max()), float((da / scale).max())


def run(outdir):
    import cupy as cp
    from gpu import constrain, integrate, pme
    fails = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    # --- PME ---
    d = np.load(os.path.join(outdir, "pme.npz"))
    x, q, box = d["x"], d["q"], d["box"]
    grid_ref, phi, f_ref = d["grid_ref"], d["phi"], d["f_ref"]
    ng = int(d["ng"])
    r_rep = x.shape[1]
    gv1 = cp.asnumpy(pme.spread(x, q, box, ng=ng, mode="v1")).reshape(r_rep, -1)
    check("spread v1 bitwise vs opus", np.array_equal(gv1, grid_ref),
          f"(sum check r0 {int(gv1[0].sum()) / float(1 << 48):.6f})")
    gt = cp.asnumpy(pme.spread(x, q, box, ng=ng, mode="tile")).reshape(r_rep, -1)
    check("spread tile bitwise vs opus", np.array_equal(gt, grid_ref))
    check("spread tile bitwise vs v1", np.array_equal(gt, gv1),
          "(E0h2 precedent)")
    F = cp.asnumpy(pme.interp_forces(x, q, phi, box, ng=ng))
    ok, mabs, mrel = dual_gate(F, f_ref)
    check("interp forces dual gate", ok,
          f"(max abs {mabs:.3e}, max rel {mrel:.3e})")

    # --- SHAKE / RATTLE ---
    d = np.load(os.path.join(outdir, "shake.npz"))
    x0, v0, m = d["x0"], d["v0"], d["m"]
    x_ref, v_ref = d["x_ref"], d["v_ref"]
    invm = (1.0 / m).astype(np.float64)
    xg = cp.asarray(x0.reshape(-1))
    constrain.shake_water(xg, invm, r_oh=float(d["r_oh"]),
                          r_hh=float(d["r_hh"]), iters=int(d["iters"]))
    xg_bit = cp.asnumpy(xg).view(np.int64)
    check("shake bitwise vs opus",
          np.array_equal(xg_bit, x_ref.reshape(-1).view(np.int64)))
    vg = cp.asarray(v0.reshape(-1))
    constrain.project_water(vg, cp.asarray(x0.reshape(-1)), invm,
                            iters=int(d["iters"]))
    vg_bit = cp.asnumpy(vg).view(np.int64)
    check("rattle bitwise vs opus",
          np.array_equal(vg_bit, v_ref.reshape(-1).view(np.int64)))

    # --- streaming baoab chain ---
    d = np.load(os.path.join(outdir, "baoab.npz"))
    x0, v0, m = d["x0"], d["v0"], d["m"]
    f_seq, dt = d["f"], float(d["dt"])
    k = int(d["ksteps"])
    x_ref, v_ref = d["x_ref"], d["v_ref"]  # rebind: d just changed file
    nw = x0.shape[0] // 3
    r = x0.shape[1]
    invm = (1.0 / m).astype(np.float64)
    xg = cp.asarray(x0.reshape(-1))
    vg = cp.asarray(v0.reshape(-1))
    fm = m
    for step in range(k):
        fk = cp.asarray(f_seq[step].reshape(-1))
        integrate.baoab_step(xg, vg, fk, fm, invm, nw * 3, r,
                             dt=dt, gamma=0.0, kT=0.0, step=step, seed=0,
                             first=(step == 0),
                             r_oh=float(d["r_oh"]), r_hh=float(d["r_hh"]))
    fk = cp.asarray(f_seq[k].reshape(-1))
    integrate.baoab_finish(xg, vg, fk, fm, invm, nw * 3, r,
                           dt=dt, r_oh=float(d["r_oh"]),
                           r_hh=float(d["r_hh"]))
    check("baoab chain x bitwise vs opus",
          np.array_equal(cp.asnumpy(xg).view(np.int64),
                         x_ref.reshape(-1).view(np.int64)))
    check("baoab chain v bitwise vs opus",
          np.array_equal(cp.asnumpy(vg).view(np.int64),
                         v_ref.reshape(-1).view(np.int64)))

    print("=" * 50)
    if fails:
        print(f"ALIGNMENT FAILED: {fails}")
        sys.exit(1)
    print("ALIGNMENT PASS: all gates green")


if __name__ == "__main__":
    mode, outdir = sys.argv[1], sys.argv[2]
    (gen if mode == "gen" else run)(outdir)
