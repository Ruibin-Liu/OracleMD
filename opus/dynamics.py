"""Dynamics (M0.5): BAOAB Langevin (VRORV) + constraints.

Determinism contract (spec §3/§6/§7):
  - forces: fixed-point accumulation (Q24.40) via the single-point machinery;
  - noise: counter-based RNG keyed (seed, absolute step, stable_atom_id, dof);
  - I-021 / M18: one step is a pure function of (x, v, step) — any K-window
    chunking or checkpoint/restart reproduces the trajectory bitwise;
  - constraints: fixed-iteration SHAKE on positions + radial velocity
    projection (pillar 4: no convergence branch; iteration count is a
    manifest constant, Q-016).  The closed-form SETTLE for rigid water is a
    GPU-phase optimization with identical semantics (fixed iteration count
    makes SHAKE order-independent across thread schedules as each sweep is
    sequential in the reference).
"""
from __future__ import annotations

import numpy as np

from . import rng
from .engine import single_point
from .ir import IRSystem

KB_KJ = 0.00831446261815324  # kJ/mol/K


# ---------------------------------------------------------------- constraints

def shake_positions(x, xref, constraints, invm, n_iter):
    """Fixed-iteration pairwise position SHAKE.  x, xref: (N, R, 3)."""
    for _ in range(n_iter):
        for (i, j, d) in ((c.a, c.b, c.distance) for c in constraints):
            dx = x[j] - x[i]
            r2 = np.sum(dx * dx, axis=-1)
            r = np.sqrt(np.where(r2 < 1e-24, 1e-24, r2))
            denom = (invm[i] + invm[j])
            corr = (r - d) / np.where(denom * r < 1e-24, 1e-24, denom * r)
            x[i] += invm[i] * corr[:, None] * dx
            x[j] -= invm[j] * corr[:, None] * dx
    return x


def project_velocities(v, x, constraints, invm, n_iter):
    """Remove radial relative velocity along each constraint bond (RATTLE
    velocity condition, fixed iterations)."""
    for _ in range(n_iter):
        for (i, j, d) in ((c.a, c.b, c.distance) for c in constraints):
            dx = x[j] - x[i]
            r2 = np.sum(dx * dx, axis=-1)
            r = np.sqrt(np.where(r2 < 1e-24, 1e-24, r2))
            dvr = np.sum((v[j] - v[i]) * dx, axis=-1)  # radial rel. velocity
            corr = dvr / np.where((invm[i] + invm[j]) * r2 < 1e-24,
                                  1e-24, (invm[i] + invm[j]) * r2)
            v[i] += invm[i] * corr[:, None] * dx
            v[j] -= invm[j] * corr[:, None] * dx
    return v


def constraint_residual(x, constraints):
    """Max |r_ij - d| over constraints (nan-safe), for the Q-016 assertion."""
    worst = 0.0
    for (i, j, d) in ((c.a, c.b, c.distance) for c in constraints):
        dx = x[j] - x[i]
        r = np.sqrt(np.sum(dx * dx, axis=-1))
        worst = max(worst, float(np.abs(r - d).max()))
    return worst


# ---------------------------------------------------------------- integrator

class Dynamics:
    """BAOAB over R slots; M0.5 scope uses R=1 (multi-replica dynamics is M2)."""

    def __init__(self, system: IRSystem, x, v, seed=0,
                 shake_iters: int = 12):
        self.system = system
        self.x = x
        self.v = v
        self.seed = seed
        self.step = 0
        self.shake_iters = shake_iters
        self.constraints = list(getattr(system, "constraints", []))
        self.invm = 1.0 / system.masses

    def forces(self):
        return single_point(self.system, self.x)

    def step_baoab(self, dt, gamma=0.0, T=300.0):
        """One VRORV step; gamma=0 -> NVE (O step identity)."""
        m = self.system.masses
        R = self.x.shape[1]

        f1 = single_point(self.system, self.x)["forces"]
        # V
        self.v = self.v + 0.5 * dt * f1 / m[:, None, None]
        # R (half)
        x_old = self.x
        self.x = self.x + 0.5 * dt * self.v
        # O
        if gamma > 0.0:
            c = np.exp(-gamma * dt)
            kT = KB_KJ * T
            N = self.x.shape[0]
            noise_scale = np.sqrt(kT * (1 - c * c) / m)   # (N,)
            for i in range(N):
                for d in range(3):
                    xi = rng.gauss_stream(self.seed, self.step, i, 0, d, 1)[0]
                    self.v[i, :, d] = (c * self.v[i, :, d]
                                       + noise_scale[i] * xi)
        # R (half)
        self.x = self.x + 0.5 * dt * self.v
        # constraints (positions vs pre-drift reference)
        if self.constraints:
            self.x = shake_positions(self.x, x_old, self.constraints,
                                     self.invm, self.shake_iters)
        # V' with new forces
        f2 = single_point(self.system, self.x)["forces"]
        self.v = self.v + 0.5 * dt * f2 / m[:, None, None]
        if self.constraints:
            self.v = project_velocities(self.v, self.x, self.constraints,
                                        self.invm, self.shake_iters)
        self.step += 1
        return self

    # ------------------------------------------------ determinism interface

    def snapshot(self):
        """Bitwise state capture (for M18 / I-021 tests)."""
        return dict(x=self.x.copy(), v=self.v.copy(), step=self.step)

    def restore(self, snap):
        self.x = snap["x"].copy()
        self.v = snap["v"].copy()
        self.step = snap["step"]


def total_energy(dyn: Dynamics) -> float:
    res = single_point(dyn.system, dyn.x)
    ke = 0.5 * float(np.sum(dyn.system.masses[:, None, None] * dyn.v ** 2))
    return ke + sum(res["energies"].values())


def kinetic_per_dof(dyn: Dynamics) -> float:
    ke = 0.5 * float(np.sum(dyn.system.masses[:, None, None] * dyn.v ** 2))
    R = dyn.v.shape[1]
    dof = 3 * dyn.v.shape[0] * R
    return 2.0 * ke / dof   # per-dof kT units
