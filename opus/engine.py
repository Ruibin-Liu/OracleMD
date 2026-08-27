"""Engine: single-point energy/force for R replicas (spec §12 M0).

Semantics:
  x: (N, R, 3) named axes (atom, slot, xyz);
  forces accumulate per *contribution* into Q24.40 fixed-point per
  (atom, slot) — bitwise order-independent (pillar 1);
  per-Force energy decomposition returned for A1's per-term comparison.
"""
from __future__ import annotations

import numpy as np

from . import fxp, pme as pme_mod
from .bonded import bonded_terms_energy_forces
from .ingest import gate_max_initial_force, gate_parameter_envelope
from .ir import IRSystem
from .nonbonded import (build_union_list, direct_space,
                        dispersion_tail, exceptions_energy_forces)


class ForceAccumulator:
    """(N, R, 3) Q24.40 accumulator with per-contribution quantization."""

    def __init__(self, n_atoms: int, n_replicas: int):
        self.acc = fxp.q24_40_forces((n_atoms, n_replicas, 3))
        self.shape = (n_atoms, n_replicas, 3)

    def add_to(self, atom: int, contrib: np.ndarray):
        """contrib: (R, 3) float64 — quantized once, added exactly.

        Overflow-safe: contributions beyond the int64 raw range (i.e.
        beyond the Q24.40 physical format bound) are detected in the
        float domain BEFORE the cast (casting a huge float to int64 wraps
        silently — that is the disaster this guards against).
        """
        scaled = np.rint(contrib * (1 << self.acc.frac_bits))
        bad = ~np.isfinite(scaled) | (np.abs(scaled) > self.acc.vmax)
        if bad.any():
            self.acc.sticky_overflow = True
            self.acc.n_saturations += int(bad.sum())
            scaled = np.where(bad, np.sign(scaled) * self.acc.vmax, scaled)
        q = scaled.astype(np.int64)
        cur = self.acc.acc[atom]
        s = cur.astype(object) + q.astype(object)
        over = (s > self.acc.vmax) | (s < self.acc.vmin)
        if over.any():
            self.acc.sticky_overflow = True
            self.acc.n_saturations += int(over.sum())
            s = np.where(s > self.acc.vmax, self.acc.vmax, s)
            s = np.where(s < self.acc.vmin, self.acc.vmin, s)
        self.acc.acc[atom] = np.asarray(s, dtype=np.int64)

    def forces(self) -> np.ndarray:
        return self.acc.to_f64()


def single_point(system: IRSystem, x: np.ndarray, *,
                 use_union_list: bool = True, skin: float = 0.28,
                 apply_gates: bool = False) -> dict:
    from .energy import EnergyAccumulator
    """x: (N, R, 3). Returns {'energies': {force: E}, 'forces': (N,R,3),
    'sticky_overflow': bool}.

    Energies are identical across replicas only if x is; per-force energy is
    summed over replicas (A1 compares single-replica systems).
    """
    if x.ndim != 3:
        raise ValueError("x must be (N, R, 3)")
    N, R, _ = x.shape
    assert N == system.n_atoms
    f_acc = ForceAccumulator(N, R)
    e_acc = EnergyAccumulator(R)

    bonded_terms_energy_forces("bond", system.bonds, x, f_acc, e_acc)
    bonded_terms_energy_forces("angle", system.angles, x, f_acc, e_acc)
    bonded_terms_energy_forces("torsion", system.torsions, x, f_acc, e_acc)

    nb = system.nonbonded
    if nb is not None:
        excl = {(e.a, e.b) for e in nb.exceptions}
        alpha = nb.ewald_alpha
        rc = nb.cutoff
        if nb.periodic:
            assert system.box is not None
            grid_fx = fxp.q16_48_grid(nb.grid)
            g = pme_mod.PmeGrid(system.box, nb.grid, alpha)
            q = np.array([a.q.value for a in nb.atoms])
            e_recip = 0.0
            f_recip = np.zeros((N, R, 3))
            inv_box = g.inv_box
            for r in range(R):
                frac = pme_mod.frac_coords(x[:, r, :], inv_box)
                e_r, F = pme_mod.reciprocal_energy(g, q, frac, grid_fx)
                e_recip += e_r
                f_recip[:, r, :] = F
                if grid_fx.sticky_overflow:
                    raise RuntimeError("Q16.48 grid overflow (sticky)")
            e_acc.add("recip", np.repeat(e_recip / max(R, 1), R))
            e_self = pme_mod.self_energy_terms(q, alpha)
            e_acc.add("self", np.repeat(float(np.sum(e_self)), R))
            for i in range(N):
                f_acc.add_to(i, f_recip[i])
        else:
            alpha = 0.0  # plain cutoff Coulomb (erfc with alpha=0 == 1/r)

        if use_union_list:
            lst = build_union_list(x, rc * rc, skin * skin,
                                   box=system.box if nb.periodic else None)
        else:
            lst = None
        box = system.box if nb.periodic else None
        direct_space(nb, x, excl, f_acc, alpha, rc, lst, box=box, e_acc=e_acc)
        exc_box = box if (box is not None and nb.exceptions_use_periodic) else None
        exceptions_energy_forces(nb, x, f_acc, alpha, box=exc_box, e_acc=e_acc)
        if nb.periodic and nb.use_dispersion_correction:
            V = float(abs(np.linalg.det(system.box)))
            e_acc.add("tail", dispersion_tail(nb, V, rc))

    forces = f_acc.forces()
    energies = e_acc.energies()
    if apply_gates and nb is not None:
        gate_parameter_envelope(nb)
        gate_max_initial_force(forces)
    return {"energies": energies,
            "forces": forces,
            "sticky_overflow": f_acc.acc.sticky_overflow or e_acc.sticky_overflow}
