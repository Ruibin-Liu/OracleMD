"""Delta debugging (spec §12 M0 deliverable): shrink a failing single-point
case by removing atoms and force terms while the failure persists.

Greedy 1-delta shrink: try removing one atom at a time (relabeling all
terms), then one force-term group at a time.  Each candidate is validated
against the *oracle pair* supplied by the caller (fails(ir, pos) -> bool).
Reference-quality (slow) — it drives full OpenMM comparisons per candidate.
"""
from __future__ import annotations

import numpy as np

from .ir import (AngleTerm, AtomParams, BondTerm, ExceptionTerm, IRSystem,
                 NonbondedParams, ParamExpr, TorsionTerm)


def _drop_atom(ir, pos, k: int):
    keep = [i for i in range(ir.n_atoms) if i != k]
    remap = {old: new for new, old in enumerate(keep)}

    def mp(idx: int) -> int:
        return remap[idx]

    bonds = [BondTerm((mp(t.atoms[0]), mp(t.atoms[1])), t.length, t.k)
             for t in ir.bonds if k not in t.atoms]
    angles = [AngleTerm(tuple(mp(a) for a in t.atoms), t.theta0, t.k)
              for t in ir.angles if k not in t.atoms]
    tors = [TorsionTerm(tuple(mp(a) for a in t.atoms), t.periodicity,
                        t.phase, t.k)
            for t in ir.torsions if k not in t.atoms]
    nb = ir.nonbonded
    nb2 = None
    if nb is not None:
        nb2 = NonbondedParams(
            cutoff=nb.cutoff, ewald_alpha=nb.ewald_alpha, grid=nb.grid,
            spline_order=nb.spline_order, coulomb14scale=nb.coulomb14scale,
            use_dispersion_correction=nb.use_dispersion_correction,
            periodic=nb.periodic,
            exceptions_use_periodic=nb.exceptions_use_periodic,
            atoms=[AtomParams(ParamExpr(nb.atoms[i].q.value),
                              ParamExpr(nb.atoms[i].sigma.value),
                              ParamExpr(nb.atoms[i].epsilon.value))
                   for i in keep],
            exceptions=[ExceptionTerm(mp(e.a), mp(e.b), e.chargeProd,
                                      e.sigma, e.epsilon)
                        for e in nb.exceptions
                        if e.a != k and e.b != k])
    ir2 = IRSystem(len(keep), bonds, angles, tors, nb2, ir.has_cm_remover,
                   ir.box, ir.masses[keep])
    return ir2, pos[keep]


def shrink_system(ir, pos, fails, verbose=False):
    """Greedy shrink; returns (ir, pos, reason_of_final_failure)."""
    cur_ir, cur_pos = ir, pos
    changed = True
    while changed and cur_ir.n_atoms > 1:
        changed = False
        for k in range(cur_ir.n_atoms):
            cand_ir, cand_pos = _drop_atom(cur_ir, cur_pos, k)
            if cand_ir.n_atoms == 0:
                continue
            try:
                if fails(cand_ir, cand_pos):
                    cur_ir, cur_pos = cand_ir, cand_pos
                    changed = True
                    if verbose:
                        print(f"delta: dropped atom {k} -> {cur_ir.n_atoms}")
                    break
            except Exception as e:  # degenerate configs may error; skip
                if verbose:
                    print(f"delta: atom {k} candidate errored: {e}")
    why = fails(cur_ir, cur_pos)
    return cur_ir, cur_pos, why
