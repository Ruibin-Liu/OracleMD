"""M-series bitwise metamorphic assertions (spec §11.2, tier 1).

M1  batch(R) == R x serial(R=1), bitwise (pillar 5 + fixed-point)
M5a lattice-vector translation, bitwise
M6  atom reordering, single-point bitwise (pillar 1)
M7  shared union list vs full list, bitwise
M8  different skin, bitwise
plus: Q24.40 saturation + sticky flag actually fires (gate machinery test)
"""
import numpy as np
import openmm
import pytest

from opus.engine import single_point
from opus.ingest import ingest_system_xml
from opus.ir import (AngleTerm, AtomParams, BondTerm, ExceptionTerm, IRSystem,
                     NonbondedParams, ParamExpr, TorsionTerm)

from .a1_harness import build_chain_system


def _make_ir(n=6, seed=0, with_nb=True):
    sys_, pos, meta = build_chain_system(n, seed=seed)
    ir = ingest_system_xml(openmm.XmlSerializer.serialize(sys_))
    return ir, pos


def _bitwise(a, b, label):
    assert a.dtype == b.dtype, label
    eq = (a.view(np.int64) == b.view(np.int64)).all() if a.dtype == np.float64 \
        else (a == b).all()
    if not eq:
        d = np.abs(a - b)
        raise AssertionError(f"{label}: NOT bitwise (max abs diff {d.max():.3e})")


def test_m1_batch_equals_serial_bitwise():
    ir, pos = _make_ir(7, seed=2)
    rng = np.random.default_rng(11)
    R = 8
    x = pos[:, None, :] + rng.normal(0, 0.01, (7, R, 3))  # replicas diverge
    res_batch = single_point(ir, x, use_union_list=True)
    assert not res_batch["sticky_overflow"]
    F_batch = res_batch["forces"]
    for r in range(R):
        res_s = single_point(ir, x[:, r:r + 1, :].copy(), use_union_list=False)
        _bitwise(F_batch[:, r, :], res_s["forces"][:, 0, :],
                 f"M1 replica {r}")


def test_m5a_lattice_translation_bitwise():
    """M5a bitwise holds under an exact-representability precondition
    (registered with I-021-family notes in the spec): dyadic box (2.0 nm)
    and coordinates snapped to a dyadic grid (2^-10 nm), so that +A[0] and
    all differences/frac reductions are fp-exact.  A non-dyadic box cannot
    be translated exactly in Cartesian fp — that is a precondition of the
    invariant, not a bug."""
    ir, pos = _make_ir(6, seed=1)
    ir.nonbonded.exceptions_use_periodic = True
    ir.box = np.diag([2.0, 2.0, 2.0])
    pos = np.round(pos / (2 ** -10)) * (2 ** -10) % 2.0
    x1 = pos + ir.box[0]  # exact: dyadic grid + dyadic shift
    assert np.array_equal(x1 - pos, ir.box[0] * np.ones_like(x1)), \
        "precondition: shift must be exact"
    res0 = single_point(ir, pos[:, None, :].copy())
    res1 = single_point(ir, x1[:, None, :].copy())
    _bitwise(res1["forces"], res0["forces"], "M5a forces")
    assert res1["energies"] == res0["energies"], "M5a energies"


def _permute_ir(ir: IRSystem, perm: np.ndarray) -> IRSystem:
    inv = np.argsort(perm)
    bonds = [BondTerm((int(perm[t.atoms[0]]), int(perm[t.atoms[1]])),
                      t.length, t.k) for t in ir.bonds]
    angles = [AngleTerm(tuple(int(perm[a]) for a in t.atoms), t.theta0, t.k)
              for t in ir.angles]
    tors = [TorsionTerm(tuple(int(perm[a]) for a in t.atoms),
                        t.periodicity, t.phase, t.k) for t in ir.torsions]
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
                   for i in inv],  # atom j of new system = old atom inv[j]
            exceptions=[ExceptionTerm(int(perm[e.a]), int(perm[e.b]),
                                      e.chargeProd, e.sigma, e.epsilon)
                        for e in nb.exceptions])
    return IRSystem(ir.n_atoms, bonds, angles, tors, nb2, ir.has_cm_remover,
                    ir.box, ir.masses[inv])


def test_m6_atom_reorder_bitwise():
    ir, pos = _make_ir(6, seed=3)
    rng = np.random.default_rng(5)
    perm = rng.permutation(6)
    inv = np.argsort(perm)
    ir2 = _permute_ir(ir, perm)
    res0 = single_point(ir, pos[:, None, :].copy())
    res1 = single_point(ir2, pos[inv][:, None, :].copy())
    # force on new atom j = force on old atom inv[j]
    _bitwise(res1["forces"][:, 0, :], res0["forces"][inv][:, 0, :], "M6 forces")
    assert res1["energies"] == res0["energies"], "M6 energies"


def test_m7_m8_list_skin_bitwise():
    ir, pos = _make_ir(6, seed=4)
    rng = np.random.default_rng(9)
    R = 4
    x = pos[:, None, :] + rng.normal(0, 0.05, (6, R, 3))
    a = single_point(ir, x, use_union_list=True, skin=0.28)
    b = single_point(ir, x, use_union_list=True, skin=0.45)
    c = single_point(ir, x, use_union_list=False)   # full all-pairs list
    _bitwise(a["forces"], b["forces"], "M8 skin")
    _bitwise(a["forces"], c["forces"], "M7 shared vs full")


def test_saturation_sticky_fires():
    """Gate machinery: overlap two atoms -> LJ diverges -> saturate (never
    wrap) and sticky flag set (B10)."""
    ir, pos = _make_ir(6, seed=0)
    pos = pos.copy()
    pos[5] = pos[0] + 1e-4  # overlap a REGULAR pair (0,5): all pairs of a
                            # 4-atom chain are exceptions (no LJ divergence)
    res = single_point(ir, pos[:, None, :].copy())
    assert res["sticky_overflow"], "overlap must set the sticky flag"
    F = res["forces"]
    assert np.abs(F).max() <= 2 ** 23 * 1.0000001, \
        "saturated force must stay within the Q24.40 format bound"
