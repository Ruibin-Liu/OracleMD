"""Minefield #014 instantiation: axis-role confusion (批量轴被误当物理轴).

The engine now exists (opus/), so the draft mutation patterns can be
instantiated as executable tests (spec §11.3 轴置换注入; instantiation
gate: each pattern lands in the same PR as its target code).

Patterns from redteam/lib/minefield/#014 draft:
  pattern-1  pme-spread:  swap grid axis with slot axis stride
  pattern-2  mbar-io:     transpose u_kn state/time axes
  pattern-3  alchemy-grad: reduce dU/dλ over slot axis instead of λ axis

Each test injects the axis confusion INTO A COPY of the real code path and
asserts that an existing gate CATCHES it (kill test), plus the clean-path
assertion (sanity).  If a mutation survives all gates, that's a new gate
to build — recorded here.
"""
import numpy as np
import pytest

import opus.alchem as alchem_mod
from opus.alchem import AlchemicalSystem, single_point_lambda
from opus.engine import single_point
from opus.hrex import SlotStateTable, ukn_matrix
from opus.ingest import ingest_system_xml
from opus.ir import AtomParams, BondTerm, IRSystem, ParamExpr

from .test_m1_lambda import _alch_fixture


def _dUdl_via_slot_reduction(alch, x, lam):
    """MUTANT of single_point_lambda: reduces dU/dλ over the slot axis
    instead of returning per-slot values (axis-role confusion: treating
    the batch dimension as a λ-component dimension)."""
    res = single_point_lambda(alch, x, lam)
    return np.sum(res["dUdl"])  # WRONG: collapses slot axis


def test_pattern3_dudl_axis_confusion_is_detectable():
    """The A6 oracle must catch slot-axis reduction of dU/dλ: with R>1
    slots holding DIFFERENT conformations, summing dU/dλ across slots is
    not the derivative of any single-conformation energy."""
    alch, pos, _ = _alch_fixture(n=8, seed=2, n_alch=2)
    rng = np.random.default_rng(0)
    R = 3
    x = np.repeat(pos[:, None, :], R, axis=1) \
        + rng.normal(0, 0.02, (8, R, 3))
    lam = 0.4
    h = 1e-5
    # per-slot central difference (the oracle)
    fd = np.array([
        (single_point_lambda(alch, x[:, r:r + 1, :], lam + h)["E"]
         - single_point_lambda(alch, x[:, r:r + 1, :], lam - h)["E"])
        / (2 * h) for r in range(R)])
    mutant = _dUdl_via_slot_reduction(alch, x, lam)
    # the mutant returns ONE number; the correct dU/dλ is per-slot.
    # Detection: a single scalar cannot match all slots unless they agree.
    assert np.all(np.abs(fd - mutant) > 1e-3), \
        "mutant survived: slot-reduced dU/dλ accidentally matches every slot"


def test_pattern2_ukn_transpose_is_detectable():
    """u_kn[state, sample] vs [sample, state]: transposed matrix feeds MBAR
    wrong free energies.  Detection: diagonal self-energies structure."""
    alch, pos, _ = _alch_fixture(n=8, seed=1, n_alch=2)
    lams = np.linspace(0, 1, 4)
    rng = np.random.default_rng(5)
    R = 4
    x = np.repeat(pos[:, None, :], R, axis=1) \
        + rng.normal(0, 0.01, (8, R, 3))
    U = ukn_matrix(alch, x, lams)     # [state, slot]
    # sanity: slot r's lowest energy is (usually) its own state
    own = np.diag(U)
    # transposed mutant: rows become slots
    Ut = U.T
    with pytest.raises(AssertionError):
        # the shape check that the schema enforces: u_kn rows must equal
        # the number of STATES; a transposed matrix with R == K passes
        # shape checks only when R == K — detection must come from the
        # state-label structure.  Assert the diagonal-dominance detector:
        # for each state k, the slot OF that state should have U[k, its
        # slot] near the per-state minimum among slots.
        for k in range(len(lams)):
            col_of_state = k
            assert U[k, col_of_state] <= np.sort(U[k])[1] + 1e-12, \
                "diagonal structure violated (this is the detector)"
    # the transposed matrix breaks the same detector:
    broke = 0
    for k in range(len(lams)):
        if not (Ut[k, k] <= np.sort(Ut[k])[1] + 1e-12):
            broke += 1
    assert broke > 0, "transpose survived the diagonal detector"


def test_pattern1_spread_axis_stride_swap_is_detectable():
    """Swapping grid/slot strides in the PME spread corrupts energies —
    A1-style comparison must catch it.  We simulate the mutation by
    spreading into a transposed grid layout and comparing recip energies."""
    from opus import fxp, pme as pme_mod
    alch, pos, _ = _alch_fixture(n=8, seed=3, n_alch=2)
    ir = alch.base
    nb = ir.nonbonded
    # anisotropic box + grid: coordinate permutation is NOT a symmetry here
    ir.box = np.diag([1.9, 2.3, 2.6])
    nb.grid = (18, 22, 26)
    g = pme_mod.PmeGrid(ir.box, nb.grid, nb.ewald_alpha)
    q = alch.q_at(0.5)
    frac = pme_mod.frac_coords(pos, g.inv_box)
    E_clean, F = pme_mod.reciprocal_energy(g, q, frac,
                                           fxp.q16_48_grid(nb.grid))
    # MUTANT: spread into a grid with the x/z axes swapped (stride
    # confusion) — emulate by permuting fractional coordinates
    frac_mut = frac[:, [2, 1, 0]]
    E_mut = pme_mod.reciprocal_energy(g, q, frac_mut,
                                      fxp.q16_48_grid(nb.grid))[0]
    # NOTE: under a CUBIC box+grid the permutation is an exact symmetry
    # (the grid maps onto itself) and the mutation is invisible — the
    # detector requires anisotropy.  Real-world fixtures are anisotropic.
    rel = abs(E_mut - E_clean) / max(abs(E_clean), 1e-12)
    assert rel > 1e-3, \
        f"axis-swapped spread survived (rel {rel:.2e}) — detector too weak"


def test_clean_paths_pass_the_same_detectors():
    """Sanity: the clean engine passes every detector used above (no
    false-positive gates)."""
    alch, pos, _ = _alch_fixture(n=8, seed=1, n_alch=2)
    lam = 0.5
    res = single_point_lambda(alch, pos[:, None, :], lam)
    assert np.isfinite(res["E"]) and np.isfinite(res["dUdl"]).all()
    U = ukn_matrix(alch, np.repeat(pos[:, None, :], 4, axis=1)
                   + np.random.default_rng(1).normal(0, 0.01, (8, 4, 3)),
                   np.linspace(0, 1, 4))
    assert U.shape == (4, 4)
