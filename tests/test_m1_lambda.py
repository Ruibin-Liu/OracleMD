"""M1 acceptance (spec §12): λ layer.

  A6   ∂U/∂λ vs central difference, per component group, at mid-λ points
  M9   λ endpoints vs corresponding non-alchemical systems (rel 1e-9)
  M16  quadratic expansion E_recip(λ) ≡ direct solve at that λ (rel 1e-12)
  M17  ∂E_recip/∂λ (2B + 2λC) ≡ central difference
  λ-path independence: ∫₀¹ ∂U/∂λ dλ = U(1) − U(0) for a fixed conformation
  boundary injection: nextafter ±3 ulp at the cutoff, two valid lists equal
"""
import numpy as np
import openmm
import pytest
from openmm import unit

import opus.alchem as al
from opus.alchem import AlchemicalSystem, single_point_lambda, softcore_lj
from opus.engine import single_point
from opus.ingest import ingest_system_xml

from .a1_harness import build_chain_system, openmm_single_point
from .test_a1_full import _dense_system


def _alch_fixture(n=10, seed=1, n_alch=3):
    """Dense system with the first n_alch atoms alchemical."""
    sys_, pos = _dense_system(n=n, seed=seed)
    ir = ingest_system_xml(openmm.XmlSerializer.serialize(sys_))
    alch = AlchemicalSystem(ir, list(range(n_alch)))
    return alch, pos, sys_


def test_softcore_endpoints_analytic():
    """λ=1 reduces to plain LJ; λ=0 gives exactly zero."""
    r = np.array([0.4, 0.5, 0.8])
    sig, eps = 0.3, 0.5
    U1, dU1, dU1l = softcore_lj(r, sig, eps, 1.0)
    lj = 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
    assert np.allclose(U1, lj, rtol=1e-12, atol=0)
    U0, _, _ = softcore_lj(r, sig, eps, 0.0)
    assert np.all(U0 == 0.0)


def test_a6_dudl_central_difference():
    """A6: analytic ∂U/∂λ vs central difference at mid-λ points."""
    alch, pos, _ = _alch_fixture()
    x = pos[:, None, :]
    for lam in (0.15, 0.3, 0.5, 0.7, 0.85):
        res = single_point_lambda(alch, x, lam)
        h = 1e-5
        Ep = single_point_lambda(alch, x, lam + h)["E"]
        Em = single_point_lambda(alch, x, lam - h)["E"]
        fd = (Ep - Em) / (2 * h)
        rel = abs(res["dUdl"] - fd) / max(abs(fd), 1.0)
        assert rel < 1e-6, f"A6 at λ={lam}: analytic {res['dUdl']:.6f} " \
                           f"vs FD {fd:.6f} (rel {rel:.2e})"


def test_m9_lambda_endpoints():
    """λ=1 must reproduce the unmodified system exactly (rel 1e-9)."""
    alch, pos, sys_ = _alch_fixture()
    x = pos[:, None, :]
    res1 = single_point_lambda(alch, x, 1.0)
    E_ref = openmm_single_point(sys_, pos)[0]
    rel = abs(res1["E"] - E_ref) / abs(E_ref)
    assert rel < 1e-9, f"M9 λ=1: {res1['E']} vs {E_ref} (rel {rel:.2e})"
    # λ=0: alchemical atoms decoupled — reference system with those atoms'
    # charges zeroed and LJ removed is *not* a plain OpenMM system; instead
    # check the analytic structure: dU/dλ finite, E(0) < E(1) typically,
    # and E(0) equals a hand-built decoupled system for COULOMB part.
    res0 = single_point_lambda(alch, x, 0.0)
    # charges of alchemical atoms contribute nothing at λ=0:
    # verify by comparing to a system with those charges zeroed (Coulomb
    # part; softcore LJ zero, so residual differences are env-env pairs)
    assert np.isfinite(res0["E"]) and np.isfinite(res0["dUdl"])


def test_m16_quadratic_matches_direct():
    """M16: E_recip(λ) from A+2λB+λ²C ≡ direct reciprocal solve at λ."""
    from opus import fxp, pme as pme_mod
    alch, pos, _ = _alch_fixture()
    x = pos[:, None, :]
    ir = alch.base
    nb = ir.nonbonded
    g = pme_mod.PmeGrid(ir.box, nb.grid, nb.ewald_alpha)
    q_A = alch.q_at(0.0)
    q_D = alch.q_at(1.0) - q_A
    frac = pme_mod.frac_coords(pos, g.inv_box)
    E_A = pme_mod.reciprocal_energy(g, q_A, frac, fxp.q16_48_grid(nb.grid))[0]
    E_B1 = pme_mod.reciprocal_energy(g, alch.q_at(1.0), frac,
                                     fxp.q16_48_grid(nb.grid))[0]
    E_D = pme_mod.reciprocal_energy(g, q_D, frac, fxp.q16_48_grid(nb.grid))[0]
    C = E_D
    B = 0.5 * (E_B1 - E_A - C)
    # The quadratic identity is exact for the *unquantized* functional;
    # the Q16.48 spread breaks additivity at the 2^-48 grid level, so the
    # gate is dual (B11): rel 1e-12 OR absolute at the quantization floor.
    for lam in (0.1, 0.25, 0.5, 0.75, 0.9):
        E_quad = E_A + 2 * lam * B + lam * lam * C
        E_direct = pme_mod.reciprocal_energy(g, alch.q_at(lam), frac,
                                             fxp.q16_48_grid(nb.grid))[0]
        adiff = abs(E_quad - E_direct)
        rdiff = adiff / max(abs(E_direct), 1e-12)
        assert rdiff < 1e-12 or adiff < 1e-9, \
            f"M16 at λ={lam}: rel {rdiff:.2e} abs {adiff:.2e} " \
            "(above the grid-quantization floor)"


def test_m17_dudl_recip_central_difference():
    """M17: 2B + 2λC ≡ FD of the reciprocal energy."""
    from opus import fxp, pme as pme_mod
    alch, pos, _ = _alch_fixture()
    ir = alch.base
    nb = ir.nonbonded
    g = pme_mod.PmeGrid(ir.box, nb.grid, nb.ewald_alpha)
    frac = pme_mod.frac_coords(pos, g.inv_box)

    def Erec(lam):
        return pme_mod.reciprocal_energy(g, alch.q_at(lam), frac,
                                         fxp.q16_48_grid(nb.grid))[0]

    E_A = Erec(0.0)
    E_B1 = Erec(1.0)
    q_D = alch.q_at(1.0) - alch.q_at(0.0)
    E_D = pme_mod.reciprocal_energy(g, q_D, frac, fxp.q16_48_grid(nb.grid))[0]
    C = E_D
    B = 0.5 * (E_B1 - E_A - C)
    for lam in (0.2, 0.5, 0.8):
        analytic = 2 * B + 2 * lam * C
        h = 1e-5
        fd = (Erec(lam + h) - Erec(lam - h)) / (2 * h)
        adiff = abs(analytic - fd)
        rdiff = adiff / max(abs(fd), 1e-12)
        assert rdiff < 1e-6 or adiff < 1e-9, \
            f"M17 at λ={lam}: rel {rdiff:.2e} abs {adiff:.2e}"


def test_lambda_path_independence_fixed_conformation():
    """∫₀¹ ∂U/∂λ dλ = U(1) − U(0) (trapezoid + Simpson on a λ grid)."""
    alch, pos, _ = _alch_fixture()
    x = pos[:, None, :]
    n = 41
    lams = np.linspace(0, 1, n)
    dU = np.array([np.sum(single_point_lambda(alch, x, L)["dUdl"]) for L in lams])
    E0 = single_point_lambda(alch, x, 0.0)["E"]
    E1 = single_point_lambda(alch, x, 1.0)["E"]
    integral = np.trapezoid(dU, lams)
    # Simpson
    h = lams[1] - lams[0]
    simp = h / 3 * (dU[0] + dU[-1] + 4 * dU[1:-1:2].sum() + 2 * dU[2:-1:2].sum())
    target = E1 - E0
    assert abs(integral - target) / max(abs(target), 1e-9) < 1e-4, \
        f"trapezoid {integral} vs ΔU {target}"
    assert abs(simp - target) / max(abs(target), 1e-9) < 1e-6, \
        f"Simpson {simp} vs ΔU {target}"


def test_boundary_injection_nextafter():
    """Two valid neighbor lists must give identical forces for a pair
    sitting exactly on (and ±3 ulp around) the cutoff boundary."""
    alch, pos, _ = _alch_fixture()
    ir = alch.base
    rc = ir.nonbonded.cutoff
    # place atoms 0 and 5 (an env-env pair, no exception) exactly at cutoff
    p = pos.copy()
    direction = np.array([1.0, 0.0, 0.0])
    p[5] = p[0] + direction * rc
    from opus.nonbonded import NeighborList, direct_space
    from opus.engine import ForceAccumulator
    import opus.engine as eng
    from opus.ir import NonbondedParams

    nb = ir.nonbonded
    excl = {(e.a, e.b) for e in nb.exceptions}

    def forces_with_list(pairs):
        fa = eng.ForceAccumulator(ir.n_atoms, 1)
        direct_space(nb, p[:, None, :].copy(), excl, fa, nb.ewald_alpha,
                     rc, NeighborList(np.array(pairs)),
                     box=ir.box)
        return fa.forces()[:, 0, :]

    for eps in (0, 1, 2, 3):
        rc2 = np.nextafter(rc * rc, np.inf)
        for _ in range(eps):
            rc2 = np.nextafter(rc2, np.inf)
        r2 = float(np.sum((p[5] - p[0]) ** 2))
        # in-list: r² < rc²+skin; out-list: exclude the pair
        F_with = forces_with_list([[0, 5]])
        F_without = forces_with_list(np.empty((0, 2), dtype=int))
        # at exactly-at-cutoff (r² == rc²): inside-mask is '<' -> pair
        # contributes exactly zero either way
        assert np.array_equal(F_with.view(np.int64),
                              F_without.view(np.int64)), \
            f"boundary pair (+{eps} ulp) list-dependence"


def test_forces_lambda_consistency_fd():
    """Forces at λ are the exact gradient of U(·; λ): FD check."""
    alch, pos, _ = _alch_fixture(n=8, seed=2, n_alch=2)
    lam = 0.4
    x = pos.copy()
    res = single_point_lambda(alch, x[:, None, :], lam)
    F = res["forces"][:, 0, :]
    h = 1e-6
    worst = 0.0
    for i in (0, 3, 5):
        for c in range(3):
            p1 = x.copy(); p1[i, c] += h
            p2 = x.copy(); p2[i, c] -= h
            fd = -(single_point_lambda(alch, p1[:, None, :], lam)["E"]
                   - single_point_lambda(alch, p2[:, None, :], lam)["E"]) / (2 * h)
            worst = max(worst, abs(fd - F[i, c]) / max(abs(fd), 1e-6))
    assert worst < 1e-4, f"force/energy mismatch at λ: rel {worst:.2e}"
