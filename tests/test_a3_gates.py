"""RNG self-test (counter-based, pillar 3) + A3 analytic oracles + gates."""
import numpy as np
import openmm
import pytest
from openmm import unit

from opus import rng
from opus.engine import single_point
from opus.ingest import (IngestError, ingest_system_xml, gate_max_initial_force)
from opus.pme import PmeGrid, naive_ewald_reciprocal
import opus.pme as pme_mod
from opus import fxp


# ---------------------------------------------------------------- RNG

def test_rng_same_seed_same_stream():
    a = rng.gauss_stream(42, 100, atom_id=3, slot=5, dof=0, n=100)
    b = rng.gauss_stream(42, 100, atom_id=3, slot=5, dof=0, n=100)
    assert np.array_equal(a, b)


def test_rng_different_slots_uncorrelated():
    a = rng.gauss_stream(42, 100, atom_id=3, slot=0, dof=0, n=10000)
    b = rng.gauss_stream(42, 100, atom_id=3, slot=1, dof=0, n=10000)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.05


def test_rng_stable_atom_id_semantics():
    """Streams follow stable atom id, not array index (M6b semantics)."""
    a = rng.gauss_stream(7, 0, atom_id=2, slot=0, dof=1, n=10)
    b = rng.gauss_stream(7, 0, atom_id=2, slot=0, dof=1, n=10)
    assert np.array_equal(a, b)
    c = rng.gauss_stream(7, 0, atom_id=3, slot=0, dof=1, n=10)
    assert not np.array_equal(a, c)


def test_host_rng_deterministic():
    a = rng.uniform_stream(1, 50, move_type=2, pair_id=3)
    b = rng.uniform_stream(1, 50, move_type=2, pair_id=3)
    assert a == b


# ---------------------------------------------------------------- A3

def test_a3_harmonic_bond_analytic():
    """Single harmonic bond: engine vs hand evaluation (A3 library form)."""
    from opus.ir import BondTerm, IRSystem, ParamExpr
    r0, k = 0.2, 250000.0
    pos = np.array([[0.0, 0.0, 0.0], [0.25, 0.03, 0.0]])
    ir = IRSystem(2, [BondTerm((0, 1), ParamExpr(r0), ParamExpr(k))],
                  [], [], None, False, None, np.array([12.0, 12.0]))
    res = single_point(ir, pos[:, None, :].copy())
    r = float(np.linalg.norm(pos[1] - pos[0]))
    e_manual = 0.5 * k * (r - r0) ** 2
    assert abs(res["energies"]["HarmonicBondForce"] - e_manual) < 1e-8


def test_a3_madelung_constant():
    """PME + naive Ewald on an NaCl-like lattice -> Madelung constant.

    E_total(periodic, neutral) / (N_pairs_scale) must reproduce
    M(NaCl) = 1.747564594633182... for alpha_L = KE * q^2 / r0.
    """
    # 2x2x2 NaCl rock-salt cell (periodic, neutral), lattice const a
    a = 0.28
    q = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=float)  # checkerboard
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                   dtype=float) * a
    box = np.diag([2 * a, 2 * a, 2 * a])
    alpha = 3.5 / a
    # Naive Ewald with the self term removed = pair energy;
    # Madelung: E_coulomb(NaCl, per ion pair) = -M * KE * q^2 / r_nn
    E_recip = naive_ewald_reciprocal(pos, q, box, alpha, kmax=40)
    E_self = -138.93545764438198 * alpha / np.sqrt(np.pi) * np.sum(q * q)
    from math import erfc
    E_dir = 0.0
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            r = np.linalg.norm((pos[j] - pos[i]) % (2 * a))
            # minimum image on the 2a cube: r or r-2a per axis —
            # use fractional MIC
            d = pos[j] - pos[i]
            d -= np.round(d / (2 * a)) * 2 * a
            r = float(np.linalg.norm(d))
            if r < 1e-9:
                continue
            E_dir += 138.93545764438198 * q[i] * q[j] * erfc(alpha * r) / r
    E_pair = E_recip + 0.5 * E_dir + E_self   # full self term (not halved)
    # 4 ion pairs in the cell; nearest-neighbor distance r0 = a
    E_per_pair = E_pair / 4.0
    M = -E_per_pair / (138.93545764438198 * 1.0 ** 2 / a)
    assert abs(M - 1.747564594633182) < 2e-3, f"Madelung {M}"


def test_a3_pme_matches_naive_ewald():
    """Oracle ladder rung: PME grid <-> naive continuum Ewald."""
    rng_ = np.random.default_rng(3)
    pos = rng_.uniform(0.2, 2.0, (8, 3))
    q = rng_.uniform(-0.5, 0.5, 8)
    q -= q.mean()
    box = np.diag([2.2, 2.2, 2.2])
    alpha = 1.1
    g = pme_mod.PmeGrid(box, (32, 32, 32), alpha)
    frac = pme_mod.frac_coords(pos, g.inv_box)
    E_pme, _ = pme_mod.reciprocal_energy(g, q, frac, fxp.q16_48_grid((32, 32, 32)))
    E_naive = naive_ewald_reciprocal(pos, q, box, alpha, kmax=30)
    assert abs(E_pme - E_naive) / max(abs(E_naive), 1e-12) < 1e-5


# ---------------------------------------------------------------- gates

def test_gate_unpinned_pme_rejected():
    from .a1_harness import build_chain_system
    sys_, pos, meta = build_chain_system(4, seed=0)
    xml = openmm.XmlSerializer.serialize(sys_)
    import re
    # strip the pinned PME parameters -> alpha=0, nx=0
    xml_bad = re.sub(r'nx="\d+" ny="\d+" nz="\d+"', 'nx="0" ny="0" nz="0"',
                     xml.replace(f'alpha="{meta["alpha"]}"', 'alpha="0"'))
    with pytest.raises(IngestError, match="G4"):
        ingest_system_xml(xml_bad)


def test_gate_unsupported_force_rejected():
    from .a1_harness import build_chain_system
    sys_, pos, meta = build_chain_system(4, seed=0)
    f = openmm.CustomBondForce("k*(r-r0)^2")
    f.addPerBondParameter("k")
    f.addBond(0, 1, [100.0])
    sys_.addForce(f)
    xml = openmm.XmlSerializer.serialize(sys_)
    with pytest.raises(IngestError, match="G2"):
        ingest_system_xml(xml)


def test_gate_max_initial_force():
    F = np.full((3, 3), 1e5)
    with pytest.raises(IngestError, match="G3"):
        gate_max_initial_force(F)
    gate_max_initial_force(np.zeros((3, 3)))  # fine


def test_gate_parameter_envelope():
    from .a1_harness import build_chain_system
    sys_, pos, meta = build_chain_system(4, seed=0)
    nb = [f for f in sys_.getForces()
          if isinstance(f, openmm.NonbondedForce)][0]
    nb.setParticleParameters(0, 2.5, 0.3, 0.5)  # |q| > 1.5 e
    xml = openmm.XmlSerializer.serialize(sys_)
    ir = ingest_system_xml(xml)
    from opus.ingest import gate_parameter_envelope
    with pytest.raises(IngestError, match="G3"):
        gate_parameter_envelope(ir.nonbonded)
