"""A1 harness: build hand-controlled systems, single-point compare vs OpenMM.

Oracle config general rule (spec §11.0): the OpenMM reference system is
built with pinned PME parameters (setPMEParameters) so both sides share
alpha/grid/order; comparison gates are abs+rel dual (B11).
"""
from __future__ import annotations

import numpy as np
import openmm
from openmm import app, unit

from opus.engine import single_point
from opus.ingest import ingest_system_xml


def dual_gate(a: float, b: float, rel=1e-10, abs_=1e-8) -> bool:
    return abs(a - b) < abs_ or abs(a - b) / max(abs(a), abs(b), 1e-300) < rel


def assert_close(label, a, b, rel=1e-10, abs_=1e-8):
    ok = dual_gate(a, b, rel, abs_)
    if not ok:
        rel_err = abs(a - b) / max(abs(a), abs(b), 1e-300)
        raise AssertionError(f"{label}: opus={a!r} openmm={b!r} rel={rel_err:.3e}")
    return True


def build_chain_system(n_atoms=6, seed=0, box=2.2, alpha=0.28,
                       grid=(24, 24, 24), dispersion=True):
    """Hand-built linear chain with bonds/angles/torsions, random-ish params."""
    rng = np.random.default_rng(seed)
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("SYS", chain)
    from openmm.app.element import Element
    elems = [Element.getByAtomicNumber(6 + (i % 2)) for i in range(n_atoms)]
    atoms = [top.addAtom(f"A{i}", elems[i], res) for i in range(n_atoms)]

    # positions: random walk, non-clashing
    pos = np.zeros((n_atoms, 3))
    for i in range(1, n_atoms):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        pos[i] = pos[i - 1] + 0.15 + 0.05 * rng.random() * d
    pos = (pos - pos.mean(0)) % box + 0.3

    sys_ = openmm.System()
    for i in range(n_atoms):
        sys_.addParticle(12.0 + i)

    bonds = openmm.HarmonicBondForce()
    bond_terms = []
    for i in range(n_atoms - 1):
        r0 = float(np.linalg.norm(pos[i + 1] - pos[i]))
        k = 80000.0 + 20000.0 * rng.random()
        bonds.addBond(i, i + 1, r0 + 0.02 * rng.random(), k)
        bond_terms.append((i, i + 1, r0, k))
    sys_.addForce(bonds)

    angles = openmm.HarmonicAngleForce()
    angle_terms = []
    for i in range(n_atoms - 2):
        a0 = 1.6 + 0.4 * rng.random()
        k = 300.0 + 200.0 * rng.random()
        angles.addAngle(i, i + 1, i + 2, a0, k)
        angle_terms.append((i, i + 1, i + 2, a0, k))
    sys_.addForce(angles)

    torsions = openmm.PeriodicTorsionForce()
    torsion_terms = []
    for i in range(n_atoms - 3):
        k = 2.0 + 3.0 * rng.random()
        phase = rng.random() * 0.5
        torsions.addTorsion(i, i + 1, i + 2, i + 3, 2, phase, k)
        torsion_terms.append((i, i + 1, i + 2, i + 3, 2, phase, k))
    sys_.addForce(torsions)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    q = rng.uniform(-0.6, 0.6, n_atoms)
    q -= q.mean()  # neutral
    sig = rng.uniform(0.25, 0.35, n_atoms)
    eps = rng.uniform(0.2, 0.8, n_atoms)
    for i in range(n_atoms):
        nb.addParticle(q[i], sig[i], eps[i])
    # exclusions: bonded pairs; exceptions: 1-3 and 1-4 with scaled params
    excl_pairs, exc_terms = [], []
    for i in range(n_atoms - 1):
        nb.addException(i, i + 1, 0.0, sig[i] * 0 + 0.3, 0.0)
        excl_pairs.append((i, i + 1))
    for i in range(n_atoms - 2):
        qq = q[i] * q[i + 2] * 0.5
        nb.addException(i, i + 2, qq, 0.5 * (sig[i] + sig[i + 2]),
                        np.sqrt(eps[i] * eps[i + 2]) * 0.5)
        exc_terms.append((i, i + 2, qq, 0.5 * (sig[i] + sig[i + 2]),
                          np.sqrt(eps[i] * eps[i + 2]) * 0.5))
    if n_atoms >= 4:
        for i in range(n_atoms - 3):
            qq = q[i] * q[i + 3] * 0.25
            nb.addException(i, i + 3, qq, 0.5 * (sig[i] + sig[i + 3]),
                            np.sqrt(eps[i] * eps[i + 3]) * 0.25)
            exc_terms.append((i, i + 3, qq, 0.5 * (sig[i] + sig[i + 3]),
                              np.sqrt(eps[i] * eps[i + 3]) * 0.25))
    nb.setPMEParameters(alpha / unit.nanometer, *grid)
    nb.setUseDispersionCorrection(dispersion)
    sys_.addForce(nb)

    sys_.setDefaultPeriodicBoxVectors([box, 0, 0], [0, box, 0], [0, 0, box])
    return sys_, pos, dict(bonds=bond_terms, angles=angle_terms,
                           torsions=torsion_terms, q=q, sig=sig, eps=eps,
                           excl=excl_pairs, exc=exc_terms, box=box,
                           alpha=alpha, grid=grid, dispersion=dispersion)


def openmm_single_point(sys_, pos, groups=None):
    """Reference-platform single point; returns (energies_by_group, forces)."""
    ctx = openmm.Context(sys_, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos * unit.nanometer)
    st = ctx.getState(getEnergy=True, getForces=True, groups=groups or -1)
    E = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    F = st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    del ctx
    return E, F


def openmm_group_energies(sys_, pos, group_map):
    out = {}
    for name, mask in group_map.items():
        ctx = openmm.Context(sys_, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(pos * unit.nanometer)
        E = ctx.getState(getEnergy=True, groups=mask).getPotentialEnergy()\
            .value_in_unit(unit.kilojoule_per_mole)
        out[name] = E
        del ctx
    return out


def compare_system(sys_openmm, pos, label="system", rel=1e-10, abs_=1e-8):
    """Serialize -> ingest -> opus single point vs OpenMM (R=1)."""
    xml = openmm.XmlSerializer.serialize(sys_openmm)
    ir = ingest_system_xml(xml)
    res = single_point(ir, pos[:, None, :].copy())
    E_omm, F_omm = openmm_single_point(sys_openmm, pos)
    E_ops = sum(res["energies"].values())
    assert_close(f"{label}: total energy", E_ops, E_omm, rel, abs_)

    F_ops = res["forces"][:, 0, :]
    fa = np.abs(F_ops).max()
    da = np.abs(F_ops - F_omm)
    denom = np.maximum(np.abs(F_omm), np.abs(F_ops))
    with np.errstate(divide="ignore", invalid="ignore"):
        relf = np.where(denom > 0, da / denom, 0.0)
    if not (da.max() < abs_ or relf.max() < rel):
        i = np.unravel_index(np.argmax(da), da.shape)
        raise AssertionError(
            f"{label}: force mismatch atom {i[0]} dim {i[1]}: "
            f"opus={F_ops[i]:.12e} openmm={F_omm[i]:.12e} "
            f"abs={da.max():.3e} rel={relf.max():.3e} (|F|max={fa:.2f})")
    return res, E_omm, F_omm
