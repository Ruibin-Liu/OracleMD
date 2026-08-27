"""A1: full-system single-point comparison vs OpenMM Reference platform.

Covers: bonds/angles/torsions, PME (pinned alpha/grid), direct space with
pairs inside cutoff, exceptions (1-2/1-3/1-4 semantics, exceptionsUsePeriodic
both values), dispersion tail.  Gates: abs+rel dual (rel 1e-10, abs 1e-8).
"""
import numpy as np
import openmm
import pytest
import openmm.app as app
from openmm import unit

from .a1_harness import (assert_close, build_chain_system, compare_system,
                         openmm_single_point)


def _dense_system(n=12, seed=3, box=2.0, min_dist=0.22):
    """Random dense placement (regular pairs inside cutoff), rejection-sampled
    to keep |F| inside the Q24.40 envelope."""
    rng = np.random.default_rng(seed)
    for _ in range(500):
        pos = rng.uniform(0.25, box - 0.25, (n, 3))
        ok = True
        for i in range(n - 1):
            d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
            if d.min() < min_dist:
                ok = False
                break
        if ok:
            break
    assert ok

    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("SYS", chain)
    from openmm.app.element import Element
    for i in range(n):
        top.addAtom(f"A{i}", Element.getByAtomicNumber(6), res)
    sys_ = openmm.System()
    for i in range(n):
        sys_.addParticle(12.0)
    sys_.setDefaultPeriodicBoxVectors([box, 0, 0], [0, box, 0], [0, 0, box])

    q = rng.uniform(-0.5, 0.5, n)
    q -= q.mean()
    sig = rng.uniform(0.25, 0.32, n)
    eps = rng.uniform(0.3, 0.7, n)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    for i in range(n):
        nb.addParticle(q[i], sig[i], eps[i])
    # a few exclusions/exceptions among close pairs
    for (i, j, s) in [(0, 1, 0.0), (1, 2, 0.0), (2, 3, 0.0),
                      (0, 2, 0.5), (1, 3, 0.5), (0, 3, 0.25)]:
        nb.addException(i, j, q[i] * q[j] * s,
                        0.5 * (sig[i] + sig[j]),
                        float(np.sqrt(eps[i] * eps[j])) * s)
    nb.setPMEParameters(0.32 / unit.nanometer, 20, 20, 20)
    nb.setUseDispersionCorrection(True)
    sys_.addForce(nb)
    return sys_, pos


@pytest.mark.parametrize("seed,n", [(0, 6), (1, 7), (2, 8), (4, 10)])
def test_chain_systems(seed, n):
    sys_, pos, meta = build_chain_system(n, seed=seed, box=2.2)
    compare_system(sys_, pos, f"chain-{n}-{seed}")


def test_dense_system():
    sys_, pos = _dense_system()
    compare_system(sys_, pos, "dense-12")


def test_chain_dense_box():
    """Non-cubic (triclinic) box."""
    sys_, pos, meta = build_chain_system(7, seed=5)
    box = np.array([[2.4, 0.0, 0.0], [0.3, 2.5, 0.0], [-0.2, 0.25, 2.3]])
    for f in sys_.getForces():
        if isinstance(f, openmm.NonbondedForce):
            f.setCutoffDistance(0.8 * unit.nanometer)
    sys_.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in box])
    pos = pos - pos.min(0) + 0.3
    compare_system(sys_, pos, "triclinic-7")


def test_water_box(tmp_path):
    """Realistic 543-atom TIP3P-FB water box (rigidWater=False so the System
    carries explicit bonds/angles; positions from OpenMM's packer)."""
    import io
    pdb = openmm.app.PDBFile(io.StringIO(
        "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    2  H1  HOH A   1       0.096   0.000   0.000  1.00  0.00           H\n"
        "HETATM    3  H2  HOH A   1      -0.024   0.093   0.000  1.00  0.00           H\n"
        "END\n"))
    ff = openmm.app.ForceField("amber14/tip3pfb.xml")
    m = openmm.app.Modeller(pdb.topology, pdb.positions)
    m.addSolvent(ff, model="tip3p", boxSize=openmm.Vec3(1.8, 1.8, 1.8) * unit.nanometers)
    sys_ = ff.createSystem(m.topology, nonbondedMethod=openmm.app.PME,
                           nonbondedCutoff=0.9 * unit.nanometers,
                           constraints=None, rigidWater=False)
    nb = [f for f in sys_.getForces()
          if isinstance(f, openmm.NonbondedForce)][0]
    nb.setPMEParameters(0.32 / unit.nanometer, 20, 20, 20)
    pos = np.array(m.positions.value_in_unit(unit.nanometer))
    compare_system(sys_, pos, "water-543", rel=1e-9)
