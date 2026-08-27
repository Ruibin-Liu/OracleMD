"""A1 stage 1: bonded-only system (no nonbonded) — pins the functional forms."""
import numpy as np
import openmm
import pytest

from .a1_harness import (build_chain_system, compare_system,
                         openmm_single_point)
from openmm import unit


def _bonded_only(sys_and_pos):
    sys_, pos, meta = sys_and_pos
    s2 = openmm.System()
    for i in range(sys_.getNumParticles()):
        s2.addParticle(12.0 + i)
    s2.setDefaultPeriodicBoxVectors([2.2, 0, 0], [0, 2.2, 0], [0, 0, 2.2])
    b = openmm.HarmonicBondForce()
    for (i, j, r0, k) in meta["bonds"]:
        b.addBond(i, j, r0, k)
    s2.addForce(b)
    a = openmm.HarmonicAngleForce()
    for (i, j, l, th, k) in meta["angles"]:
        a.addAngle(i, j, l, th, k)
    s2.addForce(a)
    t = openmm.PeriodicTorsionForce()
    for (i, j, l, m, n_, ph, k) in meta["torsions"]:
        t.addTorsion(i, j, l, m, n_, ph, k)
    s2.addForce(t)
    return s2, pos


def test_bonded_chain():
    sys_, pos, meta = build_chain_system(6)
    s2, pos = _bonded_only((sys_, pos, meta))
    res, E_omm, F_omm = compare_system(s2, pos, "bonded-only")
    print("E decomposition:", res["energies"], "openmm:", E_omm)
