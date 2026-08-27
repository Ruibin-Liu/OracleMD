"""IR round-trip: IR -> OpenMM System -> per-Force comparison (spec §12).

The ingest translation layer has no oracle of its own unless we can walk
the path backwards: rebuild an OpenMM System from the IR and compare
per-Force single-point energies with the original.  This pins the
translation (not just the physics).
"""
from __future__ import annotations

import openmm
import numpy as np
from openmm import unit


def ir_to_openmm(ir, positions: np.ndarray) -> openmm.System:
    """Rebuild an OpenMM System from the IR (supported subset)."""
    s = openmm.System()
    for m in ir.masses:
        s.addParticle(float(m))
    if ir.box is not None:
        s.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in ir.box])

    if ir.bonds:
        f = openmm.HarmonicBondForce()
        for t in ir.bonds:
            f.addBond(t.atoms[0], t.atoms[1],
                      t.length.value, t.k.value)
        s.addForce(f)
    if ir.angles:
        f = openmm.HarmonicAngleForce()
        for t in ir.angles:
            f.addAngle(*t.atoms, t.theta0.value, t.k.value)
        s.addForce(f)
    if ir.torsions:
        f = openmm.PeriodicTorsionForce()
        for t in ir.torsions:
            f.addTorsion(*t.atoms, t.periodicity, t.phase.value, t.k.value)
        s.addForce(f)

    nb = ir.nonbonded
    if nb is not None:
        f = openmm.NonbondedForce()
        f.setNonbondedMethod(openmm.NonbondedForce.PME if nb.periodic
                             else openmm.NonbondedForce.NoCutoff)
        f.setCutoffDistance(nb.cutoff * unit.nanometer)
        for a in nb.atoms:
            f.addParticle(a.q.value, a.sigma.value, a.epsilon.value)
        for e in nb.exceptions:
            f.addException(e.a, e.b, e.chargeProd.value, e.sigma.value,
                           e.epsilon.value)
        f.setPMEParameters(nb.ewald_alpha / unit.nanometer, *nb.grid)
        f.setUseDispersionCorrection(nb.use_dispersion_correction)
        f.setExceptionsUsePeriodicBoundaryConditions(nb.exceptions_use_periodic)
        s.addForce(f)
    return s
