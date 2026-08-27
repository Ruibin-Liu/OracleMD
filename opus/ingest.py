"""Ingest: OpenMM System XML -> IR, with M0 gates (spec §12).

Verified XML format (OpenMM 8.6 serializer):
  <System particles="N"><PeriodicBoxVectors><A .../><B .../><C .../></...>
    <Particles><Particle mass="..."/>...
    <Forces>
      <Force type="HarmonicBondForce" usesPeriodic="0">
        <Bonds><Bond p1 p2 d k/>...
      <Force type="HarmonicAngleForce" ...><Angles><Angle p1 p2 p3 a k/>...
      <Force type="PeriodicTorsionForce" ...><Torsions><Torsion p1..p4 periodicity phase k/>...
      <Force type="NonbondedForce" cutoff alpha nx ny nz dispersionCorrection
             ewaldTolerance method="4" ...>
        <Particles><Particle q sig eps/>...
        <Exceptions><Exception p1 p2 q sig eps/>...
      <Force type="CMMotionRemover" .../>

Gates:
  G1  virtual sites rejected (spec §1.2);
  G2  unsupported force types rejected (open item 2);
  G3  max-force / parameter envelope (Q-008);
  G4  PME parameters must be pinned (alpha>0, nx/ny/nz>0): unpinned PME
      (alpha=0, grid=0) means "platform decides" — violates the oracle
      config general rule (§11.0).  The A1 harness pins via
      NonbondedForce.setPMEParameters().
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .ir import (AngleTerm, AtomParams, BondTerm, ConstraintTerm,
                 ExceptionTerm, IRSystem, NonbondedParams, ParamExpr,
                 TorsionTerm)

UNSUPPORTED = {
    "CustomNonbondedForce", "CustomBondForce", "CustomAngleForce",
    "CustomTorsionForce", "CustomCompoundForce", "CustomExternalForce",
    "CustomCVForce", "CustomGBForce", "CustomManyParticleForce",
    "CustomIntegrator", "GayBerneForce", "GBSAOBCForce",
}

# Q-008-derived gates (docs/m0/q008_worst_force.md)
MAX_ABS_Q = 1.5          # e
MAX_SIGMA = 0.5          # nm
MAX_EPSILON = 5.0        # kJ/mol
MAX_INITIAL_FORCE = 2.6e4  # kJ/mol/nm (Q-008 bound 5.2e4 / safety 2)


class IngestError(Exception):
    pass


def ingest_system_xml(source: str) -> IRSystem:
    try:
        root = ET.fromstring(source.strip().lstrip("\ufeff"))
    except ET.ParseError:
        root = ET.parse(source).getroot()
    if root.tag != "System":
        raise IngestError("root element is not <System>")

    n_atoms = len(root.findall("./Particles/Particle"))
    if "particles" in root.attrib:
        assert int(root.attrib["particles"]) == n_atoms

    box = None
    pbv = root.find("PeriodicBoxVectors")
    if pbv is not None:
        box = np.array([[float(pbv.find(t).attrib[c]) for c in "xyz"]
                        for t in ("A", "B", "C")])

    masses = np.zeros(n_atoms)
    for i, p in enumerate(root.findall("./Particles/Particle")):
        masses[i] = float(p.attrib["mass"])
    if np.any(masses <= 0):
        raise IngestError("G1: zero/negative mass or virtual site particle")

    bonds, angles, torsions = [], [], []
    constraints = []
    nb = None
    has_cm = False

    for el in root.findall("./Forces/Force"):
        t = el.attrib.get("type", el.tag)
        if t == "HarmonicBondForce":
            for b in el.findall("./Bonds/Bond"):
                bonds.append(BondTerm((int(b.attrib["p1"]), int(b.attrib["p2"])),
                                      ParamExpr(float(b.attrib["d"])),
                                      ParamExpr(float(b.attrib["k"]))))
        elif t == "HarmonicAngleForce":
            for a in el.findall("./Angles/Angle"):
                angles.append(AngleTerm(
                    (int(a.attrib["p1"]), int(a.attrib["p2"]), int(a.attrib["p3"])),
                    ParamExpr(float(a.attrib["a"])), ParamExpr(float(a.attrib["k"]))))
        elif t == "PeriodicTorsionForce":
            for x in el.findall("./Torsions/Torsion"):
                torsions.append(TorsionTerm(
                    tuple(int(x.attrib[f"p{i}"]) for i in (1, 2, 3, 4)),
                    int(x.attrib["periodicity"]),
                    ParamExpr(float(x.attrib["phase"])),
                    ParamExpr(float(x.attrib["k"]))))
        elif t == "NonbondedForce":
            nb = _ingest_nonbonded(el, n_atoms)
        elif t == "CMMotionRemover":
            has_cm = True
        elif t in UNSUPPORTED:
            raise IngestError(f"G2: unsupported force type <{t}> (open item 2)")
        else:
            raise IngestError(f"G2: unknown force type <{t}>")

    # <Constraints> is a top-level element, sibling of <Forces>
    for c in root.findall("./Constraints/Constraint"):
        constraints.append(ConstraintTerm(int(c.attrib["p1"]),
                                          int(c.attrib["p2"]),
                                          float(c.attrib["d"])))

    return IRSystem(n_atoms, bonds, angles, torsions, nb, has_cm, box, masses,
                    constraints)


def _ingest_nonbonded(el, n_atoms) -> NonbondedParams:
    a = el.attrib
    method = int(a.get("method", "2"))
    cutoff = float(a.get("cutoff", "1.0"))
    alpha = float(a.get("alpha", "0"))
    grid = (int(a.get("nx", "0")), int(a.get("ny", "0")), int(a.get("nz", "0")))
    periodic = method in (1, 2, 4)  # Ewald=1, PMC? cutoffPeriodic=2, PME=4
    use_pme = method == 4

    if use_pme and (alpha <= 0 or any(g == 0 for g in grid)):
        raise IngestError(
            "G4: PME parameters unpinned (alpha/nx/ny/nz == 0 -> platform-decided). "
            "Pin via NonbondedForce.setPMEParameters(); oracle config general rule §11.0")

    nb = NonbondedParams(
        cutoff=cutoff, ewald_alpha=alpha if use_pme else 0.0,
        grid=grid if use_pme else None,
        coulomb14scale=1.0,  # exceptions already carry final scaled parameters
        use_dispersion_correction=a.get("dispersionCorrection", "1") == "1",
        periodic=periodic,
        exceptions_use_periodic=a.get("exceptionsUsePeriodic", "0") == "1")

    arr = [None] * n_atoms
    for i, p in enumerate(el.findall("./Particles/Particle")):
        idx = int(p.attrib.get("i", i))
        arr[idx] = AtomParams(
            ParamExpr(float(p.attrib["q"])),
            ParamExpr(float(p.attrib["sig"])),
            ParamExpr(float(p.attrib["eps"])))
    if any(x is None for x in arr):
        raise IngestError("NonbondedForce particle table incomplete")
    nb.atoms = arr

    for e in el.findall("./Exceptions/Exception"):
        nb.exceptions.append(ExceptionTerm(
            int(e.attrib["p1"]), int(e.attrib["p2"]),
            ParamExpr(float(e.attrib["q"])),
            ParamExpr(float(e.attrib["sig"])),
            ParamExpr(float(e.attrib["eps"]))))
    return nb


def gate_parameter_envelope(nb: NonbondedParams) -> None:
    """G3a: parameter envelope from the Q-008 worst-force derivation."""
    q = np.array([x.q.value for x in nb.atoms])
    sig = np.array([x.sigma.value for x in nb.atoms])
    eps = np.array([x.epsilon.value for x in nb.atoms])
    if (np.abs(q).max() > MAX_ABS_Q or sig.max() > MAX_SIGMA
            or eps.max() > MAX_EPSILON):
        raise IngestError(
            f"G3(envelope): |q|max={np.abs(q).max():.2f} e, sigma_max={sig.max():.2f} nm, "
            f"eps_max={eps.max():.2f} kJ/mol exceeds Q-008 envelope "
            f"({MAX_ABS_Q} e / {MAX_SIGMA} nm / {MAX_EPSILON} kJ/mol)")


def gate_max_initial_force(forces: np.ndarray) -> None:
    """G3b: unminimized clash inputs overflow Q24.40 (spec §3.1)."""
    fmax = float(np.abs(forces).max())
    if fmax > MAX_INITIAL_FORCE:
        raise IngestError(
            f"G3(max-force): |F|max={fmax:.3e} kJ/mol/nm > gate {MAX_INITIAL_FORCE:.3e} "
            "(Q-008); minimize before ingest")
