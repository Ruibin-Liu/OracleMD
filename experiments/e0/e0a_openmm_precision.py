#!/usr/bin/env python3
"""E0a: OpenMM mixed vs double, ~60k 原子显式水体系, A100. 自包含(内嵌种子 PDB)。"""
import sys, time, subprocess, io
import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if util > 20:
    sys.exit(f"GPU busy ({util}%), timing invalid. Aborting.")

import openmm
from openmm import app, unit
print("openmm", openmm.__version__)
plats = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print(" platforms:", plats)
if "CUDA" not in plats:
    sys.exit("CUDA platform unavailable")

SEED_PDB = """\
HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
TER
END
"""

ff = app.ForceField("amber14/tip3pfb.xml")
pdb = app.PDBFile(io.StringIO(SEED_PDB))
mod = app.Modeller(pdb.topology, pdb.positions)
L = 8.5  # nm -> ~8.5^3 * 33.4 * 3 ≈ 61.5k atoms
mod.addSolvent(ff, model="tip3p",
               boxSize=openmm.Vec3(L, L, L)*unit.nanometers)
n = mod.topology.getNumAtoms()
print(f"atoms: {n}")

system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                         nonbondedCutoff=1.0*unit.nanometers,
                         constraints=app.HBonds, rigidWater=True,
                         ewaldErrorTolerance=5e-4)

def bench(precision, steps=2000, warmup=500):
    platform = openmm.Platform.getPlatformByName("CUDA")
    props = {"Precision": precision, "DeterministicForces": "true"}
    integ = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond,
                                            2.0*unit.femtoseconds)
    ctx = openmm.Context(system, integ, platform, props)
    ctx.setPositions(mod.positions)
    ctx.setVelocitiesToTemperature(300*unit.kelvin, 1234)
    ctx.getState(getEnergy=True)
    integ.step(warmup)
    t0 = time.time()
    integ.step(steps)
    e = ctx.getState(getEnergy=True).getPotentialEnergy()
    dt = time.time() - t0
    nsday = steps * 2e-6 / dt * 86400
    del ctx, integ
    return nsday, e

results = {}
for prec in ("mixed", "double", "mixed", "double"):
    r, e = bench(prec)
    results.setdefault(prec, []).append(r)
    print(f"{prec}: {r:.1f} ns/day  (E={e})")

m = np.median(results["mixed"]); d = np.median(results["double"])
print(f"\nmedian mixed={m:.1f} double={d:.1f} ns/day, mixed/double = {m/d:.2f}x")
print(f"fp64 损失 = {m/d:.2f}x (E0a 判据: >3x 则回退双路径)")
print(f"[Q-004b] eta_serial 锚点: double 路径 {d:.1f} ns/day @ 2fs")
