#!/usr/bin/env python3
"""E0c: 单副本 N 标度(5k/20k/60k 原子),OpenMM CUDA,double 主路径(开放项 1 已决:
fp64 单路径)+ mixed 参照。Q-004b 的 N 依赖锚点;不依赖开放项 7。"""
import sys, time, subprocess, io
import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if util > 20:
    sys.exit(f"GPU busy ({util}%), timing invalid. Aborting.")

import openmm
from openmm import app, unit
print("openmm", openmm.__version__)

SEED_PDB = """\
HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
TER
END
"""
ff = app.ForceField("amber14/tip3pfb.xml")


def bench(L, precision, steps=2000, warmup=500):
    pdb = app.PDBFile(io.StringIO(SEED_PDB))
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addSolvent(ff, model="tip3p", boxSize=openmm.Vec3(L, L, L) * unit.nanometers)
    n = mod.topology.getNumAtoms()
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometers,
                             constraints=app.HBonds, rigidWater=True,
                             ewaldErrorTolerance=5e-4)
    platform = openmm.Platform.getPlatformByName("CUDA")
    props = {"Precision": precision, "DeterministicForces": "true"}
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                            2.0 * unit.femtoseconds)
    ctx = openmm.Context(system, integ, platform, props)
    ctx.setPositions(mod.positions)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    ctx.getState(getEnergy=True)
    integ.step(warmup)
    t0 = time.time()
    integ.step(steps)
    dt = time.time() - t0
    nsday = steps * 2e-6 / dt * 86400
    ms = dt / steps * 1e3
    del ctx, integ, system, mod
    return n, nsday, ms


print(f"{'N':>7} {'prec':>7} {'ns/day':>9} {'ms/step':>8}")
for target in (5000, 20000, 60000):
    L = (target / 100.0) ** (1 / 3.0)
    for prec in ("double", "mixed"):
        n, nsday, ms = bench(L, prec)
        print(f"{n:7d} {prec:>7} {nsday:9.1f} {ms:8.3f}", flush=True)
print("E0c done. 用途: Q-004b N 依赖;M2 单副本饱和度归因。")
