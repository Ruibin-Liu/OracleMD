"""A2 witness (M1): opus λ layer vs openmmtools AbsoluteAlchemicalFactory.

Configuration pinned per spec §4.6 (direction spec -> A2):
    softcore_alpha=0.5, softcore_a=1, softcore_b=1, softcore_c=6
    annihilate_electrostatics=True  (linear charge annihilation, q(l)=l·q)
    annihilate_sterics=False        (softcore decoupling)
    alchemical_pme_treatment='exact'
    dispersion correction OFF on both sides (its alchemical treatment is a
    separate A1-level concern, already covered there)

Fixed conformations, zero dynamics, full u_kn row over a λ grid.
Scope of the first witness: alchemical atoms carry no exceptions among
themselves (exception-convention matching is part of the §4.6 follow-up).
"""
import numpy as np
import openmm
import pytest
from openmm import unit

import tests._mpiplus_stub as _mpistub

mpistub = _mpistub  # keep linter happy

from .a1_harness import assert_close
from .test_a1_full import _dense_system
from opus.alchem import AlchemicalSystem, single_point_lambda
from opus.ingest import ingest_system_xml

LAMS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


def _a2_system(n=12, seed=6):
    sys_, pos = _dense_system(n=n, seed=seed)
    nb = [f for f in sys_.getForces()
          if isinstance(f, openmm.NonbondedForce)][0]
    nb.setUseDispersionCorrection(False)
    # alchemical atoms {5, 8, 11}: no exceptions among them; arrange charges
    # so the *alchemical set* is net-zero and the *environment* is net-zero
    # -> the system is neutral at every λ (G5; minefield #3)
    for i, qn in ((5, 0.30), (8, -0.30), (11, 0.0)):
        q, sg, e = nb.getParticleParameters(i)
        nb.setParticleParameters(i, qn * unit.elementary_charge, sg, e)
    qnet_env = 0.0
    for i in range(n):
        if i in (5, 8, 11):
            continue
        q, sg, e = nb.getParticleParameters(i)
        qnet_env += q.value_in_unit(unit.elementary_charge)
    # spread the env net charge over non-alchemical atoms
    n_env = n - 3
    for i in range(n):
        if i in (5, 8, 11):
            continue
        q, sg, e = nb.getParticleParameters(i)
        qn = q.value_in_unit(unit.elementary_charge) - qnet_env / n_env
        nb.setParticleParameters(i, qn * unit.elementary_charge, sg, e)
    return sys_, pos


def _openmmtools_ukn(sys_, pos, alch_atoms, lams=LAMS):
    _mpistub.install()
    from openmmtools.alchemy import (AbsoluteAlchemicalFactory,
                                     AlchemicalRegion)
    region = AlchemicalRegion(
        alchemical_atoms=alch_atoms,
        annihilate_electrostatics=True,
        annihilate_sterics=False,
        softcore_alpha=0.5, softcore_a=1, softcore_b=1, softcore_c=6)
    factory = AbsoluteAlchemicalFactory(
        consistent_exceptions=False,
        alchemical_pme_treatment="exact")
    alch_sys = factory.create_alchemical_system(sys_, region)

    from openmmtools.states import ThermodynamicState
    integrator = openmm.VerletIntegrator(1e-3)
    platform = openmm.Platform.getPlatformByName("Reference")
    ctx = openmm.Context(alch_sys, integrator, platform)
    ctx.setPositions(pos * unit.nanometer)
    names = [ctx.getParameterName(p) for p in range(ctx.getNumParameters())] \
        if hasattr(ctx, "getNumParameters") else []
    # Context parameter introspection: use the System's global parameters
    names = []
    for f in range(alch_sys.getNumForces()):
        force = alch_sys.getForce(f)
        for gp in range(getattr(force, "getNumGlobalParameters", lambda: 0)()):
            names.append(force.getGlobalParameterName(gp))
    u = []
    for lam in lams:
        for name in names:
            if "lambda_electrostatics" in name:
                ctx.setParameter(name, lam)
            elif "lambda_sterics" in name:
                ctx.setParameter(name, lam)
        E = ctx.getState(getEnergy=True).getPotentialEnergy()\
            .value_in_unit(unit.kilojoule_per_mole)
        u.append(E)
    del ctx
    return np.array(u)


def test_a2_ukn_witness():
    sys_, pos = _a2_system()
    alch_atoms = [5, 8, 11]        # no exceptions among these (see above)
    u_omm = _openmmtools_ukn(sys_, pos, alch_atoms)

    ir = ingest_system_xml(openmm.XmlSerializer.serialize(sys_))
    alch = AlchemicalSystem(ir, alch_atoms)
    u_ops = np.array([single_point_lambda(alch, pos[:, None, :], L)["E"]
                      for L in LAMS])

    for lam, a, b in zip(LAMS, u_ops, u_omm):
        assert_close(f"A2 u(λ={lam})", a, b, rel=1e-9, abs_=1e-8)
