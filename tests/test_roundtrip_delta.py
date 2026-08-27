"""Round-trip (IR -> OpenMM -> per-Force) + delta debugging driver."""
import numpy as np
import openmm
import pytest
from openmm import unit

from opus.engine import single_point
from opus.export_ir import ir_to_openmm
from opus.ingest import ingest_system_xml

from .a1_harness import build_chain_system, openmm_single_point, assert_close


def test_ir_roundtrip_per_force():
    sys_, pos, meta = build_chain_system(6, seed=8)
    ir = ingest_system_xml(openmm.XmlSerializer.serialize(sys_))
    sys2 = ir_to_openmm(ir, pos)
    E1, F1 = openmm_single_point(sys_, pos)
    E2, F2 = openmm_single_point(sys2, pos)
    assert_close("round-trip total energy", E2, E1)
    d = np.abs(F1 - F2).max()
    assert d < 1e-6, f"round-trip forces differ by {d}"


# ------------------------------------------------------------ delta debug

def make_oracle(sys_openmm_original, positions):
    """The oracle is the ORIGINAL (uncorrupted) OpenMM system — comparing a
    corrupted IR against its own round-trip is vacuous."""
    E_ref, F_ref = openmm_single_point(sys_openmm_original, positions)
    return E_ref, F_ref


def delta_debug_compare(ir, pos, E_ref, F_ref, rel=1e-8):
    """Returns None if opus matches the reference, else a label."""
    res = single_point(ir, pos[:, None, :].copy())
    E2 = sum(res["energies"].values())
    if not (abs(E2 - E_ref) < 1e-6
            or abs(E2 - E_ref) / max(abs(E_ref), 1e-300) < rel):
        return f"energy {E2} vs {E_ref}"
    d = np.abs(res["forces"][:, 0, :] - F_ref[:ir.n_atoms])
    if d.max() > 1e-3 * max(np.abs(F_ref).max(), 1.0):
        return "forces"
    return None


def test_delta_debug_shrinks_a_synthetic_failure():
    """The debugger must reduce an injected failure to a 2-atom system.

    Injection: corrupt the LJ mixing for one atom pair class (the classic
    'epsilon mixing rule' bug), then check the shrinker finds the minimal
    failing subset.
    """
    from .test_a1_full import _dense_system
    sys_, pos = _dense_system(n=10, seed=2)
    ir0 = ingest_system_xml(openmm.XmlSerializer.serialize(sys_))
    E_ref, F_ref = make_oracle(sys_, pos)
    ir = ingest_system_xml(openmm.XmlSerializer.serialize(sys_))
    # inject: wrong epsilon on atom 3 (sqrt -> arithmetic mean style error)
    ir.nonbonded.atoms[3].epsilon.value *= 1.5

    def fails(sub_ir, sub_pos):
        return delta_debug_compare(sub_ir, sub_pos, E_ref, F_ref) is not None

    assert fails(ir, pos), "injection must fail"

    from opus.delta import shrink_system
    min_ir, min_pos, why = shrink_system(ir, pos, fails)
    # minimal failing system must be small (contains atom 3 + one partner)
    assert min_ir.n_atoms <= 2, f"shrinker stalled at {min_ir.n_atoms} atoms ({why})"
