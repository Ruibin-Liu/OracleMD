"""IR — spec §4. Named-axis tensors, ForceTerm, ParamExpr (Const at M0).

Axis discipline (minefield #014): every replica-batched tensor carries named
axes.  At M0 the only batched tensors are positions/forces: (atom, slot, xyz).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Axis:
    """Named axes for replica-batched tensors (compile-time tags)."""
    ATOM = "atom"
    SLOT = "slot"
    XYZ = "xyz"
    GRID = "grid"          # (gx, gy, gz, slot)
    STATE = "state"        # lambda-dependent quantities live on state, not slot
    LAMBDA = "lambda"      # lambda-component axis (physical)
    TIME = "time"          # u_kn column axis


VALID = {Axis.ATOM, Axis.SLOT, Axis.XYZ, Axis.GRID, Axis.STATE,
         Axis.LAMBDA, Axis.TIME}


def check_axes(name: str, axes: tuple[str, ...], shape: tuple[int, ...]) -> None:
    if len(axes) != len(shape):
        raise ValueError(f"{name}: axes {axes} vs shape {shape}")
    for a in axes:
        if a not in VALID:
            raise ValueError(f"{name}: unknown axis {a!r}")


class Kind(Enum):
    BOND = "HarmonicBond"
    ANGLE = "HarmonicAngle"
    TORSION = "PeriodicTorsion"
    NONBONDED = "Nonbonded"
    CM_MOTION_REMOVER = "CMMotionRemover"


@dataclass
class ParamExpr:
    """M0: Const only.  Interp/Scale/Custom arrive with the lambda layer (M1)."""
    value: float


@dataclass
class BondTerm:
    atoms: tuple[int, int]
    length: ParamExpr
    k: ParamExpr


@dataclass
class AngleTerm:
    atoms: tuple[int, int, int]
    theta0: ParamExpr
    k: ParamExpr


@dataclass
class TorsionTerm:
    atoms: tuple[int, int, int, int]
    periodicity: int
    phase: ParamExpr
    k: ParamExpr


@dataclass
class AtomParams:
    q: ParamExpr
    sigma: ParamExpr
    epsilon: ParamExpr


@dataclass
class ExceptionTerm:
    a: int
    b: int
    chargeProd: ParamExpr
    sigma: ParamExpr
    epsilon: ParamExpr


@dataclass
class ExclusionPair:
    a: int
    b: int


@dataclass
class NonbondedParams:
    cutoff: float
    ewald_alpha: float          # pinned; >0 means PME
    grid: tuple[int, int, int] | None
    spline_order: int = 4
    coulomb14scale: float = 1.0
    use_dispersion_correction: bool = True
    periodic: bool = True
    exceptions_use_periodic: bool = False
    atoms: list[AtomParams] = field(default_factory=list)
    exceptions: list[ExceptionTerm] = field(default_factory=list)
    exclusions: list[ExclusionPair] = field(default_factory=list)


@dataclass
class ConstraintTerm:
    a: int
    b: int
    distance: float


@dataclass
class IRSystem:
    """Whole-system IR.  Positions carry named axes (atom, slot, xyz)."""
    n_atoms: int
    bonds: list[BondTerm]
    angles: list[AngleTerm]
    torsions: list[TorsionTerm]
    nonbonded: NonbondedParams | None
    has_cm_remover: bool
    box: np.ndarray | None          # 3x3, rows are box vectors
    masses: np.ndarray
    constraints: list[ConstraintTerm] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []

    def check(self) -> None:
        if self.nonbonded is not None and self.nonbonded.periodic:
            assert self.box is not None and self.nonbonded.ewald_alpha > 0
