"""GPU production kernels (M2).

Semantics oracle: opus/ reference implementation (A1-aligned to OpenMM).
Coefficient discipline: all polynomial coefficients are generated
programmatically in gpu/erfc_poly.py -- transcription into kernels is
forbidden (7 transcription-class defects in the 2026-09 session, all
caught by cross-checks; the generators + CI tripwires are the institution).
"""
from . import erfc_poly  # noqa: F401
from . import constrain  # noqa: F401
from . import integrate  # noqa: F401
from . import pme  # noqa: F401
from . import rand  # noqa: F401
from . import force_q24  # noqa: F401
