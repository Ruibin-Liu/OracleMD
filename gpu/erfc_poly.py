"""erfc polynomial: fit, validate, and emit CUDA source (production).

Form: erfc(x) = exp(-x^2) * W(x), W = 3-piece Chebyshev (piece-local t).
exp(-x^2) is shared with the force formula's second term (one exp per pair).

Pieces: [0,1.5] deg 18 / [1.5,3.0] deg 14 / [3.0,6.5] deg 18.
Measured max rel error vs scipy erfc: <= 3.2e-14 on [0, 6.5]
(E0f: 3.11e-14 device-side). Requirement: <= 1e-11 (A1 tol 1e-10 margin).

Domain note: x = alpha*r <= alpha*(rc+skin) = 3.5*1.35 = 4.725 in-mask
semantics; [3,6.5] covers the list radius. x > 6.5 clamps (abs err
<= erfc(6.5) = 5.8e-20, invisible in forces).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as _C
from numpy.polynomial import polynomial as _P

PIECES = ((0.0, 1.5, 18), (1.5, 3.0, 14), (3.0, 6.5, 18))


def _fit_piece(lo: float, hi: float, deg: int, npts: int = 4001):
    """Rel-error-weighted Chebyshev fit of W(x)=erfc(x)e^{x^2} on [lo,hi]."""
    x = np.linspace(lo, hi, npts)
    w = __import__("scipy.special", fromlist=["erfc"]).erfc(x) * np.exp(x * x)
    t = 2 * (x - lo) / (hi - lo) - 1
    V = _C.chebvander(t, deg)
    coef, *_ = np.linalg.lstsq(V / w[:, None], np.ones(npts), rcond=None)
    rel = np.abs(_C.chebval(t, coef) - w) / np.abs(w)
    return coef, float(rel.max())


import json as _json
import os as _os

_DATA = _json.load(open(_os.path.join(_os.path.dirname(__file__),
                                      "erfc_coeffs.json")))


def piece_power_coeffs(lo: float, hi: float, deg: int) -> np.ndarray:
    """Power coefficients in piece-local t (generated data file; regenerate
    with gen script -- scipy needed only for generation/validation, not
    runtime)."""
    key = "%g:%g:%d" % (lo, hi, deg)
    return np.array(_DATA["pieces"][key])


def max_rel_stored() -> float:
    return float(_DATA["max_rel"])


def horner_expr(coeffs: np.ndarray) -> str:
    """Nested Horner expression for the W device functions (ascending)."""
    e = f"{coeffs[-1]:+.17e}"
    for c in coeffs[-2::-1]:
        e = f"({e}*t{c:+.17e})"
    return e


def emit_cuda() -> str:
    """CUDA source of the W device functions + erfc_poly dispatcher."""
    parts = []
    for i, (lo, hi, deg) in enumerate(PIECES):
        pc = piece_power_coeffs(lo, hi, deg)
        parts.append(
            "__device__ __forceinline__ double WP%d(double t) {\n"
            "    return %s;\n}" % (i, horner_expr(pc)))
    k = 1.5 / (1.5 - 0.0) * 2.0  # 2/(hi-lo) per piece
    s0 = 2.0 / 1.5
    s1 = 2.0 / 1.5
    s2 = 2.0 / 3.5
    src = "\n\n".join(parts) + f"""

__device__ __forceinline__ double erfc_poly(double x, double emx2) {{
    // emx2 = exp(-x*x) precomputed (shared with force term)
    double w;
    if (x < 1.5) {{
        double t = x * {s0:.16e} - 1.0;
        w = WP0(t);
    }} else if (x < 3.0) {{
        double t = (x - 1.5) * {s1:.16e} - 1.0;
        w = WP1(t);
    }} else {{
        double xc = x > 6.5 ? 6.5 : x;
        double t = (xc - 3.0) * {s2:.16e} - 1.0;
        w = WP2(t);
    }}
    return emx2 * w;
}}"""
    return src


_PC_CACHE: dict[int, np.ndarray] = {}


def _piece_coeffs(i: int) -> np.ndarray:
    if i not in _PC_CACHE:
        lo, hi, deg = PIECES[i]
        _PC_CACHE[i] = piece_power_coeffs(lo, hi, deg)
    return _PC_CACHE[i]


def erfc_poly_host(x: float) -> float:
    """Host-side mirror of the kernel (validation oracle)."""
    emx2 = float(np.exp(-x * x))
    if x < 1.5:
        i, lo, hi = 0, 0.0, 1.5
    elif x < 3.0:
        i, lo, hi = 1, 1.5, 3.0
    else:
        i, lo, hi = 2, 3.0, 6.5
        x = min(x, 6.5)
    pc = _piece_coeffs(i)
    t = 2 * (x - lo) / (hi - lo) - 1
    r = 0.0
    for c in pc[::-1]:
        r = r * t + c
    return emx2 * r


def validate(max_rel_target: float = 1e-13) -> float:
    """Max rel error vs scipy erfc on [0.001, 6.49]; raises if above target."""
    from scipy.special import erfc as ref
    xs = np.linspace(0.001, 6.49, 20001)
    got = np.array([erfc_poly_host(float(x)) for x in xs])
    rel = np.abs(got - ref(xs)) / ref(xs)
    m = float(rel.max())
    if m > max_rel_target:
        raise AssertionError(f"erfc_poly rel {m:.2e} > {max_rel_target}")
    return m
