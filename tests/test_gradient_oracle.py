"""Gradient oracle: analytic bonded forces vs mpmath 40-digit FD.

The independent oracle for the hand-derived analytic gradients — catches
algebra slips at 1e-25 before they can hide inside A1's 1e-10 comparison.
"""
import numpy as np
import pytest
from mpmath import mp, mpf

mp.dps = 40

from opus.ir import AngleTerm, BondTerm, ParamExpr, TorsionTerm
from opus.engine import ForceAccumulator


def _terms_energy(kind, terms, x):
    from opus.bonded import _KIND
    fa = ForceAccumulator(x.shape[0], 1)
    E = _KIND[kind](terms, x, fa)
    return E, fa


def _mp_energy_bond(terms, x, i, c, h):
    xx = [row[:] for row in x]
    xx[i][c] = xx[i][c] + h
    E = mpf(0)
    for t in terms:
        (a, b), r0, k = t.atoms, t.length.value, t.k.value
        d2 = sum((xx[a][d] - xx[b][d]) ** 2 for d in range(3))
        E += mpf(k) / 2 * (mp.sqrt(d2) - mpf(r0)) ** 2
    return E


def _mp_energy_angle(terms, x, i, c, h):
    xx = [row[:] for row in x]
    xx[i][c] = xx[i][c] + h
    E = mpf(0)
    for t in terms:
        (a, b, cc), th0, k = t.atoms, t.theta0.value, t.k.value
        v1 = [xx[a][d] - xx[b][d] for d in range(3)]
        v2 = [xx[cc][d] - xx[b][d] for d in range(3)]
        s2 = (v1[1] * v2[2] - v1[2] * v2[1]) ** 2 + \
             (v1[2] * v2[0] - v1[0] * v2[2]) ** 2 + \
             (v1[0] * v2[1] - v1[1] * v2[0]) ** 2
        cdt = sum(v1[d] * v2[d] for d in range(3))
        E += mpf(k) / 2 * (mp.atan2(mp.sqrt(s2), cdt) - mpf(th0)) ** 2
    return E


def _mp_energy_torsion(terms, x, i, c, h):
    xx = [row[:] for row in x]
    xx[i][c] = xx[i][c] + h
    E = mpf(0)
    for t in terms:
        (i0, i1, i2, i3), n, ph, k = t.atoms, t.periodicity, t.phase.value, t.k.value
        Fv = [xx[i1][d] - xx[i0][d] for d in range(3)]
        Gv = [xx[i2][d] - xx[i1][d] for d in range(3)]
        Hv = [xx[i3][d] - xx[i2][d] for d in range(3)]

        def cross(u, v):
            return [u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0]]
        A = cross(Fv, Gv)
        B = cross(Gv, Hv)
        W = cross(Gv, A)
        s = sum(W[d] * B[d] for d in range(3))
        g = mp.sqrt(sum(Gv[d] ** 2 for d in range(3)))
        c_ = sum(A[d] * B[d] for d in range(3)) * g
        phi = mp.atan2(s, c_)
        E += mpf(k) * (1 + mp.cos(mpf(n) * phi - mpf(ph)))
    return E


_MP = {"bond": _mp_energy_bond, "angle": _mp_energy_angle,
       "torsion": _mp_energy_torsion}


@pytest.mark.parametrize("kind,nterms,nat", [
    ("bond", 3, 4), ("angle", 3, 5), ("torsion", 3, 6)])
def test_analytic_vs_mpmath(kind, nterms, nat):
    rng = np.random.default_rng(42 + len(kind))
    pos = rng.standard_normal((nat, 3)) * 0.3
    if kind == "bond":
        terms = [BondTerm((i, i + 1), ParamExpr(0.1 + 0.1 * rng.random()),
                          ParamExpr(50000 * rng.random() + 1000))
                 for i in range(nterms)]
    elif kind == "angle":
        terms = [AngleTerm((i, i + 1, i + 2), ParamExpr(1.0 + rng.random()),
                           ParamExpr(100 + 400 * rng.random()))
                 for i in range(nterms)]
    else:
        terms = [TorsionTerm((i, i + 1, i + 2, i + 3), 2,
                             ParamExpr(0.5 * rng.random()),
                             ParamExpr(5 * rng.random()))
                 for i in range(nterms)]

    x = pos[:, None, :].copy()          # (N, R=1, 3)
    _, fa = _terms_energy(kind, terms, x)
    F = fa.forces()[:, 0, :]
    F = F[[a for t in terms for a in t.atoms] and slice(None)]  # keep all

    h = mpf(10) ** (-25)
    Fmp = np.zeros_like(F)
    mpfied = [[mpf(repr(v)) for v in row] for row in pos.tolist()]
    for i in range(nat):
        for c in range(3):
            ep = _MP[kind](terms, mpfied, i, c, h)
            em = _MP[kind](terms, mpfied, i, c, -h)
            Fmp[i, c] = float(-(ep - em) / (2 * h))
    # only atoms appearing in terms are meaningful
    used = sorted({a for t in terms for a in t.atoms})
    d = np.abs(F[used] - Fmp[used]).max() / max(np.abs(Fmp[used]).max(), 1e-30)
    assert d < 1e-10, f"{kind}: analytic vs mpmath rel diff {d:.3e}"
