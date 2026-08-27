"""Nonbonded direct space + exceptions + dispersion tail (spec §5.2/§5.3).

Semantics (pinned against OpenMM 8.6 Reference platform):
  - Lorentz-Berthelot mixing: sigma_ij = (sigma_i+sigma_j)/2,
    epsilon_ij = sqrt(eps_i eps_j);
  - Coulomb constant KE = ONE_4PI_EPS0 = 138.93545764438198 kJ·nm/mol/e²;
  - direct space: KE q_i q_j erfc(alpha r)/r + 4 eps_ij[(s/r)^12-(s/r)^6],
    exceptions excluded from the standard sum;
  - exceptions: KE (q_exc - q_i q_j erf(alpha r))/r
    + 4 eps_exc[(s_exc/r)^12-(s_exc/r)^6]; distance uses minimum image only
    if exceptionsUsePeriodic (ingested flag);
  - dispersion tail: OpenMM NonbondedForceImpl::calcDispersionCorrection
    (verified against source): class pairs with self-pairs, N(N+1)/2
    normalization, includes the r^-12 integral term; energy only.

Pillar 5 (masking bitwise-strict): pairs beyond cutoff contribute exactly
zero (multiply-by-zero); the branch decision uses the same r2 as the force
path; boundary (r2 == rc2) is '<' (compile-time constant BOUNDARY).
"""
from __future__ import annotations

import numpy as np

BOUNDARY = "lt"  # r² < rc² belongs to "inside" — frozen at compile time

# Coulomb constant, OpenMM's ONE_4PI_EPS0 (kJ·nm/mol/e²)
KE = 138.93545764438198

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)

from math import erfc as _m_erfc


def _erfc(x):
    return np.vectorize(_m_erfc, otypes=[np.float64])(x)


def mic_diff(x_j, x_i, inv_box=None):
    """x_j − x_i with minimum-image convention (periodic) or raw (aperiodic).

    Fractional wrap: u = diff·A^{-1} (row convention), round to nearest
    integer, unwrap.  Box is an initialization-time constant (spec §6.5).
    """
    diff = x_j - x_i
    if inv_box is None:
        return diff
    u = np.einsum("rc,dc->rd", diff, inv_box)
    u -= np.rint(u)
    return np.einsum("rd,cd->rc", u, np.linalg.inv(inv_box))


def _inside(r2, rc2):
    if BOUNDARY == "lt":
        return r2 < rc2
    return r2 <= rc2


class NeighborList:
    """Shared-across-replicas union list (spec §5.2)."""

    def __init__(self, pairs: np.ndarray):
        self.pairs = pairs  # (P, 2) int, i < j


def build_union_list(x: np.ndarray, rc2: float, skin2: float,
                     box: np.ndarray | None = None) -> NeighborList:
    """x: (N, R, 3). List radius = cutoff + skin; union over replicas.

    Minimum image (periodic) — a pair whose raw separation exceeds the
    list radius but whose MIC distance is inside it MUST be in the list
    (M7's raison d'être).
    """
    N, R, _ = x.shape
    inv_box = np.linalg.inv(box) if box is not None else None
    d2min = None
    for r in range(R):
        xi = x[:, r, :]
        diff = xi[:, None, :] - xi[None, :, :]
        if inv_box is not None:
            u = np.einsum("abc,dc->abd", diff, inv_box)
            u -= np.rint(u)
            diff = np.einsum("abd,cd->abc", u, box)
        d2 = np.sum(diff * diff, axis=-1)
        d2min = d2 if d2min is None else np.minimum(d2min, d2)
    iu, ju = np.triu_indices(N, k=1)
    keep = d2min[iu, ju] < (np.sqrt(rc2) + np.sqrt(skin2)) ** 2
    return NeighborList(np.stack([iu[keep], ju[keep]], axis=1))


def direct_space(nb, x: np.ndarray, excl_set: set, f_acc, alpha: float,
                 rc: float, list_: NeighborList | None = None,
                 box: np.ndarray | None = None, e_acc=None):
    """Direct-space Coulomb(erfc) + LJ over the shared (or full) list.

    Masking is multiply-by-zero: out-of-cutoff pairs flow through the same
    arithmetic with an exact 0 factor (pillar 5 + §2.1 no-branch rule).
    """
    q = np.array([a.q.value for a in nb.atoms])
    sig = np.array([a.sigma.value for a in nb.atoms])
    eps = np.array([a.epsilon.value for a in nb.atoms])
    N, R, _ = x.shape
    rc2 = rc * rc
    inv_box = np.linalg.inv(box) if box is not None else None

    if list_ is None:
        iu, ju = np.triu_indices(N, k=1)
        pairs = np.stack([iu, ju], axis=1)
    else:
        pairs = list_.pairs

    mask_std = np.ones(len(pairs), dtype=bool)
    for (a, b) in excl_set:
        mask_std &= ~(((pairs[:, 0] == a) & (pairs[:, 1] == b)) |
                      ((pairs[:, 0] == b) & (pairs[:, 1] == a)))
    pairs = pairs[mask_std]

    E = 0.0
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    sig_ij = 0.5 * (sig[i_idx] + sig[j_idx])
    eps_ij = np.sqrt(eps[i_idx] * eps[j_idx])
    q_ij = q[i_idx] * q[j_idx]
    sr6_base = sig_ij ** 6

    for p in range(len(pairs)):
        i, j = int(i_idx[p]), int(j_idx[p])
        diff = mic_diff(x[j], x[i], inv_box)     # (R, 3)
        r2 = np.sum(diff * diff, axis=-1)        # (R,)
        inside = _inside(r2, rc2).astype(np.float64)   # exact 0/1 mask
        r2s = np.where(r2 < 1e-12, 1e-12, r2)
        r = np.sqrt(r2s)
        inv_r = 1.0 / r
        inv_r2 = inv_r * inv_r
        sr6 = sr6_base[p] * inv_r2 ** 3
        e_lj = 4.0 * eps_ij[p] * (sr6 * sr6 - sr6)
        # dU/dr (radial derivative; force on i = U'(r) * rhat_ij)
        du_lj = -24.0 * eps_ij[p] * (2.0 * sr6 * sr6 - sr6) * inv_r
        ar = alpha * r
        er = _erfc(ar)
        e_c = KE * q_ij[p] * er * inv_r
        dcoul_dr = -(er * inv_r2 + 2.0 * alpha * _INV_SQRT_PI
                     * np.exp(-ar * ar) * inv_r)   # d/dr [erfc(a r)/r]
        du_c = KE * q_ij[p] * dcoul_dr
        coef = (du_lj + du_c) * inside * inv_r     # F_i = coef * diff
        if e_acc is not None:
            e_acc.add("direct", (e_lj + e_c) * inside)
        fi = coef[:, None] * diff
        f_acc.add_to(i, fi)
        f_acc.add_to(j, -fi)
    return E


def exceptions_energy_forces(nb, x, f_acc, alpha: float,
                             box: np.ndarray | None = None, e_acc=None):
    """Exception pairs: KE (q_exc - q_full erf(alpha r))/r + LJ_exc."""
    E = 0.0
    q = np.array([a.q.value for a in nb.atoms])
    inv_box = np.linalg.inv(box) if box is not None else None
    for e in nb.exceptions:
        i, j = e.a, e.b
        diff = mic_diff(x[j], x[i], inv_box)
        r2 = np.sum(diff * diff, axis=-1)
        r2s = np.where(r2 < 1e-12, 1e-12, r2)
        r = np.sqrt(r2s)
        inv_r = 1.0 / r
        inv_r2 = inv_r * inv_r
        q_exc = e.chargeProd.value
        sig_e = e.sigma.value
        eps_e = e.epsilon.value
        q_full = q[i] * q[j]

        erfc_ar = _erfc(alpha * r)
        erf_ar = 1.0 - erfc_ar
        e_c = KE * (q_exc - q_full * erf_ar) * inv_r
        d_c = KE * (-q_exc * inv_r2
                    - q_full * (2.0 * alpha * _INV_SQRT_PI
                                * np.exp(-(alpha * r) ** 2) * inv_r
                                - erf_ar * inv_r2))
        sr6 = (sig_e ** 6) * inv_r2 ** 3
        e_lj = 4.0 * eps_e * (sr6 * sr6 - sr6)
        du_lj = -24.0 * eps_e * (2.0 * sr6 * sr6 - sr6) * inv_r

        du_total = d_c + du_lj                # dU/dr (radial)
        if e_acc is not None:
            e_acc.add("exceptions", e_c + e_lj)
        fi = du_total[:, None] * inv_r[:, None] * diff
        f_acc.add_to(i, fi)
        f_acc.add_to(j, -fi)
    return E


def dispersion_tail(nb, box_volume: float, rc: float) -> float:
    """OpenMM NonbondedForceImpl::calcDispersionCorrection (verified
    against source, openmmapi/src/NonbondedForceImpl.cpp):

        class pairs: within class n(n+1)/2 (self-pairs included), cross n1*n2
        <c6>  = sum2 / (N(N+1)/2),  <c12> = sum1 / (N(N+1)/2)
        E     = 8 pi N^2 ( <c12>/(9 rc^9) - <c6>/(3 rc^3) )   [no switch]

    Energy only, no forces (OpenMM semantics).
    """
    sig = np.array([a.sigma.value for a in nb.atoms])
    eps = np.array([a.epsilon.value for a in nb.atoms])
    N = len(sig)
    num_interactions = N * (N + 1) / 2.0
    sum1 = 0.0
    sum2 = 0.0
    for i in range(N):
        for j in range(i, N):          # unordered incl self-pairs
            s_ij = 0.5 * (sig[i] + sig[j])
            e_ij = np.sqrt(eps[i] * eps[j])
            s6 = s_ij ** 6
            sum1 += e_ij * s6 * s6
            sum2 += e_ij * s6
    c12_avg = sum1 / num_interactions
    c6_avg = sum2 / num_interactions
    # NOTE: the published source return-line appears to lack the 1/V factor,
    # but the executed 8.6 wheel scales exactly as 1/V (verified empirically
    # to 12 digits by a two-volume test) and dimensional analysis requires
    # it.  Pinned to executed behavior; I-012-style empirical invariant.
    return float(8 * np.pi * N * N / box_volume * (
        c12_avg / (9 * rc ** 9) - c6_avg / (3 * rc ** 3)))
