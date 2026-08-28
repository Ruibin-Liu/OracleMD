"""λ layer (M1): ParamExpr, frozen softcore (spec §4.3/§4.6), ∂U/∂λ.

Frozen analytic definition (spec v1.1.1 §4.3):
    U_sc(r; λ) = 4 λ^a ε [ S^{-2} - S^{-1} ],  S = α(1-λ)^b + (r/σ)^c
    with α = 0.5, a = 1, b = 1, c = 6; σ_ij, ε_ij by Lorentz-Berthelot on
    the λ-interpolated PER-ATOM parameters (combine-then-softcore).
    Direction: spec -> A2 (openmmtools configured to match this definition).

Electrostatics: linear charge scaling q_i(λ) = λ q_i on alchemical atoms
(single λ component at M1; multi-component λ and REST2 β are M3/backlog).

∂U/∂λ: every λ-dependent contribution also produces its analytic λ
derivative, accumulated into a separate fixed-point register (state-owned,
spec §4.4).  The PME three places (recip / self / exclusion correction) all
carry the same q(λ) = q_A + λ Δq interpolation, so each is a quadratic in
λ with coefficients computed once per conformation (C3 quadratic expansion,
spec §5.1) — M16/M17 territory.
"""
from __future__ import annotations

import numpy as np

from . import fxp
from .dynamics import KB_KJ  # noqa: F401 (unit constants)
from .engine import ForceAccumulator
from .ir import IRSystem
from .nonbonded import KE, mic_diff
from .pme import PmeGrid, frac_coords, reciprocal_energy
from math import erfc as _m_erfc

# frozen softcore parameters (spec §4.3)
ALPHASC, A_EXP, B_EXP, C_EXP = 0.5, 1, 1, 6
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def _erfc(x):
    return np.vectorize(_m_erfc, otypes=[np.float64])(x)


# ---------------------------------------------------------------- softcore

def softcore_lj(r, sigma, eps, lam):
    """U_sc and derivatives dU/dr, dU/dλ.  Arrays broadcast; r > 0."""
    S = ALPHASC * (1.0 - lam) ** B_EXP + (r / sigma) ** C_EXP
    S = np.where(S < 1e-300, 1e-300, S)
    inv_S = 1.0 / S
    pref = 4.0 * eps * lam ** A_EXP
    U = pref * (inv_S * inv_S - inv_S)
    # dU/dr: dS/dr = c r^{c-1} / σ^c = c S_r-part / r ; here S has explicit r
    dS_dr = C_EXP * (r / sigma) ** C_EXP / r
    dU_dS = pref * (-2.0 * inv_S ** 3 + inv_S ** 2)
    dU_dr = dU_dS * dS_dr
    # dU/dλ: λ^a prefactor + S's λ dependence (b=1)
    dS_dlam = -ALPHASC * B_EXP * (1.0 - lam) ** (B_EXP - 1)
    du_dlam = 4.0 * eps * A_EXP * lam ** (A_EXP - 1) * (inv_S ** 2 - inv_S) \
        + dU_dS * dS_dlam
    return U, dU_dr, du_dlam


# ---------------------------------------------------------------- alchemical system

class AlchemicalSystem:
    """Wraps an IRSystem with a set of alchemical atoms.

    λ = 0: alchemical atoms fully decoupled (U_sc = 0, q = 0)
    λ = 1: fully coupled — must equal the unmodified system (M9).
    """

    def __init__(self, base: IRSystem, alchemical_atoms, *,
                 softcore_lj=True, scale_charges=True,
                 alch_alch_lj="full", alchemical_bonds=None):
        self.base = base
        self.alch = sorted(set(int(a) for a in alchemical_atoms))
        self.softcore_lj = softcore_lj
        self.scale_charges = scale_charges
        # perturbed bonds: {bond_index: (kA, kB)} — k(λ) = kA + λ(kB-kA),
        # r0 fixed (spec: 柔性微扰键; RBFE dual-topology groundwork)
        self.alchemical_bonds = dict(alchemical_bonds or {})
        # decouple semantics (openmmtools default, spec §4.6 direction):
        # solute-internal LJ stays FULLY coupled at every λ; only
        # solute-environment sterics soften.  "all" = annihilate (alch-alch
        # also softens) — kept for the A6 analytic sweep.
        assert alch_alch_lj in ("full", "all")
        self.alch_alch_lj = alch_alch_lj
        nb = base.nonbonded
        self.has_nb = nb is not None
        self.q0 = (np.array([a.q.value for a in nb.atoms])
                   if nb is not None else None)
        self.sig0 = (np.array([a.sigma.value for a in nb.atoms])
                     if nb is not None else None)
        self.eps0 = (np.array([a.epsilon.value for a in nb.atoms])
                     if nb is not None else None)
        self.alch_set = set(self.alch)

    def q_at(self, lam):
        q = self.q0.copy()
        if self.scale_charges and self.q0 is not None:
            q[self.alch] *= lam
        return q

    def is_alch_pair(self, i, j):
        return (i in self.alch_set) or (j in self.alch_set)


# ---------------------------------------------------------------- single point at λ

def single_point_lambda(alch: AlchemicalSystem, x: np.ndarray, lam: float,
                        e_acc=None) -> dict:
    """Single-point U(x; λ), F(x; λ), dU/dλ(x; λ) for R=1...R replicas.

    λ-dependence:
      bonded terms: none (M1: no alchemical bonds — perturbed bonds arrive
        with the dual-topology RBFE layer, M1.5)
      LJ on pairs with >=1 alchemical atom: softcore (frozen form)
      Coulomb: q_i(λ) = λ q_i on alchemical atoms — direct erfc term linear
        in the pair product, recip/self/exclusion quadratic (C3 machinery)
      dispersion tail: λ-interpolated ε,σ enter the class averages
    """
    base = alch.base
    nb = base.nonbonded
    N, R, _ = x.shape
    f_acc = ForceAccumulator(N, R)
    dUdl = np.zeros(R)
    energies: dict[str, float] = {}
    from .energy import EnergyAccumulator
    if e_acc is None:
        e_acc = EnergyAccumulator(R)

    from .bonded import bonded_terms_energy_forces
    if alch.alchemical_bonds:
        # λ-interpolated bonds: k(λ) = kA + λ(kB-kA); ∂U/∂λ analytic
        import dataclasses
        plain = [t for i, t in enumerate(base.bonds)
                 if i not in alch.alchemical_bonds]
        bonded_terms_energy_forces("bond", plain, x, f_acc, e_acc)
        for idx, (kA, kB) in alch.alchemical_bonds.items():
            t = base.bonds[idx]
            k_l = kA + lam * (kB - kA)
            rij = x[t.atoms[1]] - x[t.atoms[0]]
            r = np.sqrt(np.sum(rij * rij, axis=-1))
            d = r - t.length.value
            e_term = 0.5 * k_l * d ** 2
            e_acc.add("HarmonicBondForce", e_term)
            coef = k_l * d / r
            fi = coef[:, None] * rij
            f_acc.add_to(t.atoms[0], fi)
            f_acc.add_to(t.atoms[1], -fi)
            dUdl += np.sum(0.5 * (kB - kA) * d ** 2)
    else:
        bonded_terms_energy_forces("bond", base.bonds, x, f_acc, e_acc)
    bonded_terms_energy_forces("angle", base.angles, x, f_acc, e_acc)
    bonded_terms_energy_forces("torsion", base.torsions, x, f_acc, e_acc)

    if not alch.has_nb:
        return {"energies": {"HarmonicBondForce": 0.0},
                "E": sum(e_acc.energies().values()),
                "forces": f_acc.forces(),
                "dUdl": dUdl / max(1, x.shape[1]),
                "sticky_overflow": f_acc.acc.sticky_overflow}
    alpha = nb.ewald_alpha
    rc = nb.cutoff
    rc2 = rc * rc
    box = base.box if nb.periodic else None
    inv_box = np.linalg.inv(box) if box is not None else None
    q_lam = alch.q_at(lam)

    excl = {(e.a, e.b) for e in nb.exceptions}
    exc_by_pair = {(e.a, e.b): e for e in nb.exceptions}

    # ---- direct space over all non-exception pairs
    iu, ju = np.triu_indices(N, k=1)
    sig_ij = 0.5 * (alch.sig0[iu] + alch.sig0[ju])
    eps_ij = np.sqrt(alch.eps0[iu] * alch.eps0[ju])
    q_ij_lam = q_lam[iu] * q_lam[ju]
    # d(q_i q_j)/dλ for the pair: only alchemical atoms scale
    di = np.isin(iu, alch.alch)
    dj = np.isin(ju, alch.alch)
    # q_i(λ)q_j(λ) = λ² q q if both, λ q q if one, q q if none
    both = di & dj
    one = di ^ dj
    q_ij_full = alch.q0[iu] * alch.q0[ju]
    dqdl_pair = np.where(both, 2.0 * lam * q_ij_full,
                         np.where(one, q_ij_full, 0.0))

    pair_is_alch = di | dj
    E_direct = 0.0
    for p in range(len(iu)):
        i, j = int(iu[p]), int(ju[p])
        if (i, j) in excl or (j, i) in excl:
            continue
        diff = mic_diff(x[j], x[i], inv_box)
        r2 = np.sum(diff * diff, axis=-1)
        inside = (r2 < rc2).astype(np.float64)
        r2s = np.where(r2 < 1e-12, 1e-12, r2)
        r = np.sqrt(r2s)
        inv_r = 1.0 / r
        inv_r2 = inv_r * inv_r
        ar = alpha * r
        er = _erfc(ar)
        e_c = KE * q_ij_lam[p] * er * inv_r
        dcoul_dr = -(er * inv_r2 + 2.0 * alpha * _INV_SQRT_PI
                     * np.exp(-ar * ar) * inv_r)
        du_c = KE * q_ij_lam[p] * dcoul_dr
        dudl_c = KE * dqdl_pair[p] * er * inv_r
        both_alch = i in alch.alch_set and j in alch.alch_set
        if (pair_is_alch[p] and alch.softcore_lj
                and not (both_alch and alch.alch_alch_lj == "full")):
            U, dU_dr, dU_dlam = softcore_lj(r, sig_ij[p], eps_ij[p], lam)
            du_lj = dU_dr
            e_lj = U
        else:
            sr6 = (sig_ij[p] ** 6) * inv_r2 ** 3
            e_lj = 4.0 * eps_ij[p] * (sr6 * sr6 - sr6)
            du_lj = -24.0 * eps_ij[p] * (2.0 * sr6 * sr6 - sr6) * inv_r
            dU_dlam = np.zeros_like(r)
        coef = (du_lj + du_c) * inside * inv_r
        fi = coef[:, None] * diff
        f_acc.add_to(i, fi)
        f_acc.add_to(j, -fi)
        E_direct += float(np.sum((e_lj + e_c) * inside))
        dUdl += np.sum(((dU_dlam + dudl_c) * inside))
    energies["direct"] = E_direct

    # ---- exceptions / exclusions (charge products scale when alchemical)
    E_exc = 0.0
    for e in nb.exceptions:
        i, j = e.a, e.b
        diff = mic_diff(x[j], x[i], inv_box if
                        (nb.exceptions_use_periodic and inv_box is not None)
                        else None)
        r2 = np.sum(diff * diff, axis=-1)
        r2s = np.where(r2 < 1e-12, 1e-12, r2)
        r = np.sqrt(r2s)
        inv_r = 1.0 / r
        inv_r2 = inv_r * inv_r
        q_exc = e.chargeProd.value
        sig_e = e.sigma.value
        eps_e = e.epsilon.value
        ai = i in alch.alch_set
        aj = j in alch.alch_set
        # exception chargeProd scales like the pair product if alchemical
        # (RBFE convention: exceptions among alchemical atoms carry the
        #  same λ interpolation as the parent nonbonded terms)
        if ai and aj:
            q_exc_l = q_exc * lam * lam
            dq_exc = 2.0 * lam * q_exc
        elif ai or aj:
            q_exc_l = q_exc * lam
            dq_exc = q_exc
        else:
            q_exc_l = q_exc + 0.0 * lam
            dq_exc = 0.0
        q_full_lam = q_lam[i] * q_lam[j]
        dq_full = (2 * lam * alch.q0[i] * alch.q0[j] if (ai and aj)
                   else (alch.q0[i] * alch.q0[j] if (ai ^ aj) else 0.0))
        erfc_ar = _erfc(alpha * r)
        erf_ar = 1.0 - erfc_ar
        e_c = KE * (q_exc_l - q_full_lam * erf_ar) * inv_r
        d_c = KE * (-q_exc_l * inv_r2
                    - q_full_lam * (2.0 * alpha * _INV_SQRT_PI
                                    * np.exp(-(alpha * r) ** 2) * inv_r
                                    - erf_ar * inv_r2))
        dudl_c = KE * (dq_exc - dq_full * erf_ar) * inv_r
        sr6 = (sig_e ** 6) * inv_r2 ** 3
        e_lj = 4.0 * eps_e * (sr6 * sr6 - sr6)
        du_lj = -24.0 * eps_e * (2.0 * sr6 * sr6 - sr6) * inv_r
        du_total = d_c + du_lj
        fi = du_total[:, None] * inv_r[:, None] * diff
        f_acc.add_to(i, fi)
        f_acc.add_to(j, -fi)
        E_exc += float(np.sum(e_c + e_lj))
        dUdl += np.sum(dudl_c)
    energies["exceptions"] = E_exc

    # ---- PME with λ charges + self + ∂/∂λ via the C3 quadratic expansion
    #
    # q(λ) = q_A + λΔq (Δq = -q_alch).  E_recip is a quadratic form in q:
    #     E(λ) = A + 2λB + λ²C
    #     A = E(q_A),  C = E(Δq),  B = ½[E(q_B) − A − C]  (q_B = q(1))
    # dU/dλ = 2B + 2λC exactly; E at ANY λ comes free (M16).
    if nb.periodic:
        g = PmeGrid(base.box, nb.grid, alpha)
        grid_fx = fxp.q16_48_grid(nb.grid)
        # q(λ) = q_env + λ·q_alch :  λ=0 decoupled, λ=1 fully coupled
        # (q_at() convention; the quadratic rides the SAME direction)
        q_A = alch.q_at(0.0)               # decoupled charges
        q_B = alch.q_at(1.0)               # full charges
        q_D = q_B - q_A                    # Δq = +q_alch
        E_recip = 0.0
        dUdl_recip = 0.0
        f_recip = np.zeros((N, R, 3))
        for r in range(R):
            frac = frac_coords(x[:, r, :], g.inv_box)
            E_l, F = reciprocal_energy(g, q_lam, frac, grid_fx)
            E_A = reciprocal_energy(g, q_A, frac, fxp.q16_48_grid(nb.grid))[0]
            E_B = reciprocal_energy(g, q_B, frac, fxp.q16_48_grid(nb.grid))[0]
            E_D = reciprocal_energy(g, q_D, frac, fxp.q16_48_grid(nb.grid))[0]
            E_recip += E_l
            f_recip[:, r, :] = F
            Cq = E_D
            Bq = 0.5 * (E_B - E_A - Cq)
            dUdl_recip += 2.0 * Bq + 2.0 * lam * Cq
            if grid_fx.sticky_overflow:
                raise RuntimeError("Q16.48 grid overflow")
        energies["recip"] = E_recip
        E_self_A = -KE * alpha / np.sqrt(np.pi) * float(np.sum(q_A * q_A))
        E_self_B = -KE * alpha / np.sqrt(np.pi) * float(np.sum(q_B * q_B))
        E_self_D = -KE * alpha / np.sqrt(np.pi) * float(np.sum(q_D * q_D))
        Cs, Bs = E_self_D, 0.5 * (E_self_B - E_self_A - E_self_D)
        energies["self"] = R * (E_self_A + 2 * lam * Bs + lam * lam * Cs)
        dUdl += dUdl_recip + R * (2.0 * Bs + 2.0 * lam * Cs)
        for i in range(N):
            f_acc.add_to(i, f_recip[i])

    # ---- dispersion tail with λ-interpolated parameters
    if nb.periodic and nb.use_dispersion_correction:
        sig_l = alch.sig0.copy()
        eps_l = alch.eps0.copy()
        # softcore handles alchemical LJ in direct space; the homogeneous
        # tail keeps full parameters (openmmtools treats the correction as
        # part of the environment; documented simplification for M1)
        V = float(abs(np.linalg.det(base.box)))
        from .nonbonded import dispersion_tail
        energies["tail"] = R * dispersion_tail(nb, V, rc)

    energies_total = sum(energies.values())
    return {"energies": energies, "E": energies_total,
            "forces": f_acc.forces(), "dUdl": dUdl / max(R, 1),
            "sticky_overflow": f_acc.acc.sticky_overflow}
