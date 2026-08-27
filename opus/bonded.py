"""Bonded terms: energy (OpenMM-convention) + analytic forces.

Functional forms (pinned against OpenMM 8.6 Reference platform):
    E_bond     = (1/2) k (r - r0)^2
    E_angle    = (1/2) k (theta - theta0)^2,  theta = atan2(|a x b|, a.b)
    E_torsion  = k (1 + cos(n phi - phase)),
                 phi = atan2( (G x (F x G)).(G x H), (F x G).(G x H) * |G| )

Force derivations (validated against mpmath 40-digit central differences in
tests/test_gradient_oracle.py):

Angle (a = x_i - x_j, b = x_k - x_j, n = a x b, s = |n|, c = a.b):
    grad_a theta = ( c * (b x n)/s - s * b ) / (s^2 + c^2)
    grad_b theta = ( c * (n x a)/s - s * a ) / (s^2 + c^2)
    F_i = -k dtheta grad_a theta ; F_k symmetric ; F_j = -(F_i + F_k)

Torsion (F = x_j - x_i, G = x_k - x_j, H = x_l - x_k, A = F x G,
 B = G x H, W = G x A, s = W.B, c = (A.B)|G|, f = dE/dphi):
    grad_F phi = ( c [ |G|^2 B - (G.B) G ] - s |G| (G x B) ) / (s^2 + c^2)
    grad_H phi = ( c [ |G|^2 A - (G.A) G ] ... (derived; see code)
    grad_G phi via the full differential (below).
    F_i = +f grad_F phi ; F_l = -f grad_H phi
    F_k = -f (grad_G phi - grad_H phi) ; F_j = -f (grad_F phi - grad_G phi)
    (sum of the four forces is identically zero — translation invariance)
"""
from __future__ import annotations

import numpy as np


def bond_energy_forces(terms, x, f_acc, e_acc=None) -> float:
    E = 0
    for t in terms:
        i, j = t.atoms
        rij = x[j] - x[i]                       # (R, 3)
        r = np.sqrt(np.sum(rij * rij, axis=-1))
        dtheta = r - t.length.value
        e_term = 0.5 * t.k.value * dtheta ** 2  # (R,)
        if e_acc is not None:
            e_acc.add("HarmonicBondForce", e_term)
        E += int(np.sum(np.rint(e_term * (1 << 30))))
        coef = t.k.value * dtheta / r           # on rij for atom i
        fi = coef[:, None] * rij
        f_acc.add_to(i, fi)
        f_acc.add_to(j, -fi)
    return E / (1 << 30)


def _angle_common(a, b):
    n = np.cross(a, b)
    s = np.sqrt(np.sum(n * n, axis=-1))
    c = np.sum(a * b, axis=-1)
    safe_s = np.where(s < 1e-12, 1e-12, s)
    theta = np.arctan2(s, c)
    return n, s, c, safe_s, theta


def angle_energy_forces(terms, x, f_acc, e_acc=None) -> float:
    E = 0
    for t in terms:
        i, j, k = t.atoms
        a = x[i] - x[j]
        b = x[k] - x[j]
        n, s, c, safe_s, theta = _angle_common(a, b)
        dth = theta - t.theta0.value
        e_term = 0.5 * t.k.value * dth ** 2
        if e_acc is not None:
            e_acc.add("HarmonicAngleForce", e_term)
        E += int(np.sum(np.rint(e_term * (1 << 30))))
        coef = t.k.value * dth / (s * s + c * c)   # dE * 1/(s^2+c^2)
        ga = coef[:, None] * ((c[:, None] * np.cross(b, n)) / safe_s[:, None]
                              - s[:, None] * b)
        gb = coef[:, None] * ((c[:, None] * np.cross(n, a)) / safe_s[:, None]
                              - s[:, None] * a)
        f_acc.add_to(i, -ga)
        f_acc.add_to(k, -gb)
        f_acc.add_to(j, ga + gb)
    return E


def _torsion_common(F, G, H):
    A = np.cross(F, G)
    B = np.cross(G, H)
    g2 = np.sum(G * G, axis=-1)
    g = np.sqrt(g2)
    W = np.cross(G, A)
    s = np.sum(W * B, axis=-1)
    c = np.sum(A * B, axis=-1) * g
    phi = np.arctan2(s, c)
    return A, B, g, g2, W, s, c, phi


def torsion_energy_forces(terms, x, f_acc, e_acc=None) -> float:
    E = 0
    for t in terms:
        i, j, k, l = t.atoms
        F = x[j] - x[i]
        G = x[k] - x[j]
        H = x[l] - x[k]
        A, B, g, g2, W, s, c, phi = _torsion_common(F, G, H)
        n_, ph = t.periodicity, t.phase.value
        e_term = t.k.value * (1.0 + np.cos(n_ * phi - ph))
        if e_acc is not None:
            e_acc.add("PeriodicTorsionForce", e_term)
        E += int(np.sum(np.rint(e_term * (1 << 30))))
        f = -t.k.value * n_ * np.sin(n_ * phi - ph)    # dE/dphi

        denom = (s * s + c * c)
        # grad_F phi
        gF = (c[:, None] * (g2[:, None] * B - np.sum(G * B, axis=-1)[:, None] * G)
              - (s * g)[:, None] * np.cross(G, B)) / denom[:, None]
        # grad_H phi:  s = W.B with dB/dH = G x dH  ->  grad_H s = B?? derive:
        #   grad_H s = (G x W) x ... :  W.(G x dH) = dH.(W x G)? triple:
        #   W·(G×dH) = G·(dH×W) = dH·(W×G)  -> grad_H s = W x G
        #   grad_H c = |G| (A x G)   [mirror of grad_F c = |G| (G x B)]
        gH = (c[:, None] * np.cross(W, G)
              - (s * g)[:, None] * np.cross(A, G)) / denom[:, None]
        # grad_G phi (full differential):
        #   ds: dG contributes  dG x A + F (G.dG) - dG (G.F)  [to W=d(Gx(FxG))]
        #       plus W . (dG x H)
        #   dc: (B x F + H x A)|G| + (A.B) G/|G|
        gs = (np.cross(A, B) + np.sum(F * B, axis=-1)[:, None] * G
              - np.sum(G * F, axis=-1)[:, None] * B
              + np.cross(H, W))
        gc = (np.cross(B, F) + np.cross(H, A)) * g[:, None] \
            + (np.sum(A * B, axis=-1) / np.where(g < 1e-12, 1e-12, g))[:, None] * G
        gG = (c[:, None] * gs - s[:, None] * gc) / denom[:, None]

        f_acc.add_to(i, f[:, None] * gF)
        f_acc.add_to(j, -f[:, None] * (gF - gG))
        f_acc.add_to(k, -f[:, None] * (gG - gH))
        f_acc.add_to(l, -f[:, None] * gH)
    return E


_KIND = {
    "bond": bond_energy_forces,
    "angle": angle_energy_forces,
    "torsion": torsion_energy_forces,
}


def bonded_terms_energy_forces(kind: str, terms, x, f_acc, e_acc=None) -> float:
    if not terms:
        return 0.0
    return _KIND[kind](terms, x, f_acc, e_acc)
