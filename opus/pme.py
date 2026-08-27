"""PME — spec §5.1. Structural isomorphism: real grid, B-spline spread,
FFT, influence function; forces are the *exact gradient of the discrete
energy functional* via the adjoint potential grid (energy/force consistent
by construction).

Pillar 2: charge spread accumulates into an int64 Q16.48 grid (quantize
once per contribution, exact integer adds), dequantized pointwise to fp64
for the FFT (in-place same-width reinterpret on GPU).

Oracle ladder: this PME  <->  naive Ewald (below, O(N^2) continuum
reciprocal)  <->  OpenMM Reference PME (pinned alpha/grid).

Conventions (Essmann 1995, AMBER/OpenMM-compatible):
  - order-4 cardinal B-splines M_p; particle at fractional u spreads to
    grid anchors a = floor(u*N) - p + 1 .. a + p - 1, weights M_p(u*N - g);
  - b_d(m) = sum_{t=1}^{p-1} M_p(t) exp(2πi m t/N_d)  (assignment FT);
  - E_recip = (1/2) sum_{m != 0} |Qhat(m)|^2 c(m),
    c(m) = (4π / V |k|^2) exp(-|k|^2 / 4α^2) / prod_d |b_d(m_d)|^2;
  - k(m) = 2π A^{-T} n, n = fftfreq indices (integer, half-space).
"""
from __future__ import annotations

import numpy as np

from . import fxp
from .nonbonded import KE


# ---------------------------------------------------------------- B-splines

def bspline_weights(u_frac: float, n_grid: int, p: int = 4):
    """Anchor indices and weights for one dimension.

    Returns (anchors (p,), weights (p,)) with anchors = a-p+1+t,
    a = floor(u*n_grid), w_t = M_p(u*n_grid - g_t).
    """
    x = u_frac * n_grid
    a = int(np.floor(x))
    anchors = a - p + 1 + np.arange(p)
    w = np.array([cardinal_bspline(x - g, p) for g in anchors])
    return anchors, w


def cardinal_bspline(x: float, p: int) -> float:
    """M_p(x) on support (0, p), via recursion."""
    if p == 1:
        return 1.0 if 0.0 <= x < 1.0 else 0.0
    return (x / (p - 1)) * cardinal_bspline(x, p - 1) + \
           ((p - x) / (p - 1)) * cardinal_bspline(x - 1, p - 1)


def bspline_deriv(x: float, p: int) -> float:
    """M_p'(x) = M_{p-1}(x) - M_{p-1}(x-1)."""
    if p == 1:
        return 0.0
    return cardinal_bspline(x, p - 1) - cardinal_bspline(x - 1, p - 1)


def b_factor(m: np.ndarray, n_grid: int, p: int = 4) -> np.ndarray:
    """|b(m)| for one grid dimension; b(m) = sum_t M_p(t) e^{2πi m t/N}."""
    t = np.arange(1, p)
    M = np.array([cardinal_bspline(float(tt), p) for tt in t])
    b = np.zeros(m.shape, dtype=np.complex128)
    for i, mm in enumerate(m.ravel()):
        b.ravel()[i] = np.sum(M * np.exp(2j * np.pi * mm * t / n_grid))
    return np.abs(b)


# ---------------------------------------------------------------- PME core

class PmeGrid:
    """Per-system constants (box-fixed — spec §6.5 initialization-time)."""

    def __init__(self, box: np.ndarray, grid: tuple[int, int, int],
                 alpha: float, order: int = 4):
        self.box = np.asarray(box, dtype=float)
        self.grid = tuple(grid)
        self.alpha = alpha
        self.p = order
        self.V = float(abs(np.linalg.det(self.box)))
        self.inv_box = np.linalg.inv(self.box)          # frac = x @ inv_box.T? see frac()
        self.n1, self.n2, self.n3 = grid
        # k-vectors for fftfreq layout
        n = np.array(np.meshgrid(np.fft.fftfreq(self.n1) * self.n1,
                                 np.fft.fftfreq(self.n2) * self.n2,
                                 np.fft.fftfreq(self.n3) * self.n3, indexing="ij"))
        n_int = np.rint(n).astype(int)                    # (3, N1, N2, N3)
        # k = 2π n A^{-T}  -> k components = 2π n @ inv_box.T
        k = 2 * np.pi * np.einsum("dabc,de->eabc", n_int, self.inv_box.T)
        self.k2 = np.sum(k * k, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = KE * (4 * np.pi / (self.V * self.k2)) * np.exp(
                -self.k2 / (4 * alpha * alpha))
        c[0, 0, 0] = 0.0
        b1 = b_factor(np.rint(np.fft.fftfreq(self.n1) * self.n1), self.n1, order)
        b2 = b_factor(np.rint(np.fft.fftfreq(self.n2) * self.n2), self.n2, order)
        b3 = b_factor(np.rint(np.fft.fftfreq(self.n3) * self.n3), self.n3, order)
        bsq = (b1[:, None, None] * b2[None, :, None] * b3[None, None, :]) ** 2
        self.c = c / bsq
        # adjoint normalization: np.fft.ifftn divides by N_grid
        self.n_grid_total = self.n1 * self.n2 * self.n3


def frac_coords(x: np.ndarray, inv_box: np.ndarray) -> np.ndarray:
    """Cartesian (..., 3) -> fractional in [0,1): u = x · A^{-T} rows?  

    Box rows A (a_i); x = Σ u_i a_i  =>  u = x A^{-1} (row-vector convention).
    """
    return np.einsum("...c,dc->...d", x, inv_box)


def spread(frac: np.ndarray, q: np.ndarray, g: PmeGrid,
           grid_fx: fxp.FixedPointAccumulator):
    """Q16.48 fixed-point spread of one replica's charges (pillar 2).

    frac: (N, 3) fractional coordinates in [0,1)."""
    p = g.p
    N = frac.shape[0]
    for i in range(N):
        anchors = []
        weights = []
        for d in range(3):
            u = frac[i, d] % 1.0
            a, w = bspline_weights(u, g.grid[d], p)
            anchors.append(a % g.grid[d])
            weights.append(w)
        w0, w1, w2 = weights
        val = q[i] * w0[:, None, None] * w1[None, :, None] * w2[None, None, :]
        gidx = (anchors[0][:, None, None], anchors[1][None, :, None],
                anchors[2][None, None, :])
        grid_fx.scatter_add_f64(gidx, val)


def reciprocal_energy(g: PmeGrid, q: np.ndarray, frac: np.ndarray,
                      grid_fx: fxp.FixedPointAccumulator):
    """Returns (E_recip, forces_cart (N,3)) — exact discrete-functional gradient.

    Forces: F_i = -q_i Σ_g φ_g ∇_x [Π_d M(u_d - g_d)],  φ = N·ifftn(Q̂·c).
    """
    grid_fx.acc[:] = 0
    spread(frac, q, g, grid_fx)
    qgrid = grid_fx.to_f64()                       # deterministic dequantize
    qhat = np.fft.fftn(qgrid)
    E = 0.5 * float(np.sum(np.abs(qhat) ** 2 * g.c).real)
    phi = g.n_grid_total * np.fft.ifftn(qhat * g.c).real

    N = frac.shape[0]
    F = np.zeros((N, 3))
    p = g.p
    for i in range(N):
        anchors, weights, dws = [], [], []
        for d in range(3):
            u = frac[i, d] % 1.0
            x = u * g.grid[d]
            a = int(np.floor(x))
            anc = a - p + 1 + np.arange(p)
            anchors.append(anc % g.grid[d])
            off = a - p + 1 + np.arange(p)
            weights.append(np.array([cardinal_bspline(x - float(gg), p)
                                     for gg in off]))
            # dM/du = N_d · dM/dx(grid units)
            dws.append(np.array([bspline_deriv(x - float(gg), p)
                                 for gg in off]) * g.grid[d])
        m0, m1, m2 = weights
        ph = phi[np.ix_(anchors[0], anchors[1], anchors[2])]
        dE_du = np.array([
            q[i] * np.sum(dws[0][:, None, None] * m1[None, :, None] * m2[None, None, :] * ph),
            q[i] * np.sum(m0[:, None, None] * dws[1][None, :, None] * m2[None, None, :] * ph),
            q[i] * np.sum(m0[:, None, None] * m1[None, :, None] * dws[2][None, None, :] * ph),
        ])
        # x = Σ_d u_d a_d (rows a) -> dE/dx = A^{-T} dE/du?  u = x A^{-1}
        # (row convention) => du_d/dx_c = (A^{-1})_{c d} => dE/dx_c = Σ_d (A^{-1})_{cd} dE/du_d
        F[i] = -np.einsum("cd,d->c", g.inv_box, dE_du)
    return E, F


def self_energy(q: np.ndarray, alpha: float) -> float:
    return float(np.sum(self_energy_terms(q, alpha)))


def self_energy_terms(q: np.ndarray, alpha: float) -> np.ndarray:
    return -KE * alpha / np.sqrt(np.pi) * q * q


# ------------------------------------------------------- naive Ewald oracle

def naive_ewald_reciprocal(x: np.ndarray, q: np.ndarray, box: np.ndarray,
                           alpha: float, kmax: int = 8) -> float:
    """O(N^2 · kgrid) continuum reciprocal Ewald sum — oracle ladder rung."""
    inv_box = np.linalg.inv(box)
    V = abs(np.linalg.det(box))
    E = 0.0
    n1 = np.arange(-kmax, kmax + 1)
    for n1v in n1:
        for n2v in n1:
            for n3v in n1:
                if (n1v, n2v, n3v) == (0, 0, 0):
                    continue
                n = np.array([n1v, n2v, n3v])
                k = 2 * np.pi * n @ inv_box.T
                k2 = k @ k
                s = np.sum(q * np.exp(1j * (x @ k)))
                E += KE * (4 * np.pi / (V * k2)) * np.exp(-k2 / (4 * alpha * alpha)) \
                     * abs(s) ** 2
    return 0.5 * E
