"""Direct-space nonbonded kernel (production, M2).

Semantics (must match opus.nonbonded.direct_space):
  - LJ Lorentz-Berthelot: sigma_ij = 0.5(si+sj), eps_ij = sqrt(ei*ej)
  - Ewald real-space Coulomb: F = KE*q_i*q_j*[erfc(a r)/r^2
    + (2a/sqrt(pi)) e^{-a^2 r^2} / r]  (KE folded outside; kernel returns
    forces in unit-charge units, caller folds q_i * KE -- prefold form)
  - out-of-cutoff: exact +0 (branch-skip == absent, pillar-5 legal);
    exclusions handled by the caller via list construction ("absent")
  - erfc via gpu.erfc_poly (generated coefficients)

The prototype measured 15.65 ms/step @R=48/N=60k/skin0.35 (E0g/G); this
module is the same kernel, productized.
"""
from __future__ import annotations

import numpy as np

from .erfc_poly import emit_cuda

_RSQRTPI = 0.5641895835477563

_KERNEL_TMPL = r"""
#define RSQRTPI %s

__ERFC__

extern "C" __global__ void direct_f64(
    const double* __restrict__ x, const int* __restrict__ nlist,
    const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig,
    const double* __restrict__ eps, const double* __restrict__ se,
    double* __restrict__ F, int N, int R, int maxnb,
    double rc2, double alpha, double Lx, double Ly, double Lz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a * R;
    if (a >= N) return;
    double xi = x[((long long)a * R + r) * 3],
           yi = x[((long long)a * R + r) * 3 + 1],
           zi = x[((long long)a * R + r) * 3 + 2];
    double qi = q[a], si = sig[a], sei = se[a];
    double lx = 0., ly = 0., lz = 0., cx = 0., cy = 0., cz = 0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a * maxnb + k];
        double dx = x[((long long)j * R + r) * 3] - xi,
               dy = x[((long long)j * R + r) * 3 + 1] - yi,
               dz = x[((long long)j * R + r) * 3 + 2] - zi;
        dx -= round(dx / Lx) * Lx;   // minimum image (orthogonal box)
        dy -= round(dy / Ly) * Ly;
        dz -= round(dz / Lz) * Lz;
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2) {
            double invr2 = 1.0 / r2;
            double rr = sqrt(r2);
            double invr = rr * invr2;
            double s = 0.5 * (si + sig[j]);
            double sr2 = s * s * invr2, sr6 = sr2 * sr2 * sr2;
            double flj = -24.0 * se[j] * (2.0 * sr6 * sr6 - sr6) * invr2;
            double y = alpha * alpha * r2;
            double emx2 = exp(-y);
            double ec = erfc_poly(alpha * rr, emx2);
            double fc = -q[j] * (ec * invr2 + 2.0 * alpha * RSQRTPI * emx2 * invr) * invr;
            lx += flj * dx; ly += flj * dy; lz += flj * dz;
            cx += fc * dx; cy += fc * dy; cz += fc * dz;
        }
    }
    F[((long long)a * R + r) * 3]     = sei * lx + qi * cx;
    F[((long long)a * R + r) * 3 + 1] = sei * ly + qi * cy;
    F[((long long)a * R + r) * 3 + 2] = sei * lz + qi * cz;
}
"""

_SRC_CACHE: dict[str, object] = {}


def kernel_source() -> str:
    return (_KERNEL_TMPL % (repr(_RSQRTPI))).replace(
        "__ERFC__", emit_cuda())


def direct_forces(x, q, sig, eps, nlist, ncnt, alpha: float, rc: float,
                  box_diag=None, block: int = 128):
    """Direct-space forces (unit-charge units, prefold form).

    x: (N, R, 3) f64; q/sig/eps: (N,); nlist: (N, maxnb) int32;
    ncnt: (N,); box_diag: (3,) f64 orthorhombic box lengths (MIC in-kernel).
    Returns (N, R, 3) f64 = [sqrt(eps_i)*LJ + q_i*coul] (unit-charge coul;
    caller folds KE). Requires cupy + CUDA.
    """
    import cupy as cp
    src = kernel_source()
    if "mod" not in _SRC_CACHE:
        _SRC_CACHE["mod"] = cp.RawModule(code=src)
        _SRC_CACHE["k"] = _SRC_CACHE["mod"].get_function("direct_f64")
    k = _SRC_CACHE["k"]
    N, R, _ = x.shape
    maxnb = nlist.shape[1]
    se = np.sqrt(eps)
    d = lambda a: cp.asarray(a)
    F = cp.zeros(N * R * 3)
    grid = ((N * R + block - 1) // block,)
    L = np.asarray(box_diag, dtype=np.float64).reshape(3)
    k(grid, (block,),
      (d(x), d(np.ascontiguousarray(nlist)), d(ncnt), d(q), d(sig),
       d(eps), d(se), F, N, R, maxnb, rc * rc, alpha,
       float(L[0]), float(L[1]), float(L[2])))
    cp.cuda.Stream.null.synchronize()
    return F.reshape(N, R, 3)
