"""Direct-space forces with Q24.40 fixed-point accumulation (pillar 1).

Semantic source: opus.nonbonded.direct_space + engine.ForceAccumulator --
NOT the E0g k2 bench, and NOT gpu/direct.py's fp64 prefold form (that
stays: it is the floor-baseline k2 shape with its own A1 alignment).

Pillar-1 semantics (spec 3.1): each PAIR contribution is quantized once
(round-half-even of f * 2^40) and integer-added to both endpoints -- the
int64 sum is exact and order-independent.  opus quantizes quantize(fi) to
i and quantize(-fi) to j; since quantize is odd-symmetric, the full-list
kernel's per-thread contributions reproduce this exactly (Newton-3 holds
BITWISE in Q24.40 -- pod-verified).

Bitwise-layer plan (three artifacts, e0-pattern):
  - mirror_opus (tests): alpha=0 kills the transcendental approximations
    (erfc(0)=1, exp(0)=1 exact on every platform); with opus's own
    np.power/ufunc calls this reproduces opus BITWISE in CI -- proving the
    quantization/accumulation/MIC/layout semantics layer;
  - mirror_kernel (this file): the kernel's exact arithmetic (k2 grouping
    s*s*inv_r2, erfc poly with kernel-form t) -- the pod compares the CUDA
    against it BITWISE at alpha=0 (exp(-0) is exact everywhere, the poly
    is pure mul/add), proving the transcription layer;
  - kernel vs opus forces: A1 dual gate end-to-end (poly-erfc dominant
    error, phase-1 style).

Divergences from opus (documented, dual-gated):
  - sr6 path: opus sig_ij**6 / inv_r2**3 hit libm pow (bitwise-unportable
    to CUDA); the kernel uses the k2 grouping sr2 = s*s*inv_r2,
    sr6 = sr2*sr2*sr2 (ulp-level, quantization may flip +/-1 LSB/contrib);
  - erfc: gpu.erfc_poly approximation (max rel 3.2e-14), phase-1 aligned;
  - running-sum saturation: opus saturates the accumulator per add;
    the kernel adds int64 registers (wraparound impossible in-envelope
    per Q-008 + format range == int64 range) and flags overflow in the
    FLOAT domain per contribution (opus add_to semantics) into a sticky
    counter -- checked at graph/window boundaries per spec.
"""
from __future__ import annotations

import numpy as np

from .erfc_poly import emit_cuda

_KERNEL_TMPL = r"""
#define KE %(ke)s
#define INV_SQRT_PI %(ispi)s

__ERFC__

// Q24.40 direct-space forces: per-pair quantize-once, integer accumulate.
// Full list: every ordered pair processed by both endpoint threads
// (Newton-3 bitwise, pod-verified).  MIC in opus order: fractional round
// trip through the box diagonal (u = d*invL; u -= rint(u); d = u*L).
extern "C" __global__ void direct_q(
    const double* __restrict__ x, const int* __restrict__ nlist,
    const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig,
    const double* __restrict__ eps, long long* __restrict__ Fq,
    int* __restrict__ sticky,
    int N, int R, int maxnb, int periodic,
    double rc2, double alpha,
    double Lx, double Ly, double Lz,
    double invlx, double invly, double invlz,
    double scale, double vmaxd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a * R;
    if (a >= N) return;
    long long i = ((long long)a * R + r) * 3;
    double xi = x[i], yi = x[i + 1], zi = x[i + 2];
    double qa = q[a], sa = sig[a], ea = eps[a];
    long long ax = 0, ay = 0, az = 0;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a * maxnb + k];
        long long jb = ((long long)j * R + r) * 3;
        double dx = x[jb] - xi, dy = x[jb + 1] - yi, dz = x[jb + 2] - zi;
        if (periodic) {
            double ux = dx * invlx; ux -= rint(ux); dx = ux * Lx;
            double uy = dy * invly; uy -= rint(uy); dy = uy * Ly;
            double uz = dz * invlz; uz -= rint(uz); dz = uz * Lz;
        }
        double r2 = (dx * dx + dy * dy) + dz * dz;
        double inside = (r2 < rc2) ? 1.0 : 0.0;  // pillar 5: multiply-0
        double r2s = r2 < 1e-12 ? 1e-12 : r2;
        double rr = sqrt(r2s);
        double inv_r = 1.0 / rr;
        double inv_r2 = inv_r * inv_r;
        double s = 0.5 * (sa + sig[j]);
        double sr2 = s * s * inv_r2;
        double sr6 = sr2 * sr2 * sr2;
        double eij = sqrt(ea * eps[j]);
        double dulj = -24.0 * eij * (2.0 * sr6 * sr6 - sr6) * inv_r;
        double ar = alpha * rr;
        double y = ar * ar;
        double emx2 = exp(-y);
        double er = erfc_poly(ar, emx2);
        double dcoul = -(er * inv_r2 + 2.0 * alpha * INV_SQRT_PI * emx2
                         * inv_r);
        double duc = KE * (qa * q[j]) * dcoul;
        double coef = (dulj + duc) * inside * inv_r;
        double fx = coef * dx, fy = coef * dy, fz = coef * dz;
        // clamp bound: float(2^63-1) rounds UP to 2^63, whose llrint
        // overflows int64 (asymmetric INT64_MIN garbage, Newton-3 breaks).
        // Saturate to +/-2^62 instead -- same sticky semantics, exactly
        // representable (E0o hunt 2026-09-10).
        double sx = fx * scale, sy = fy * scale, sz = fz * scale;
        double bnd = vmaxd * 0.25;  // 2^62
        if (!(fabs(sx) <= vmaxd)) {
            atomicAdd(sticky, 1);
            sx = (sx > 0.0) ? bnd : -bnd;
        }
        if (!(fabs(sy) <= vmaxd)) {
            atomicAdd(sticky, 1);
            sy = (sy > 0.0) ? bnd : -bnd;
        }
        if (!(fabs(sz) <= vmaxd)) {
            atomicAdd(sticky, 1);
            sz = (sz > 0.0) ? bnd : -bnd;
        }
        ax += llrint(sx);
        ay += llrint(sy);
        az += llrint(sz);
    }
    Fq[i] = ax;
    Fq[i + 1] = ay;
    Fq[i + 2] = az;
}
"""

_SRC_CACHE: dict[str, object] = {}


def _constants():
    """KE / 2/sqrt(pi) imported from opus (trap class 8: never retype).
    float() conversion required: opus _INV_SQRT_PI is a numpy float64 whose
    numpy-2.x repr is np.float64(...) -- emitted verbatim into CUDA source
    it is an undefined identifier (caught by E0o compile 2026-09-10)."""
    from opus.nonbonded import KE, _INV_SQRT_PI
    return float(KE), float(_INV_SQRT_PI)


def kernel_source() -> str:
    ke, ispi = _constants()
    return (_KERNEL_TMPL % {"ke": repr(ke), "ispi": repr(ispi)}).replace(
        "__ERFC__", emit_cuda())


def _module():
    import cupy as cp
    if "k" not in _SRC_CACHE:
        mod = cp.RawModule(code=kernel_source(), options=("-fmad", "false"))
        _SRC_CACHE["k"] = mod.get_function("direct_q")
    return _SRC_CACHE["k"]


def direct_forces_q(x, q, sig, eps, nlist, ncnt, alpha: float, rc: float,
                    box=None, block: int = 128):
    """Q24.40 direct-space forces (pillar 1 production form).

    Returns (Fq, n_overflow): Fq int64 (N, R, 3) raw accumulator
    (dequantize with Fq.to_f64()-equivalent Fq * 2**-40); n_overflow is
    the per-contribution float-domain sticky count (opus add_to).
    """
    import cupy as cp
    from opus.fxp import Q24_40
    k = _module()
    ke, ispi = _constants()
    N, R, _ = x.shape
    maxnb = nlist.shape[1]
    scale = float(1 << Q24_40["frac_bits"])
    vmax = (1 << 63) - 1  # fxp._limits(24, 40): format range == int64 range
    if box is not None:
        box = np.asarray(box, dtype=np.float64)
        inv_box = np.linalg.inv(box)
        bd = np.linalg.inv(inv_box)  # opus mic transforms via inv(inv_box)
        ld = np.diag(bd)
        ild = np.diag(inv_box)
    xh = np.ascontiguousarray(cp.asnumpy(x) if hasattr(x, "get") else x,
                              dtype=np.float64)
    Fq = cp.zeros(N * R * 3, dtype=cp.int64)
    sticky = cp.zeros(1, dtype=cp.int32)
    args = (cp.asarray(xh), cp.asarray(np.ascontiguousarray(nlist)),
            cp.asarray(ncnt), cp.asarray(np.ascontiguousarray(q, np.float64)),
            cp.asarray(np.ascontiguousarray(sig, np.float64)),
            cp.asarray(np.ascontiguousarray(eps, np.float64)),
            Fq, sticky, N, R, maxnb, 1 if box is not None else 0,
            rc * rc, alpha)
    if box is not None:
        args = args + (float(ld[0]), float(ld[1]), float(ld[2]),
                       float(ild[0]), float(ild[1]), float(ild[2]))
    else:
        args = args + (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    args = args + (scale, float(vmax))
    grid = ((N * R + block - 1) // block,)
    k(grid, (block,), args)
    cp.cuda.Stream.null.synchronize()
    return Fq.reshape(N, R, 3), int(cp.asnumpy(sticky)[0])


# ------------------------------------------------------- kernel mirror

def erfc_poly_bitwise(x: float, emx2: float) -> float:
    """numpy mirror of the erfc_poly device function (kernel t-form:
    t = x*s0 - 1.0, NOT erfc_poly_host's 2*(x-lo)/(hi-lo) -- different
    rounding)."""
    from .erfc_poly import _piece_coeffs
    if x < 1.5:
        i, s = 0, 2.0 / 1.5
        t = x * s - 1.0
    elif x < 3.0:
        i, s = 1, 2.0 / 1.5
        t = (x - 1.5) * s - 1.0
    else:
        i, s = 2, 2.0 / 3.5
        xc = 6.5 if x > 6.5 else x
        t = (xc - 3.0) * s - 1.0
    r = 0.0
    for c in _piece_coeffs(i)[::-1]:
        r = r * t + c
    return emx2 * r


def direct_q_mirror_kernel(x, q, sig, eps, nlist, ncnt, alpha: float,
                           rc: float, box=None):
    """numpy mirror of the direct_q kernel (transcription checkpoint).

    Op-order exact: k2 sr6 grouping, ar*ar, kernel-form erfc poly, rint
    quantization, per-pair float-domain overflow guard.  Pod alignment
    target (bitwise at alpha=0; exp is the only cross-platform ulp risk
    at alpha>0)."""
    from opus.fxp import Q24_40
    ke, ispi = _constants()
    scale = float(1 << Q24_40["frac_bits"])
    vmax = (1 << 63) - 1
    vmaxd = float(vmax)
    x = np.asarray(x, dtype=np.float64)
    N, R, _ = x.shape
    xf = x.reshape(-1)  # flat components (kernel indexing)
    q = np.asarray(q, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    Fq = np.zeros((N, R, 3), dtype=np.int64)
    sticky = 0
    if box is not None:
        box = np.asarray(box, dtype=np.float64)
        inv_box = np.linalg.inv(box)
        bd = np.linalg.inv(inv_box)
        ild = np.diag(inv_box)
        ld = np.diag(bd)
    rc2 = rc * rc
    for a in range(N):
        sa, ea, qa = sig[a], eps[a], q[a]
        for r in range(R):
            base = (a * R + r) * 3
            xi, yi, zi = xf[base], xf[base + 1], xf[base + 2]
            acc = np.zeros(3, dtype=np.int64)
            for k in range(int(ncnt[a])):
                j = int(nlist[a, k])
                jb = (j * R + r) * 3
                dx, dy, dz = (xf[jb] - xi, xf[jb + 1] - yi,
                              xf[jb + 2] - zi)
                if box is not None:
                    ux = dx * ild[0]
                    ux -= np.rint(ux)
                    dx = ux * ld[0]
                    uy = dy * ild[1]
                    uy -= np.rint(uy)
                    dy = uy * ld[1]
                    uz = dz * ild[2]
                    uz -= np.rint(uz)
                    dz = uz * ld[2]
                r2 = (dx * dx + dy * dy) + dz * dz
                inside = 1.0 if r2 < rc2 else 0.0
                r2s = 1e-12 if r2 < 1e-12 else r2
                rr = float(np.sqrt(r2s))
                inv_r = 1.0 / rr
                inv_r2 = inv_r * inv_r
                s = 0.5 * (sa + sig[j])
                sr2 = s * s * inv_r2
                sr6 = sr2 * sr2 * sr2
                eij = float(np.sqrt(ea * eps[j]))
                dulj = -24.0 * eij * (2.0 * sr6 * sr6 - sr6) * inv_r
                ar = alpha * rr
                y = ar * ar
                emx2 = float(np.exp(-y))
                er = erfc_poly_bitwise(ar, emx2)
                dcoul = -(er * inv_r2 + 2.0 * alpha * ispi * emx2 * inv_r)
                duc = ke * (qa * q[j]) * dcoul
                coef = (dulj + duc) * inside * inv_r
                f = np.array([coef * dx, coef * dy, coef * dz])
                scaled = np.rint(f * scale)
                bad = ~np.isfinite(scaled) | (np.abs(scaled) > vmax)
                if bad.any():
                    sticky += int(bad.sum())
                    scaled = np.where(bad, np.sign(scaled) * vmax, scaled)
                acc += scaled.astype(np.int64)
            Fq[a, r] = acc
    return Fq, sticky
