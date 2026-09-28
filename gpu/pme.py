"""PME spread / gather kernels (production, M2).

Semantic source: opus/pme.py (spec section 5.1) -- NOT the E0h/E0h2 bench
kernels, which are cost-representative only (truncation quantization,
closed-form weights, y-major layout, potential-proxy gather).

Bitwise contract (spread): the int64 Q16.48 grid is bitwise-equal to
opus.pme.spread + fxp.FixedPointAccumulator, because
  - weights replicate the opus cardinal-B-spline recursion operation for
    operation (same IEEE double ops; no FMA contraction: compiled with
    -fmad=false),
  - deposition order q*w0*w1*w2 matches numpy left-to-right evaluation,
  - quantization is round-half-even (llrint == np.rint here) on the same
    scaled double (x2^48 is exponent shift, exact),
  - accumulation is exact integer addition (order-free), grid layout is
    opus C-order (R, n1, n2, n3) = replica, x, y, z.
Preconditions: orthorhombic box (diagonal inv_box, off-diagonals exact 0
so einsum frac == x*inv_diag bitwise), n_grid a power of two (mask == mod),
and llrint/np.rint tie cases measure-zero (both round-half-even anyway).

Gather (interp) is dual-gate, not bitwise: opus sums the 4x4x4 adjoint
block with numpy pairwise reduction; the kernel accumulates sequentially
(same 64 products, different association). Expected rel ~ 1e-15; gate
A1 dual (rel < 1e-10 or abs < 1e-8). FFT chain is out of scope here:
this module consumes the potential grid phi (R, n^3) f64 and produces
forces; the cuFFT glue (bitwise per E0e) lands with pillar-2 integration.

Periodic cell assignment (tile): block (bx,by,bz) owns grid points
[bx*tc, bx*tc+tc) per dim (last block short when tc does not divide ng).
Every atom gets one entry per block whose owned range intersects its
stencil hull (bbox of anchors over replicas, seam-split at the ng
boundary), so each stencil point is flushed exactly once -- by its owner
block only (halo accumulates into shared tile, never flushed). Anchors
near the ng seam carry a per-(entry, replica, dim) unwrapping shift in
{-1,0,+1}*ng (host-selected by max overlap of the stencil with the tile
receive window; ties to earlier candidate in (0,+1,-1)) -- without it a
wrapped representative (e.g. anchor 0 vs owner block at the grid end)
is invisible from the tile and the point would be dropped. Envelope:
stencil hull span per dim << ng (replica spread small); a hull spanning
the whole grid is legal but degenerates to all-blocks entries
(registered M2 note).
"""
from __future__ import annotations

import numpy as np

from opus.fxp import Q16_48

# Constant discipline (trap class 8): scale imported from opus.fxp,
# never retyped.
SCALE = 1 << Q16_48["frac_bits"]


def kernel_source(ng: int = 128, tc: int = 12) -> str:
    """CUDA source with ng/tc baked in (both must give ASCII source)."""
    assert ng >= 8 and (ng & (ng - 1)) == 0, "ng must be a power of two"
    assert 4 <= tc <= 16
    ts = tc + 3
    return _KERNEL_TMPL % {"ng": ng, "tc": tc, "ts": ts,
                           "cc": -(-ng // tc),
                           "scale": repr(float(SCALE))}


_KERNEL_TMPL = r"""
#define NG %(ng)d
#define TC %(tc)d
#define TS %(ts)d
#define C %(cc)d

// closed-form M4 weights + M3-difference derivative for interp_gather.
// interp is DUAL-GATED (E0n: rel < 1e-10, not bitwise -- numpy pairwise
// vs sequential sums already differ), so the opus-faithful recursive
// cbs() (2 genuine IEEE /3 divisions per cbs(4) call) is replaced by the
// piecewise cubic form here -- the division chain was measured at ~70 percent of
// interp cost (2026-09-18 attribution).  Anchors: integer ops identical
// to wts4.  Values differ from the recursive form by ~1e-15 rel -- well
// inside the dual gate.  spread_tile keeps the recursive form (bitwise
// locked via v1 == tile == opus).
__device__ __forceinline__ void wts4_cf(double u, int* g, int* gu,
                                        double* w, double* dw) {
    double xg = u * (double)NG;
    int a = (int)floor(xg);
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int gg = a - 3 + t;
        gu[t] = gg;
        g[t] = gg & (NG - 1);
        double xx = xg - (double)gg;   // in [0, 4)
        double wv, m3a, m3b;
        if (xx < 1.0) {
            wv = xx * xx * xx / 6.0;
            m3a = xx * xx / 2.0;
        } else if (xx < 2.0) {
            wv = ((-3.0 * xx * xx * xx + 12.0 * xx * xx) - 12.0 * xx + 4.0)
                 / 6.0;
            m3a = (-2.0 * xx * xx + 6.0 * xx - 3.0) / 2.0;
        } else if (xx < 3.0) {
            wv = ((3.0 * xx * xx * xx - 24.0 * xx * xx) + 60.0 * xx - 44.0)
                 / 6.0;
            double v = 3.0 - xx;
            m3a = v * v / 2.0;
        } else {
            double v = 4.0 - xx;
            wv = v * v * v / 6.0;
            m3a = 0.0;
        }
        double xm = xx - 1.0;
        if (xm <= 0.0) {
            m3b = 0.0;
        } else if (xm < 1.0) {
            m3b = xm * xm / 2.0;
        } else if (xm < 2.0) {
            m3b = (-2.0 * xm * xm + 6.0 * xm - 3.0) / 2.0;
        } else {
            double v = 3.0 - xm;
            m3b = v * v / 2.0;
        }
        w[t] = wv;
        dw[t] = (m3a - m3b) * (double)NG;
    }
}

// opus.pme.cardinal_bspline, op-for-order exact (x/(p-1))*M(x)
// + ((p-x)/(p-1))*M(x-1); -fmad=false keeps the + uncontracted.
__device__ __forceinline__ double cbs(int p, double x) {
    if (p == 1) return (0.0 <= x && x < 1.0) ? 1.0 : 0.0;
    double l = x / (double)(p - 1) * cbs(p - 1, x);
    double r = ((double)p - x) / (double)(p - 1) * cbs(p - 1, x - 1.0);
    return l + r;
}

// derivative: opus.pme.bspline_deriv = M_{p-1}(x) - M_{p-1}(x-1)
__device__ __forceinline__ double cbs_deriv(int p, double x) {
    return cbs(p - 1, x) - cbs(p - 1, x - 1.0);
}

// opus.pme.bspline_weights: xg = u*ng; a = floor(xg); anchors a-3+t;
// w_t = M4(xg - g_t).  g[] wrapped (& (NG-1) == mod for power of two,
// two's complement), gu[] kept unwrapped for tile-local indexing.
__device__ __forceinline__ void wts4(double u, int* g, int* gu, double* w) {
    double xg = u * (double)NG;
    int a = (int)floor(xg);
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int gg = a - 3 + t;
        gu[t] = gg;
        g[t] = gg & (NG - 1);
        w[t] = cbs(4, xg - (double)gg);
    }
}

// Pyramid-shared wts4 for spread_tile: the four cbs(4, .) calls share
// their M3/M2/M1 subevaluations instead of re-expanding the full
// recursion tree per t.  BITWISE == the recursive form: every level uses
// the identical expression text (divide-then-multiply, l+r order), and in
// in this domain (anchors a >= 3: all subtractions reduce magnitude and
// are exact in f64), x_t-1.0 == x_{t+1} exactly and sharing is
// value-identical.  ~8x fewer instructions than the fully-inlined
// recursion tree -- the measured spread cost driver (2026-09-18: division
// recip-mul probe was SLOWER, noinline probe flat; instruction count is
// the mechanism).  spread_v1 keeps the plain recursive form so E0n's
// tile==v1 gate double-checks this pyramid bitwise.
__device__ __forceinline__ void wts4_py(double u, int* g, int* gu,
                                        double* w) {
    double xg = u * (double)NG;
    int a = (int)floor(xg);
    int gg0 = a - 3;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        gu[t] = gg0 + t;
        g[t] = (gg0 + t) & (NG - 1);
    }
    if (gg0 < 0) {
        // seam corner: xg - (negative k) jumps to a COARSER binade and can
        // round, breaking the x_t-1.0 == x_{t+1} identity the sharing
        // relies on (caught bitwise by E0n's seam-stressed atoms).  Fall
        // back to the verbatim recursive form for these atoms.
        #pragma unroll
        for (int t = 0; t < 4; ++t)
            w[t] = cbs(4, xg - (double)(gg0 + t));
        return;
    }
    double x0 = xg - (double)gg0;          // [3, 4)
    double x1 = xg - (double)(gg0 + 1);    // [2, 3)
    double x2 = xg - (double)(gg0 + 2);    // [1, 2)
    double x3 = xg - (double)(gg0 + 3);    // [0, 1)
    double xm1 = x3 - 1.0, xm2 = x3 - 2.0, xm3 = x3 - 3.0;
    // level 1: (0.0 <= y && y < 1.0) ? 1.0 : 0.0   (cbs base text)
    double b0 = (0.0 <= x0 && x0 < 1.0) ? 1.0 : 0.0;
    double b1 = (0.0 <= x1 && x1 < 1.0) ? 1.0 : 0.0;
    double b2 = (0.0 <= x2 && x2 < 1.0) ? 1.0 : 0.0;
    double b3 = (0.0 <= x3 && x3 < 1.0) ? 1.0 : 0.0;
    double bm1 = (0.0 <= xm1 && xm1 < 1.0) ? 1.0 : 0.0;
    double bm2 = (0.0 <= xm2 && xm2 < 1.0) ? 1.0 : 0.0;
    double bm3 = (0.0 <= xm3 && xm3 < 1.0) ? 1.0 : 0.0;
    // level 2: (y / 1.0) * M1(y) + ((2.0 - y) / 1.0) * M1(y - 1.0)
    //          (division by 1.0 is exact identity in IEEE -- written plain)
    double c20 = x0 * b0 + (2.0 - x0) * b1;
    double c21 = x1 * b1 + (2.0 - x1) * b2;
    double c22 = x2 * b2 + (2.0 - x2) * b3;
    double c23 = x3 * b3 + (2.0 - x3) * bm1;
    double c2m1 = xm1 * bm1 + (2.0 - xm1) * bm2;
    double c2m2 = xm2 * bm2 + (2.0 - xm2) * bm3;
    // level 3: (y / 2.0) * M2(y) + ((3.0 - y) / 2.0) * M2(y - 1.0)
    double d30 = (x0 / 2.0) * c20 + ((3.0 - x0) / 2.0) * c21;
    double d31 = (x1 / 2.0) * c21 + ((3.0 - x1) / 2.0) * c22;
    double d32 = (x2 / 2.0) * c22 + ((3.0 - x2) / 2.0) * c23;
    double d33 = (x3 / 2.0) * c23 + ((3.0 - x3) / 2.0) * c2m1;
    double d3m1 = (xm1 / 2.0) * c2m1 + ((3.0 - xm1) / 2.0) * c2m2;
    // level 4: (y / 3.0) * M3(y) + ((4.0 - y) / 3.0) * M3(y - 1.0)
    //          (the /3.0 divisions are the genuine ones -- kept verbatim)
    w[0] = (x0 / 3.0) * d30 + ((4.0 - x0) / 3.0) * d31;
    w[1] = (x1 / 3.0) * d31 + ((4.0 - x1) / 3.0) * d32;
    w[2] = (x2 / 3.0) * d32 + ((4.0 - x2) / 3.0) * d33;
    w[3] = (x3 / 3.0) * d33 + ((4.0 - x3) / 3.0) * d3m1;
}

// frac %% 1.0 (numpy) == exact fmod + sign fix (no rounding either way)
__device__ __forceinline__ double uwrap(double v) {
    double m = fmod(v, 1.0);
    return (m < 0.0) ? m + 1.0 : m;
}

__device__ __forceinline__ void coords_u(
    const double* __restrict__ x, long long i,
    double invlx, double invly, double invlz,
    double* u0, double* u1, double* u2)
{
    *u0 = uwrap(x[i]     * invlx);
    *u1 = uwrap(x[i + 1] * invly);
    *u2 = uwrap(x[i + 2] * invlz);
}

// v1: global-atomic spread (oracle path; bitwise == opus spread)
extern "C" __global__ void spread_v1(
    const double* __restrict__ x, const double* __restrict__ q,
    long long* __restrict__ grid, int N, int R,
    double invlx, double invly, double invlz, double scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a * R;
    if (a >= N) return;
    double u0, u1, u2;
    coords_u(x, ((long long)a * R + r) * 3, invlx, invly, invlz, &u0, &u1, &u2);
    int ax[4], ay[4], az[4], axu[4], ayu[4], azu[4];
    double wx[4], wy[4], wz[4];
    wts4(u0, ax, axu, wx);
    wts4(u1, ay, ayu, wy);
    wts4(u2, az, azu, wz);
    long long* g = grid + (long long)r * NG * NG * NG;
    double qi = q[a];
    for (int dz = 0; dz < 4; ++dz)
        for (int dy = 0; dy < 4; ++dy)
            for (int dx = 0; dx < 4; ++dx) {
                // opus order: val = q*w0*w1*w2 (left-to-right), then one
                // round-half-even quantization, then exact integer add
                double val = ((qi * wx[dx]) * wy[dy]) * wz[dz];
                long long dep = llrint(val * scale);
                atomicAdd(reinterpret_cast<unsigned long long*>(
                              &g[((long long)ax[dx] * NG + ay[dy]) * NG
                                 + az[dz]]),
                          (unsigned long long)dep);
            }
}

// tile: cell-sorted + shared staging; flush only cell-owned points.
// Same weights/quantization as spread_v1 (bitwise oracle v1 == tile).
// shift: (E, R, 3) int8 in {-1,0,+1}: per-entry per-replica per-dim
// unwrapping selected host-side (max overlap of the anchor stencil with
// the tile receive window), so anchors near the ng seam still reach their
// owner block's tile; misaligned replicas are skipped by the guards (their
// points are flushed by their own owner blocks).
extern "C" __global__ void spread_tile(
    const double* __restrict__ xs, const double* __restrict__ qs,
    const signed char* __restrict__ shift,
    const int* __restrict__ cell_start, const int* __restrict__ cell_end,
    const int* __restrict__ cell_origin,
    long long* __restrict__ grid, int ncell_used, int R,
    double invlx, double invly, double invlz, double scale)
{
    __shared__ long long tile[TS * TS * TS];
    int c = blockIdx.x %% ncell_used;
    int r = blockIdx.x / ncell_used;
    for (int i = threadIdx.x; i < TS * TS * TS; i += blockDim.x)
        tile[i] = 0;
    __syncthreads();
    int lo = cell_start[c], hi = cell_end[c];
    int ox = cell_origin[c * 3], oy = cell_origin[c * 3 + 1],
        oz = cell_origin[c * 3 + 2];
    for (int e = lo + threadIdx.x; e < hi; e += blockDim.x) {
        double u0, u1, u2;
        coords_u(xs, ((long long)e * R + r) * 3, invlx, invly, invlz,
                 &u0, &u1, &u2);
        int ax[4], ay[4], az[4], axu[4], ayu[4], azu[4];
        double wx[4], wy[4], wz[4];
        wts4_py(u0, ax, axu, wx);
        wts4_py(u1, ay, ayu, wy);
        wts4_py(u2, az, azu, wz);
        int sx = (int)shift[(e * R + r) * 3] * NG;
        int sy = (int)shift[(e * R + r) * 3 + 1] * NG;
        int sz = (int)shift[(e * R + r) * 3 + 2] * NG;
        double qi = qs[e];
        for (int dz = 0; dz < 4; ++dz) {
            int tz = azu[dz] + sz - oz + 1;
            if (tz < 0 || tz >= TS) continue;
            for (int dy = 0; dy < 4; ++dy) {
                int ty = ayu[dy] + sy - oy + 1;
                if (ty < 0 || ty >= TS) continue;
                for (int dx = 0; dx < 4; ++dx) {
                    int tx = axu[dx] + sx - ox + 1;
                    if (tx < 0 || tx >= TS) continue;
                    double val = ((qi * wx[dx]) * wy[dy]) * wz[dz];
                    long long dep = llrint(val * scale);
                    atomicAdd(reinterpret_cast<unsigned long long*>(
                                  &tile[(tz * TS + ty) * TS + tx]),
                              (unsigned long long)dep);
                }
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TS * TS * TS; i += blockDim.x) {
        long long v = tile[i];
        if (v == 0) continue;
        int tz = i / (TS * TS), rem = i - tz * TS * TS;
        int ty = rem / TS, tx = rem - ty * TS;
        // flush only cell-owned points: t* in [1, TC], i.e. unwrapped
        // [ox, ox+TC).  The LAST block is short when TC does not divide
        // NG: points ox+NG..ox+TC-1 wrap into block 0's territory and
        // MUST NOT be flushed here (caught by E0n alignment 2026-09-10:
        // exact x4/x8 double-flush at the seam; the E0h2 bench avoided
        // it by the >=2-grid-point-from-edge envelope only).
        if (tx < 1 || tx > TC || ty < 1 || ty > TC || tz < 1 || tz > TC)
            continue;
        if (ox + tx - 1 >= NG || oy + ty - 1 >= NG || oz + tz - 1 >= NG)
            continue;
        int gx = (ox + tx - 1) & (NG - 1);
        int gy = (oy + ty - 1) & (NG - 1);
        int gz = (oz + tz - 1) & (NG - 1);
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &grid[(((long long)r * NG + gx) * NG + gy) * NG + gz]),
                  (unsigned long long)v);
    }
}

// adjoint-gradient gather (production interp; NOT the bench potential
// proxy).  opus reciprocal_energy force branch, op-order exact except the
// 64-point sum (numpy pairwise vs sequential here -> dual gate).
//   dE/du_0 = q * sum (dw0*m1*m2)*phi ;  F_0 = -(invL_0 * dE/du_0)
extern "C" __global__ void interp_gather(
    const double* __restrict__ x, const double* __restrict__ q,
    const double* __restrict__ pot, double* __restrict__ F,
    int N, int R, double invlx, double invly, double invlz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a * R;
    if (a >= N) return;
    double u0, u1, u2;
    coords_u(x, ((long long)a * R + r) * 3, invlx, invly, invlz, &u0, &u1, &u2);
    int ax[4], ay[4], az[4], axu[4], ayu[4], azu[4];
    double wx[4], wy[4], wz[4];
    double dwx[4], dwy[4], dwz[4];
    wts4_cf(u0, ax, axu, wx, dwx);
    wts4_cf(u1, ay, ayu, wy, dwy);
    wts4_cf(u2, az, azu, wz, dwz);
    const double* g = pot + (long long)r * NG * NG * NG;
    double qi = q[a];
    double sx = 0.0, sy = 0.0, sz = 0.0;
    // opus product order per force component (left-to-right over dims):
    //   F0: ((dw0*m1)*m2)*phi   F1: ((m0*dw1)*m2)*phi   F2: ((m0*m1)*dw2)*phi
    // sum association differs from numpy pairwise -> dual gate, not bitwise
    for (int dz = 0; dz < 4; ++dz)
        for (int dy = 0; dy < 4; ++dy)
            for (int dx = 0; dx < 4; ++dx) {
                double ph = g[((long long)ax[dx] * NG + ay[dy]) * NG
                              + az[dz]];
                sx += ((dwx[dx] * wy[dy]) * wz[dz]) * ph;
                sy += ((wx[dx] * dwy[dy]) * wz[dz]) * ph;
                sz += ((wx[dx] * wy[dy]) * dwz[dz]) * ph;
            }
    long long i = (long long)a * R + r;
    // opus: dE = q * sum(...); F = -(inv_box @ dE) (diagonal box:
    // exact-zero off-diagonals make the contraction single-term)
    F[i * 3]     = -(invlx * (qi * sx));
    F[i * 3 + 1] = -(invly * (qi * sy));
    F[i * 3 + 2] = -(invlz * (qi * sz));
}

// ============ device cell_sort (bitwise == host _cell_sort_vec) ========
// Port of the host vectorized cell assignment (the production host
// bottleneck: ~1.1 s/step at 60k/R48 -- 4x the GPU step work).  All
// integer ops; the only float section (anchor computation) uses the
// IDENTICAL expressions as the host (fmod + sign fix + *NG + floor).
// Floor division: all operands non-negative -> C '/' == numpy '//'.

// stage A: per-atom anchors, per-dim block runs (seam-split), sorted
// emission == boolean-matrix union order (increasing block id)
extern "C" __global__ void cs_anchors(
    const double* __restrict__ x,
    int* __restrict__ B,      // (3, N, C) padded with -1 (host prefill)
    int* __restrict__ cnt,    // (3, N)
    int N, int R, double invlx, double invly, double invlz)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= N) return;
    double invl[3] = {invlx, invly, invlz};
    for (int d = 0; d < 3; ++d) {
        int amin = NG, amax = -1;
        for (int r = 0; r < R; ++r) {
            double u = uwrap(x[((long long)a * R + r) * 3 + d] * invl[d]);
            int au = (int)floor(u * (double)NG);
            if (au < amin) amin = au;
            if (au > amax) amax = au;
        }
        int klo = amin - 3, khi = amax;
        // runs: main in-grid / head wrap (klo<0) / tail wrap (khi>=NG)
        int plo[3], phi[3], nrun = 0;
        int lo = klo > 0 ? klo : 0, hi = khi < NG - 1 ? khi : NG - 1;
        if (hi >= lo) { plo[nrun] = lo / TC; phi[nrun] = hi / TC; ++nrun; }
        if (klo < 0) {
            plo[nrun] = (klo + NG) / TC; phi[nrun] = (NG - 1) / TC; ++nrun;
        }
        if (khi >= NG) { plo[nrun] = 0; phi[nrun] = (khi - NG) / TC; ++nrun; }
        // emit in increasing first-block order (== boolean-matrix union);
        // runs CAN overlap (wide replica-decorrelated hull crossing the
        // seam: main [0..3] meets head [3..3]) -- the host union dedups,
        // so emit with a last-emitted cursor
        int* Ba = B + ((long long)d * N + a) * C;
        int n_out = 0;
        int last = -1;
        for (int rep = 0; rep < nrun; ++rep) {
            // pick the not-yet-emitted run with the smallest plo
            int bi = -1;
            for (int q2 = 0; q2 < nrun; ++q2) {
                if (plo[q2] < 0) continue;
                if (bi < 0 || plo[q2] < plo[bi]) bi = q2;
            }
            for (int b = plo[bi]; b <= phi[bi]; ++b) {
                if (b > last) { Ba[n_out++] = b; last = b; }
            }
            plo[bi] = -1;
        }
        cnt[d * N + a] = n_out;
    }
}

// stage B: per (atom, dim, slot) shift for every replica -- the
// three-candidate max-overlap rule, candidate order (0, +1, -1), strict >
// keeps the FIRST max on ties (identical to the host enumeration).
extern "C" __global__ void cs_shifts(
    const double* __restrict__ x,
    const int* __restrict__ B, const int* __restrict__ cnt,
    signed char* __restrict__ SX,     // (3, N, C, R)
    int N, int R, double invlx, double invly, double invlz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // (a, d, s) flat
    int ns_tot = N * 3 * C;
    if (idx >= ns_tot) return;
    int s = idx %% C;
    int ad = idx / C;
    int d = ad %% 3;
    int a = ad / 3;
    if (s >= cnt[d * N + a]) return;
    int blk = B[((long long)d * N + a) * C + s];
    int ox = blk * TC;
    int w0 = ox - 1, w1 = ox + TC + 1;
    double invl[3] = {invlx, invly, invlz};
    signed char* out = SX + (((long long)d * N + a) * C + s) * R;
    for (int r = 0; r < R; ++r) {
        double u = uwrap(x[((long long)a * R + r) * 3 + d] * invl[d]);
        int ka = (int)floor(u * (double)NG);
        int lo0 = ka - 3, hi0 = ka;
        int best = -1;
        signed char bs = 0;
        int sv0 = 0, sv1 = 1, svm = -1;
        int ov0 = (hi0 + sv0 * NG < w1 ? hi0 + sv0 * NG : w1)
                - (lo0 + sv0 * NG > w0 ? lo0 + sv0 * NG : w0) + 1;
        int ov1 = (hi0 + sv1 * NG < w1 ? hi0 + sv1 * NG : w1)
                - (lo0 + sv1 * NG > w0 ? lo0 + sv1 * NG : w0) + 1;
        int ovm = (hi0 + svm * NG < w1 ? hi0 + svm * NG : w1)
                - (lo0 + svm * NG > w0 ? lo0 + svm * NG : w0) + 1;
        if (ov0 < 0) ov0 = 0;
        if (ov1 < 0) ov1 = 0;
        if (ovm < 0) ovm = 0;
        if (ov0 > best) { best = ov0; bs = 0; }
        if (ov1 > best) { best = ov1; bs = 1; }
        if (ovm > best) { best = ovm; bs = -1; }
        out[r] = bs;
    }
}

// stage C: ragged cartesian expansion (per-entry enumeration order ==
// itertools.product over the sorted per-dim block lists; key encodes
// (cid, enum_idx) so ANY correct sort yields the unique bitwise order)
extern "C" __global__ void cs_expand(
    const int* __restrict__ B, const int* __restrict__ cnt,
    const signed char* __restrict__ SX,
    const long long* __restrict__ off,   // (N,) exclusive prefix of m_per
    unsigned long long* __restrict__ keys,
    int* __restrict__ atom_out, signed char* __restrict__ shift_out,
    int* __restrict__ org_out, int* __restrict__ cid_out,
    int N, int R, int E)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int lo = 0, hi = N - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (off[mid] <= (long long)e) lo = mid; else hi = mid - 1;
    }
    int a = lo;
    int c0 = cnt[a], c1 = cnt[N + a], c2 = cnt[2 * N + a];
    int w = e - (int)off[a];
    int c12 = c1 * c2;
    int i0 = w / c12;
    int i1 = (w - i0 * c12) / c2;   // == (w mod c12) / c2, non-negative
    int i2 = w - i0 * c12 - i1 * c2;
    int bx = B[(0 * N + a) * C + i0];
    int by = B[(1 * N + a) * C + i1];
    int bz = B[(2 * N + a) * C + i2];
    int cid = ((bx %% C) * C + (by %% C)) * C + (bz %% C);
    keys[e] = ((unsigned long long)(unsigned)cid << 32) | (unsigned)e;
    cid_out[e] = cid;
    atom_out[e] = a;
    org_out[e * 3] = bx * TC;
    org_out[e * 3 + 1] = by * TC;
    org_out[e * 3 + 2] = bz * TC;
    const signed char* s0 = SX + (((0 * N + a) * C + i0) * R);
    const signed char* s1 = SX + (((1 * N + a) * C + i1) * R);
    const signed char* s2 = SX + (((2 * N + a) * C + i2) * R);
    signed char* so = shift_out + (long long)e * R * 3;
    for (int r = 0; r < R; ++r) {
        so[r * 3] = s0[r];
        so[r * 3 + 1] = s1[r];
        so[r * 3 + 2] = s2[r];
    }
}
"""

_SRC_CACHE: dict[tuple, object] = {}


def _module(ng: int, tc: int):
    import cupy as cp
    key = (ng, tc)
    if key not in _SRC_CACHE:
        mod = cp.RawModule(code=kernel_source(ng, tc),
                           options=("-fmad", "false"))
        _SRC_CACHE[key] = (
            mod,
            mod.get_function("spread_v1"),
            mod.get_function("spread_tile"),
            mod.get_function("interp_gather"),
            mod.get_function("cs_anchors"),
            mod.get_function("cs_shifts"),
            mod.get_function("cs_expand"),
        )
    return _SRC_CACHE[key]


def _inv_diag(box) -> np.ndarray:
    """opus-identical inv_box diagonal (np.linalg.inv of the diagonal box).

    The inverse of a diagonal matrix is exactly diagonal in IEEE arithmetic
    (LAPACK triangular solves on exact zeros), so this matches
    opus.pme.PmeGrid.inv_box bitwise and frac == x * inv_diag."""
    b = np.asarray(box, dtype=np.float64)
    assert b.shape == (3, 3), "box must be 3x3"
    assert np.all(b == np.diag(np.diag(b))), \
        "orthorhombic only (triclinic PME is later pillar work)"
    return np.linalg.inv(b)


def _shift_select(ka: np.ndarray, ox: int, ng: int, tc: int) -> np.ndarray:
    """Per-replica unwrapping shift for one (atom, block, dim).

    ka: (R,) anchor floors; picks s in {-1,0,+1} maximizing the overlap of
    the stencil hull [ka-3+s*ng, ka+s*ng] with the tile receive window
    [ox-1, ox+tc+1]; ties resolve to the earlier candidate in (0,+1,-1).
    Deterministic; only affects which block's tile stages the point, never
    the deposited value (integer adds are order-free)."""
    lo, hi = ka - 3, ka
    best = np.zeros(ka.shape, dtype=np.int8)
    best_ov = np.zeros(ka.shape, dtype=np.int64)
    for s in (0, 1, -1):
        ov = (np.minimum(hi + s * ng, ox + tc + 1)
              - np.maximum(lo + s * ng, ox - 1) + 1)
        ov = np.maximum(ov, 0)
        take = ov > best_ov
        best[take] = s
        best_ov[take] = ov[take]
    return best


def cell_sort(x: np.ndarray, box, ng: int, tc: int):
    """Host cell assignment for spread_tile (seam-correct, periodic),
    VECTORIZED (60k atoms: ~45 s/window python loop -> ~ms; the loop form
    remains as cell_sort_reference, CI-compared array-bitwise).

    x: (N, R, 3) f64 cartesian.  Returns (atom_idx (E,), shift (E, R, 3)
    int8, cell_start, cell_end (nb,), cell_origin (nb, 3), ncell_used)
    with entries sorted by wrapped cell id ((bx*C+by)*C+bz, C = ceil(ng/tc)).

    Block set per atom: seam-split of the stencil hull (opus asymmetric
    anchor convention [a-3, a]) into <=3 in-grid runs; the union of run
    block ranges is computed exactly as a boolean matrix over the C block
    columns (increasing order == sorted).  Flush-owner invariant: every
    (atom, replica, stencil point) is flushed exactly once, by its owner
    block (shifts align each replica's anchors with the owner's tile;
    misaligned replicas are skipped by the tile guards -- their owner
    blocks hold their own entries).
    """
    return _cell_sort_vec(x, box, ng, tc)


def cell_sort_reference(x: np.ndarray, box, ng: int, tc: int):
    """Scalar loop form: CI reference (array-bitwise equivalence) and the
    degenerate-input fallback."""
    x = np.asarray(x, dtype=np.float64)
    inv = _inv_diag(box)
    inv_d = np.diag(inv)
    gx = x * inv_d  # (N, R, 3); == opus frac for diagonal box (exact zeros)
    C = -(-ng // tc)
    # anchors in GRID units, computed exactly as the kernel does:
    # u = fmod-wrap(frac); a = floor(u * ng)
    u = np.fmod(gx, 1.0)
    u[u < 0] += 1.0
    au = np.floor(u * ng).astype(np.int64)  # (N, R, 3)
    # opus anchor convention: stencil = [a-3, a] per replica (ASYMMETRIC;
    # the E0h2 bench's [k-1, k+2] padding does NOT apply here)
    kmin = au.min(axis=1) - 3  # (N, 3)
    kmax = au.max(axis=1)

    def dim_parts(d: int):
        """(plo, phi, ok-mask) runs of the in-grid hull, seam-split."""
        klo, khi = kmin[:, d], kmax[:, d]
        lo = np.maximum(klo, 0)
        hi = np.minimum(khi, ng - 1)
        ok = hi >= lo
        parts = [(np.where(ok, lo // tc, -1), np.where(ok, hi // tc, -1), ok)]
        m_head = klo < 0
        if m_head.any():
            parts.append((np.where(m_head, (klo + ng) // tc, -1),
                          np.where(m_head, (ng - 1) // tc, -1), m_head))
        m_tail = khi >= ng
        if m_tail.any():
            parts.append((np.where(m_tail, 0, -1),
                          np.where(m_tail, (khi - ng) // tc, -1), m_tail))
        return parts

    parts_xyz = [dim_parts(d) for d in range(3)]
    from itertools import product
    n_rep = x.shape[1]
    entries = []  # (cid, ox, oy, oz, atom, sx(R,), sy(R,), sz(R,))
    for a in range(x.shape[0]):
        dim_opts = []
        for d in range(3):
            ids = set()
            for (plo, phi, ok) in parts_xyz[d]:
                if ok[a]:
                    ids.update(range(int(plo[a]), int(phi[a]) + 1))
            dim_opts.append([])
            for bx in sorted(ids):
                sh = _shift_select(au[a, :, d], bx * tc, ng, tc)
                dim_opts[-1].append((bx, sh))
        for (bx, sx), (by, sy), (bz, sz) in product(*dim_opts):
            cid = ((bx % C) * C + (by % C)) * C + (bz % C)
            entries.append((cid, bx * tc, by * tc, bz * tc, a, sx, sy, sz))
    if not entries:
        e = np.zeros(0, dtype=np.int64)
        return (e, np.zeros((0, n_rep, 3), np.int8),
                np.zeros(0, np.int32), np.zeros(0, np.int32),
                np.zeros((0, 3), np.int32), 0)
    entries.sort(key=lambda t: t[0])  # stable: insertion order within cell
    atom_e = np.array([t[4] for t in entries], dtype=np.int64)
    shift = np.stack([
        np.stack([np.asarray(t[5 + d], dtype=np.int8) for t in entries], 0)
        for d in range(3)], axis=-1)  # (E, R, 3)
    cids = np.array([t[0] for t in entries], dtype=np.int64)
    used_ids = np.unique(cids)
    cs = np.searchsorted(cids, used_ids, "left").astype(np.int32)
    ce = np.searchsorted(cids, used_ids, "right").astype(np.int32)
    origin = np.array([[t[1], t[2], t[3]] for t in entries], dtype=np.int32)[cs]
    return atom_e, shift, cs, ce, origin, int(len(used_ids))


def _cell_sort_vec(x: np.ndarray, box, ng: int, tc: int):
    """Vectorized cell assignment -- array-bitwise identical to
    cell_sort_reference (same per-atom enumeration order, same stable
    cid sort)."""
    x = np.asarray(x, dtype=np.float64)
    inv = _inv_diag(box)
    inv_d = np.diag(inv)
    gx = x * inv_d
    C = -(-ng // tc)
    n, n_rep, _ = x.shape
    u = np.fmod(gx, 1.0)
    u[u < 0] += 1.0
    au = np.floor(u * ng).astype(np.int64)  # (n, R, 3)
    kmin = au.min(axis=1) - 3
    kmax = au.max(axis=1)

    cols = np.arange(C)
    B = np.full((3, n, C), -1, dtype=np.int64)   # padded sorted block ids
    cnt = np.zeros((3, n), dtype=np.int64)
    for d in range(3):
        klo, khi = kmin[:, d], kmax[:, d]
        lo = np.maximum(klo, 0)
        hi = np.minimum(khi, ng - 1)
        ok = hi >= lo
        M = np.zeros((n, C), dtype=bool)
        for (plo, phi, m) in (
                (np.where(ok, lo // tc, -1), np.where(ok, hi // tc, -1), ok),
                (np.where(klo < 0, (klo + ng) // tc, C), np.where(
                    klo < 0, (ng - 1) // tc, -1), klo < 0),
                (np.where(khi >= ng, 0, -1), np.where(
                    khi >= ng, (khi - ng) // tc, -1), khi >= ng)):
            # empty runs are encoded as plo > phi (C sentinel / -1 lower)
            M |= m[:, None] & (cols[None, :] >= plo[:, None]) \
                & (cols[None, :] <= phi[:, None])
        cnt[d] = M.sum(axis=1)
        rank = np.cumsum(M, axis=1)          # 1-based rank of each True
        ai, bi = np.nonzero(M)
        B[d][ai, rank[ai, bi] - 1] = bi       # compact, increasing order

    # per-dim shift selection, COMPACTED over the actual (atom, slot)
    # pairs (~1.3 blocks/atom vs the C=16 padded span; 10x less work).
    # Identical first-max tie-break as cell_sort_reference; the values
    # depend only on (ka, ox) so the padded table is rebuilt by scatter.
    SX = []
    for d in range(3):
        bd_atom, bd_slot = np.nonzero(B[d] >= 0)
        bd_blk = B[d][bd_atom, bd_slot]
        ka_c = au[bd_atom, :, d].astype(np.int32)          # (K, R)
        ox_c = (bd_blk * tc).astype(np.int32)              # (K,)
        lo0 = ka_c - 3
        hi0 = ka_c
        w1 = (ox_c + tc + 1)[:, None]                      # (K, 1)
        w0 = (ox_c - 1)[:, None]
        best = np.full((len(bd_atom), n_rep), -1, dtype=np.int32)
        best_idx = np.zeros((len(bd_atom), n_rep), dtype=np.int8)
        for cand, sv in enumerate((0, 1, -1)):
            ov = np.minimum(hi0 + sv * ng, w1) - np.maximum(
                lo0 + sv * ng, w0) + 1
            np.maximum(ov, 0, out=ov)
            take = ov > best
            best = np.where(take, ov, best)
            best_idx = np.where(take, np.int8(sv), best_idx)
        sx_padded = np.zeros((n, C, n_rep), dtype=np.int8)
        sx_padded[bd_atom, bd_slot, :] = best_idx
        SX.append(sx_padded)
    SX = tuple(SX)

    # ragged cartesian product across dims (per-atom enumeration order ==
    # itertools.product over the sorted per-dim block lists)
    m_per = cnt[0] * cnt[1] * cnt[2]
    E = int(m_per.sum())
    ea = np.repeat(np.arange(n), m_per)
    off = np.concatenate([[0], np.cumsum(m_per)[:-1]])
    w = np.arange(E) - np.repeat(off, m_per)
    c12 = np.repeat(cnt[1] * cnt[2], m_per)
    c2 = np.repeat(cnt[2], m_per)
    i0 = w // c12
    i1 = (w % c12) // c2
    i2 = w % c2
    bx = B[0][ea, i0]
    by = B[1][ea, i1]
    bz = B[2][ea, i2]
    sx = SX[0][ea, i0]                   # (E, R)
    sy = SX[1][ea, i1]
    sz = SX[2][ea, i2]
    ox = bx * tc
    oy = by * tc
    oz = bz * tc
    cid = ((bx % C) * C + (by % C)) * C + (bz % C)
    order = np.argsort(cid, kind="stable")
    atom_e = ea[order]
    shift = np.stack([sx[order], sy[order], sz[order]], axis=-1) \
        .astype(np.int8)                 # (E, R, 3)
    cids = cid[order]
    used_ids = np.unique(cids)
    cs = np.searchsorted(cids, used_ids, "left").astype(np.int32)
    ce = np.searchsorted(cids, used_ids, "right").astype(np.int32)
    origin = np.stack([ox[order], oy[order], oz[order]],
                      axis=1).astype(np.int32)[cs]
    return atom_e, shift, cs, ce, origin, int(len(used_ids))


def cell_sort_device(x, box, ng: int, tc: int):
    """Device cell assignment, array-bitwise == _cell_sort_vec (CI-pinned).

    x: (N, R, 3) f64 DEVICE array.  Returns device arrays
    (atom_idx (E,) int32, shift (E, R, 3) int8, cell_start/cell_end (nb,)
    int32, cell_origin (nb, 3) int32, ncell_used) -- same values as the
    host forms.  Sort keys are unique (cid, enum_idx) u64 pairs, so the
    ordering is bitwise-deterministic regardless of the sort algorithm.
    """
    import cupy as cp
    fns = _module(ng, tc)
    k_a, k_s, k_e = fns[4], fns[5], fns[6]
    N, R, _ = x.shape
    C = -(-ng // tc)
    inv = _inv_diag(box)
    B = cp.full((3, N, C), -1, dtype=cp.int32)
    cnt = cp.zeros((3, N), dtype=cp.int32)
    k_a(((N + 255) // 256,), (256,),
        (x, B, cnt, N, R, float(inv[0, 0]), float(inv[1, 1]),
         float(inv[2, 2])))
    SX = cp.zeros((3, N, C, R), dtype=cp.int8)
    nst = N * 3 * C
    k_s(((nst + 255) // 256,), (256,),
        (x, B, cnt, SX, N, R, float(inv[0, 0]), float(inv[1, 1]),
         float(inv[2, 2])))
    m_per = cnt[0].astype(cp.int64) * cnt[1] * cnt[2]
    off_incl = cp.cumsum(m_per)
    E = int(off_incl[-1].get())          # one scalar sync (10-20 us)
    off = off_incl - m_per
    keys = cp.empty(E, dtype=cp.uint64)
    atom_pre = cp.empty(E, dtype=cp.int32)
    shift_pre = cp.empty((E, R, 3), dtype=cp.int8)
    org_pre = cp.empty((E, 3), dtype=cp.int32)
    cid_pre = cp.empty(E, dtype=cp.int32)
    k_e(((E + 255) // 256,), (256,),
        (B, cnt, SX, off, keys, atom_pre, shift_pre.reshape(-1),
         org_pre.reshape(-1), cid_pre, N, R, E))
    order = cp.argsort(keys)             # unique keys -> unique answer
    atom_e = atom_pre[order]
    shift = shift_pre[order]
    org_sorted = org_pre[order]
    cid_sorted = cid_pre[order]
    used = cp.unique(cid_sorted)
    cs = cp.searchsorted(cid_sorted, used).astype(cp.int32)
    ce = cp.searchsorted(cid_sorted, used, side="right").astype(cp.int32)
    origin = org_sorted[cs.astype(cp.int64)]
    return atom_e, shift, cs, ce, origin, int(used.size)


def spread(x, q, box, *, ng: int = 128, tc: int = 12, mode: str = "tile",
           block: int = 128):
    """Q16.48 charge grid (R, ng**3) int64, bitwise == opus.pme.spread.

    mode='v1' is the global-atomic oracle path (same bitwise result,
    slower); mode='tile' is the production path (E0h2 shape, 2.78x)."""
    import cupy as cp
    assert mode in ("v1", "tile")
    mod, k_v1, k_tile, _k_interp, k_csa, k_css, k_cse = _module(ng, tc)
    N, R = x.shape[0], x.shape[1]
    inv = _inv_diag(box)
    grid = cp.zeros(R * ng ** 3, dtype=cp.int64)
    # 设备输入走设备路径,避免 host 往返;宿主输入才下载一次
    on_dev = hasattr(x, "get")
    if on_dev:
        d_x = x
        d_q = q if hasattr(q, "get") else cp.asarray(
            np.ascontiguousarray(q, dtype=np.float64))
    else:
        xh = np.ascontiguousarray(x, dtype=np.float64)
        qh = np.ascontiguousarray(q, dtype=np.float64)
        d_x, d_q = cp.asarray(xh), cp.asarray(qh)
    if mode == "v1":
        k_v1(((N * R + 255) // 256,), (256,),
             (d_x, d_q, grid, N, R, float(inv[0, 0]), float(inv[1, 1]),
              float(inv[2, 2]), float(SCALE)))
    else:
        if hasattr(x, "get"):
            # device input: cell assignment on device (no host roundtrip;
            # array-bitwise == host cell_sort, CI-pinned)
            atom_e, shift, cs, ce, org, ncell = cell_sort_device(
                x, box, ng, tc)
            aidx = atom_e.astype(cp.int64)
            xs_s = d_x[aidx]
            qs_s = d_q[aidx]
            k_tile((max(ncell, 1) * R,), (block,),
                   (xs_s, qs_s, shift, cs, ce,
                    org.reshape(-1),
                    grid, np.int32(ncell), np.int32(R),
                    float(inv[0, 0]), float(inv[1, 1]), float(inv[2, 2]),
                    float(SCALE)))
        else:
            atom_e, shift, cs, ce, org, ncell = cell_sort(xh, box, ng, tc)
            k_tile((max(ncell, 1) * R,), (block,),
                   (cp.asarray(np.ascontiguousarray(xh[atom_e])),
                    cp.asarray(np.ascontiguousarray(qh[atom_e])),
                    cp.asarray(shift),
                    cp.asarray(cs), cp.asarray(ce),
                    cp.asarray(org.reshape(-1)),
                    grid, np.int32(ncell), np.int32(R),
                    float(inv[0, 0]), float(inv[1, 1]), float(inv[2, 2]),
                    float(SCALE)))
    cp.cuda.Stream.null.synchronize()
    return grid.reshape(R, ng, ng, ng)


def interp_forces(x, q, pot, box, *, ng: int = 128):
    """Adjoint-gradient reciprocal forces (N, R, 3) f64.

    pot: (R, ng**3) f64 potential grid = N_grid * irfftn-chain phi
    (production: cuFFT chain; alignment: numpy chain).  Dual gate vs opus
    (sequential vs pairwise 64-point sum); sign/gradient semantics exact."""
    import cupy as cp
    _, _, _, k_interp, _csa, _css, _cse = _module(ng, 12)
    dev = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float64)) \
        if not (hasattr(a, "get")) else cp.ascontiguousarray(a, np.float64)
    x = dev(x)
    q = dev(q)
    pot = dev(pot)
    N, R, _ = x.shape
    inv = _inv_diag(box)
    F = cp.zeros(N * R * 3)
    k_interp(((N * R + 255) // 256,), (256,),
             (x, q, pot, F, N, R, float(inv[0, 0]), float(inv[1, 1]),
              float(inv[2, 2])))
    cp.cuda.Stream.null.synchronize()
    return F.reshape(N, R, 3)
