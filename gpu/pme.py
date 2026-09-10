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
                           "scale": repr(float(SCALE))}


_KERNEL_TMPL = r"""
#define NG %(ng)d
#define TC %(tc)d
#define TS %(ts)d

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
        wts4(u0, ax, axu, wx);
        wts4(u1, ay, ayu, wy);
        wts4(u2, az, azu, wz);
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
    wts4(u0, ax, axu, wx);
    wts4(u1, ay, ayu, wy);
    wts4(u2, az, azu, wz);
    double dwx[4], dwy[4], dwz[4];
    {
        double xg = u0 * (double)NG;
        int a0 = (int)floor(xg);
        for (int t = 0; t < 4; ++t)
            dwx[t] = cbs_deriv(4, xg - (double)(a0 - 3 + t)) * (double)NG;
    }
    {
        double xg = u1 * (double)NG;
        int a0 = (int)floor(xg);
        for (int t = 0; t < 4; ++t)
            dwy[t] = cbs_deriv(4, xg - (double)(a0 - 3 + t)) * (double)NG;
    }
    {
        double xg = u2 * (double)NG;
        int a0 = (int)floor(xg);
        for (int t = 0; t < 4; ++t)
            dwz[t] = cbs_deriv(4, xg - (double)(a0 - 3 + t)) * (double)NG;
    }
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
    """Host cell assignment for spread_tile (seam-correct, periodic).

    x: (N, R, 3) f64 cartesian.  Returns (atom_idx (E,), shift (E, R, 3)
    int8, cell_start, cell_end (nb,), cell_origin (nb, 3), ncell_used)
    with entries sorted by wrapped cell id ((bx*C+by)*C+bz, C = ceil(ng/tc)).

    Block set per atom: seam-split of the stencil hull
    [floor(min_r u)-1, floor(max_r u)+2] into <=3 in-grid runs; blocks are
    the tc-blocks of each run.  Flush-owner invariant: every (atom,
    replica, stencil point) is flushed exactly once, by its owner block
    (shifts align each replica's anchors with the owner's tile; points of
    misaligned replicas within this tile are halo/foreign and are skipped
    by the guards -- their owner blocks hold their own entries).
    """
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


def spread(x, q, box, *, ng: int = 128, tc: int = 12, mode: str = "tile",
           block: int = 128):
    """Q16.48 charge grid (R, ng**3) int64, bitwise == opus.pme.spread.

    mode='v1' is the global-atomic oracle path (same bitwise result,
    slower); mode='tile' is the production path (E0h2 shape, 2.78x)."""
    import cupy as cp
    assert mode in ("v1", "tile")
    mod, k_v1, k_tile, _ = _module(ng, tc)
    xh = np.ascontiguousarray(cp.asnumpy(x) if hasattr(x, "get") else x,
                              dtype=np.float64)
    qh = np.ascontiguousarray(cp.asnumpy(q) if hasattr(q, "get") else q,
                              dtype=np.float64)
    N, R, _ = xh.shape
    inv = _inv_diag(box)
    grid = cp.zeros(R * ng ** 3, dtype=cp.int64)
    d_x, d_q = cp.asarray(xh), cp.asarray(qh)
    if mode == "v1":
        k_v1(((N * R + 255) // 256,), (256,),
             (d_x, d_q, grid, N, R, float(inv[0, 0]), float(inv[1, 1]),
              float(inv[2, 2]), float(SCALE)))
    else:
        atom_e, shift, cs, ce, org, ncell = cell_sort(xh, box, ng, tc)
        k_tile((max(ncell, 1) * R,), (block,),
               (cp.asarray(np.ascontiguousarray(xh[atom_e])),
                cp.asarray(np.ascontiguousarray(qh[atom_e])),
                cp.asarray(shift),
                cp.asarray(cs), cp.asarray(ce), cp.asarray(org.reshape(-1)),
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
    _, _, _, k_interp = _module(ng, 12)
    x = cp.asarray(np.ascontiguousarray(x, dtype=np.float64))
    q = cp.asarray(np.ascontiguousarray(q, dtype=np.float64))
    pot = cp.asarray(np.ascontiguousarray(pot, dtype=np.float64))
    N, R, _ = x.shape
    inv = _inv_diag(box)
    F = cp.zeros(N * R * 3)
    k_interp(((N * R + 255) // 256,), (256,),
             (x, q, pot, F, N, R, float(inv[0, 0]), float(inv[1, 1]),
              float(inv[2, 2])))
    cp.cuda.Stream.null.synchronize()
    return F.reshape(N, R, 3)
