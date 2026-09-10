"""Constraint kernels: rigid-water SHAKE + RATTLE velocity projection.

Semantic source: opus/dynamics.py shake_positions / project_velocities --
NOT the E0j bench shake_rigid, whose correction (d2-d^2)/(d2*denom) is the
classic 2x-overshoot variant: same fixed point, different iteration path
(a first-round transcription-trap candidate caught during collection).

Bitwise contract vs opus (given -fmad=false, IEEE doubles):
  - r2 sum of 3 products: numpy sums sequentially for n < 8 (pairwise_sum
    small-n path) == left-associated dx*dx+dy*dy+dz*dz;
  - clamps replicate np.where guards exactly (1e-24 on r2 before sqrt;
    on denom*r / denom*r2 products before divide);
  - update order (invm_i*corr)*dx per component; constraint order OH1, OH2,
    HH per water == the water-major constraint list order in opus (waters
    are disjoint, so global list iteration factors per water).

Fixed-iteration discipline (Q-016): iteration count is a caller manifest
constant (opus Dynamics default 12); no convergence branch anywhere.
Water layout: atoms (O,H,H) contiguous per water, replica-inner
x[((3w+r)*3+d)] -- same as the benches and the direct kernel's (N,R,3).
"""
from __future__ import annotations

_KERNEL_TMPL = r"""
// one thread per (water, replica); 3 constraints x iters, serial in-thread
extern "C" __global__ void shake_rigid_water(
    double* __restrict__ x, const double* __restrict__ invm,
    int nw, int R, int iters, double doh, double dhh)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nw * R) return;
    int w = idx / R, r = idx - w * R;
    long long o  = ((long long)(3 * w) * R + r) * 3;
    long long h1 = o + 3 * R, h2 = o + 6 * R;
    double imo = invm[3 * w], imh = invm[3 * w + 1];
    for (int it = 0; it < iters; ++it) {
        // OH1: i=o, j=h1
        {
            double dx = x[h1] - x[o], dy = x[h1 + 1] - x[o + 1],
                   dz = x[h1 + 2] - x[o + 2];
            double r2 = (dx * dx + dy * dy) + dz * dz;
            double rr = sqrt(r2 < 1e-24 ? 1e-24 : r2);
            double dr = (imo + imh) * rr;
            double corr = (rr - doh) / (dr < 1e-24 ? 1e-24 : dr);
            double co = imo * corr, ch = imh * corr;
            x[o]     += co * dx; x[o + 1] += co * dy; x[o + 2] += co * dz;
            x[h1]    -= ch * dx; x[h1 + 1] -= ch * dy; x[h1 + 2] -= ch * dz;
        }
        // OH2: i=o, j=h2
        {
            double dx = x[h2] - x[o], dy = x[h2 + 1] - x[o + 1],
                   dz = x[h2 + 2] - x[o + 2];
            double r2 = (dx * dx + dy * dy) + dz * dz;
            double rr = sqrt(r2 < 1e-24 ? 1e-24 : r2);
            double dr = (imo + imh) * rr;
            double corr = (rr - doh) / (dr < 1e-24 ? 1e-24 : dr);
            double co = imo * corr, ch = imh * corr;
            x[o]     += co * dx; x[o + 1] += co * dy; x[o + 2] += co * dz;
            x[h2]    -= ch * dx; x[h2 + 1] -= ch * dy; x[h2 + 2] -= ch * dz;
        }
        // HH: i=h1, j=h2
        {
            double dx = x[h2] - x[h1], dy = x[h2 + 1] - x[h1 + 1],
                   dz = x[h2 + 2] - x[h1 + 2];
            double r2 = (dx * dx + dy * dy) + dz * dz;
            double rr = sqrt(r2 < 1e-24 ? 1e-24 : r2);
            double dr = (imh + imh) * rr;
            double corr = (rr - dhh) / (dr < 1e-24 ? 1e-24 : dr);
            double c1 = imh * corr, c2 = imh * corr;
            x[h1]    += c1 * dx; x[h1 + 1] += c1 * dy; x[h1 + 2] += c1 * dz;
            x[h2]    -= c2 * dx; x[h2 + 1] -= c2 * dy; x[h2 + 2] -= c2 * dz;
        }
    }
}

// RATTLE radial velocity projection (opus project_velocities), one thread
// per (water, replica); constraint order and clamps as in shake above.
extern "C" __global__ void project_rigid_water(
    double* __restrict__ v, const double* __restrict__ x,
    const double* __restrict__ invm,
    int nw, int R, int iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nw * R) return;
    int w = idx / R, r = idx - w * R;
    long long o  = ((long long)(3 * w) * R + r) * 3;
    long long h1 = o + 3 * R, h2 = o + 6 * R;
    double imo = invm[3 * w], imh = invm[3 * w + 1];
    for (int it = 0; it < iters; ++it) {
        // OH1
        {
            double dx = x[h1] - x[o], dy = x[h1 + 1] - x[o + 1],
                   dz = x[h1 + 2] - x[o + 2];
            double r2 = (dx * dx + dy * dy) + dz * dz;
            double dvx = v[h1] - v[o], dvy = v[h1 + 1] - v[o + 1],
                   dvz = v[h1 + 2] - v[o + 2];
            double dvr = (dvx * dx + dvy * dy) + dvz * dz;
            double pr = (imo + imh) * r2;
            double corr = dvr / (pr < 1e-24 ? 1e-24 : pr);
            double co = imo * corr, ch = imh * corr;
            v[o]     += co * dx; v[o + 1] += co * dy; v[o + 2] += co * dz;
            v[h1]    -= ch * dx; v[h1 + 1] -= ch * dy; v[h1 + 2] -= ch * dz;
        }
        // OH2
        {
            double dx = x[h2] - x[o], dy = x[h2 + 1] - x[o + 1],
                   dz = x[h2 + 2] - x[o + 2];
            double r2 = (dx * dx + dy * dy) + dz * dz;
            double dvx = v[h2] - v[o], dvy = v[h2 + 1] - v[o + 1],
                   dvz = v[h2 + 2] - v[o + 2];
            double dvr = (dvx * dx + dvy * dy) + dvz * dz;
            double pr = (imo + imh) * r2;
            double corr = dvr / (pr < 1e-24 ? 1e-24 : pr);
            double co = imo * corr, ch = imh * corr;
            v[o]     += co * dx; v[o + 1] += co * dy; v[o + 2] += co * dz;
            v[h2]    -= ch * dx; v[h2 + 1] -= ch * dy; v[h2 + 2] -= ch * dz;
        }
        // HH
        {
            double dx = x[h2] - x[h1], dy = x[h2 + 1] - x[h1 + 1],
                   dz = x[h2 + 2] - x[h1 + 2];
            double r2 = (dx * dx + dy * dy) + dz * dz;
            double dvx = v[h2] - v[h1], dvy = v[h2 + 1] - v[h1 + 1],
                   dvz = v[h2 + 2] - v[h1 + 2];
            double dvr = (dvx * dx + dvy * dy) + dvz * dz;
            double pr = (imh + imh) * r2;
            double corr = dvr / (pr < 1e-24 ? 1e-24 : pr);
            double c1 = imh * corr, c2 = imh * corr;
            v[h1]    += c1 * dx; v[h1 + 1] += c1 * dy; v[h1 + 2] += c1 * dz;
            v[h2]    -= c2 * dx; v[h2 + 1] -= c2 * dy; v[h2 + 2] -= c2 * dz;
        }
    }
}
"""

_SRC_CACHE: dict[str, object] = {}


def _kernels():
    import cupy as cp
    if "k" not in _SRC_CACHE:
        mod = cp.RawModule(code=_KERNEL_TMPL, options=("-fmad", "false"))
        _SRC_CACHE["k"] = (mod.get_function("shake_rigid_water"),
                           mod.get_function("project_rigid_water"))
    return _SRC_CACHE["k"]


def shake_water(x, invm, *, r_oh: float, r_hh: float, iters: int = 12):
    """In-place fixed-iteration SHAKE on rigid waters (x cupy (N*R*3,))."""
    import cupy as cp
    k_shake, _ = _kernels()
    nw = invm.shape[0] // 3
    R = x.shape[0] // (nw * 9)
    k_shake(((nw * R + 255) // 256,), (256,),
            (x, cp.asarray(invm), nw, R, iters, float(r_oh), float(r_hh)))
    cp.cuda.Stream.null.synchronize()


def project_water(v, x, invm, *, iters: int = 12):
    """In-place radial velocity projection (v cupy (N*R*3,))."""
    import cupy as cp
    _, k_proj = _kernels()
    nw = invm.shape[0] // 3
    R = v.shape[0] // (nw * 9)
    k_proj(((nw * R + 255) // 256,), (256,), (v, x, cp.asarray(invm),
                                              nw, R, iters))
    cp.cuda.Stream.null.synchronize()
