"""BAOAB streaming integrator kernels (production, M2).

Semantic source: opus/dynamics.py Dynamics.step_baoab -- NOT the E0j/E0l
bench integrate kernels (philox-lite hash, same-f-twice kick structure:
cost-representative only).

Streaming decomposition (bitwise == opus VRORV, one force eval per step):
opus step n (forces f1 = F(x_n) at entry, f2 = F(x_{n+1}) at exit)
    v  += s*f1/m            s = 0.5*dt            (kick)
    x  += s*v                                     (half drift)
    O: v = c*v + ns*g       (gamma > 0 only)
    x  += s*v                                     (half drift)
    SHAKE(x)                                      (fixed iters, Q-016)
    v  += s*f2/m                                  (kick)
    project(v)                                    (RATTLE)
Since f1 of step n+1 == f2 of step n, one held force serves
    [kick(f) -> project -> kick(f) -> drift -> O -> drift -> SHAKE]
per invocation, with first=True skipping the leading kick+project (chain
start) and baoab_finish closing the tail of the last step.  The caller
recomputes f = F(x) between invocations (production loop; graph capture
later -- E0l's device-side step_inc is the graph variant of `step`).

Bitwise contract vs opus (given -fmad=false):
  - kicks DIVIDE by mass (opus f1/m); noise DIVIDES inside the sqrt
    (opus kT*(1-c*c)/m) -- reciprocal-multiply is the transcription
    defect this module exists to prevent;
  - drifts have no mass factor; c = exp(-gamma*dt) per-step scalar,
    ns = sqrt((kT*(1-c*c))/m) per atom;
  - numpy small-n sums are sequential == left-associated expressions here.
Layout: x/v/f flat (N*R*3,), component idx -> atom idx // (3R).

RNG (pillar 3 PENDING): the O-step gaussian is the E0j cost-representative
philox4x32-7 + Box-Muller, keyed (seed, step, idx, dof).  It is NOT the
opus.rng.gauss_stream stream (numpy Philox4x64-10 + ziggurat); dynamics
alignment runs gamma=0.  Replacing the device function with the
opus-equivalent stream is its own work item -- do not align gamma > 0
dynamics until it lands.
"""
from __future__ import annotations

import numpy as np

_KERNEL_TMPL = r"""
#define PHILOX_M4x32_0 0xD2511F53u
#define PHILOX_M4x32_1 0xCD9E8D57u
#define PHILOX_W32_0   0x9E3779B9u
#define PHILOX_W32_1   0xBB67AE85u

// PILLAR 3 PENDING: cost-representative RNG (E0j), NOT opus.rng.gauss_stream
__device__ __forceinline__ uint4 philox4x32(uint4 c, uint2 k) {
    for (int r = 0; r < 7; ++r) {
        unsigned hi, lo;
        lo = PHILOX_M4x32_0 * c.x;
        hi = __umulhi(PHILOX_M4x32_0, c.x);
        unsigned t0 = lo ^ c.z ^ k.x;
        unsigned t1 = hi ^ c.w ^ k.y;
        lo = PHILOX_M4x32_1 * c.y;
        hi = __umulhi(PHILOX_M4x32_1, c.y);
        c.z = lo ^ c.w ^ k.y;
        c.w = hi ^ c.x ^ k.x;
        c.x = t0; c.y = t1;
        k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    }
    return c;
}

__device__ __forceinline__ double gauss_pair(unsigned u1, unsigned u2,
                                             double* second) {
    double a = (u1 + 1.0) * 2.3283064365386963e-10;
    double b = (u2 + 1.0) * 2.3283064365386963e-10;
    double r = sqrt(-2.0 * log(a));
    *second = r * sin(6.283185307179586 * b);
    return r * cos(6.283185307179586 * b);
}

// v = v + (s*f)/m per component (opus kick; division, not invm multiply)
extern "C" __global__ void half_kick(
    double* __restrict__ v, const double* __restrict__ f,
    const double* __restrict__ m, int n_tot, int at_stride, double s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tot) return;
    v[idx] = v[idx] + (s * f[idx]) / m[idx / at_stride];
}

// x = x + s*v;  O: v = c*v + ns*g (uniform gamma>0 branch);  x = x + s*v
// one thread per (atom, replica)
extern "C" __global__ void drift_orn(
    double* __restrict__ x, double* __restrict__ v,
    const double* __restrict__ m, int nr, int R,
    double s, double gamma, double dt, double kT,
    unsigned long long step, unsigned seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr) return;
    int a = idx / R;
    double c = exp(-gamma * dt);
    double ns = sqrt((kT * (1.0 - c * c)) / m[a]);
    long long base = (long long)idx * 3;
    for (int d = 0; d < 3; ++d) {
        x[base + d] = x[base + d] + s * v[base + d];
    }
    if (gamma > 0.0) {
        for (int d = 0; d < 3; ++d) {
            uint4 ctr = make_uint4((unsigned)step, (unsigned)(step >> 32),
                                   (unsigned)idx, (unsigned)d);
            uint2 key = make_uint2(seed, 0u);
            uint4 r4 = philox4x32(ctr, key);
            double g2;
            double g1 = gauss_pair(r4.x, r4.y, &g2);
            v[base + d] = c * v[base + d] + ns * g1;
        }
    }
    for (int d = 0; d < 3; ++d) {
        x[base + d] = x[base + d] + s * v[base + d];
    }
}
"""

_SRC_CACHE: dict[str, object] = {}


def _kernels():
    import cupy as cp
    if "k" not in _SRC_CACHE:
        mod = cp.RawModule(code=_KERNEL_TMPL, options=("-fmad", "false"))
        _SRC_CACHE["k"] = (mod.get_function("half_kick"),
                           mod.get_function("drift_orn"))
    return _SRC_CACHE["k"]


def _d(a):
    import cupy as cp
    if hasattr(a, "get"):  # already on device
        return cp.ascontiguousarray(a, dtype=np.float64)
    return cp.asarray(np.ascontiguousarray(a, dtype=np.float64))


def half_kick(v, f, mass, n: int, r: int, s: float):
    """In-place opus kick: v += (s*f)/m (v/f flat (n*r*3,), mass (n,))."""
    import cupy as cp
    k_kick, _ = _kernels()
    n_tot = n * r * 3
    k_kick(((n_tot + 255) // 256,), (256,),
           (v, f, _d(mass), n_tot, r * 3, float(s)))
    cp.cuda.Stream.null.synchronize()


def drift_orn(x, v, mass, n: int, r: int, *, s: float, gamma: float,
              dt: float, kT: float, step: int, seed: int):
    """In-place half-drift + (optional) O-step + half-drift."""
    import cupy as cp
    _, k_drift = _kernels()
    nr = n * r
    k_drift(((nr + 255) // 256,), (256,),
            (x, v, _d(mass), nr, r, float(s), float(gamma), float(dt),
             float(kT), np.uint64(step), np.uint32(seed)))
    cp.cuda.Stream.null.synchronize()


def baoab_step(x, v, f, mass, invm, n: int, r: int, *, dt: float,
               gamma: float, kT: float, step: int, seed: int, first: bool,
               r_oh: float | None = None, r_hh: float | None = None,
               iters: int = 12):
    """One streaming BAOAB step; forces f held = F(x) (recompute after).

    Composition == opus Dynamics.step_baoab bitwise (gamma=0 aligned;
    gamma>0 pending pillar-3 RNG):
      not first: half_kick(f) closes the previous step, then RATTLE;
      half_kick(f) opens this step (f1(n) == f2(n-1)); drift+O+drift;
      SHAKE (when water geometry given, i.e. constraints present).
    """
    from .constrain import project_water, shake_water
    s = 0.5 * dt
    constrained = r_oh is not None
    if not first:
        half_kick(v, f, mass, n, r, s)
        if constrained:
            project_water(v, x, invm, iters=iters)
    half_kick(v, f, mass, n, r, s)
    drift_orn(x, v, mass, n, r, s=s, gamma=gamma, dt=dt, kT=kT,
              step=step, seed=seed)
    if constrained:
        shake_water(x, invm, r_oh=r_oh, r_hh=r_hh, iters=iters)


def baoab_finish(x, v, f, mass, invm, n: int, r: int, *, dt: float,
                 r_oh: float | None = None, r_hh: float | None = None,
                 iters: int = 12):
    """Tail: close the final step (last kick + RATTLE) so (x, v) match the
    opus post-step state after K streamed invocations."""
    from .constrain import project_water
    constrained = r_oh is not None
    half_kick(v, f, mass, n, r, 0.5 * dt)
    if constrained:
        project_water(v, x, invm, iters=iters)

