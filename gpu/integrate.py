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

RNG (pillar 3, LANDED 2026-09-10): the O-step gaussian is the real numpy
Philox4x64-10 + 256-level ziggurat stream (gpu/rand.py emit_cuda),
keyed opus.rng._mix(global_seed, step+1, atom+1, slot+1, dof+1) with
slot = replica index (M2 multi-replica extension of the composition
contract; stable atom id == array index pre-reorder).  gamma>0 dynamics
are alignable against opus.bitwise provided the device log1p/exp parity
probe passes (e0n report).
"""
from __future__ import annotations

import numpy as np

from . import rand

_KERNEL_TMPL = r"""
__RAND__

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
// one thread per (atom, replica); O-step gaussian = opus.rng.gauss_stream
// (pillar 3): key=(seed, step, atom, slot=r, dof=d), numpy ziggurat stream
extern "C" __global__ void drift_orn(
    double* __restrict__ x, double* __restrict__ v,
    const double* __restrict__ m, int nr, int R,
    double s, double gamma, double dt, double kT,
    unsigned long long step, unsigned long long seed,
    const double* __restrict__ wi,
    const unsigned long long* __restrict__ ki,
    const double* __restrict__ fi,
    double nor_r, double nor_inv_r)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr) return;
    int a = idx / R, r = idx - a * R;
    double c = exp(-gamma * dt);
    double ns = sqrt((kT * (1.0 - c * c)) / m[a]);
    long long base = (long long)idx * 3;
    for (int d = 0; d < 3; ++d) {
        x[base + d] = x[base + d] + s * v[base + d];
    }
    if (gamma > 0.0) {
        for (int d = 0; d < 3; ++d) {
            double g1 = gauss_stream1(seed, step,
                                      (unsigned long long)a,
                                      (unsigned long long)r,
                                      (unsigned long long)d,
                                      wi, ki, fi, nor_r, nor_inv_r);
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
        from .rand import emit_cuda
        src = _KERNEL_TMPL.replace("__RAND__", emit_cuda())
        mod = cp.RawModule(code=src, options=("-fmad", "false"))
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


_TABLES_DEV: dict = {}


def _rng_tables():
    """Upload ziggurat tables once (device cache)."""
    import cupy as cp
    if "wi" not in _TABLES_DEV:
        t = rand._TABLES
        _TABLES_DEV["wi"] = cp.asarray(np.array(t["wi"], dtype=np.float64))
        _TABLES_DEV["ki"] = cp.asarray(np.array(t["ki"], dtype=np.uint64))
        _TABLES_DEV["fi"] = cp.asarray(np.array(t["fi"], dtype=np.float64))
        _TABLES_DEV["nor_r"] = float(t["nor_r"])
        _TABLES_DEV["nor_inv_r"] = float(t["nor_inv_r"])
    return _TABLES_DEV


def drift_orn(x, v, mass, n: int, r: int, *, s: float, gamma: float,
              dt: float, kT: float, step: int, seed: int):
    """In-place half-drift + (optional) O-step + half-drift."""
    import cupy as cp
    _, k_drift = _kernels()
    t = _rng_tables()
    nr = n * r
    k_drift(((nr + 255) // 256,), (256,),
            (x, v, _d(mass), nr, r, float(s), float(gamma), float(dt),
             float(kT), np.uint64(step), np.uint64(seed),
             t["wi"], t["ki"], t["fi"], t["nor_r"], t["nor_inv_r"]))
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

