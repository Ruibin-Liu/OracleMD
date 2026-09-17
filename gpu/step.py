"""Production MD step assembly (M2): the full GPU step chain.

    spread (Q16.48 tile) -> dequantize -> rfftn -> c(m) -> irfftn -> phi
    -> interp_gather (adjoint reciprocal forces, fp64)
    + direct_q (Q24.40, per-pair pillar-1 accumulation)
    -> force combine (elementwise int64 add of the quantized reciprocal
       contribution -- opus single_point quantizes the recip force once
       per atom; elementwise == per (atom, replica, component), order-free)
    -> dequantize (exact 2^-40 scaling)
    -> baoab_step (+SHAKE / RATTLE inline)

K-window chunking (the neighbor-list rebuild cadence) must not perturb
the trajectory: pillar 5 (different lists -> identical forces; pairs
beyond rc+skin are exact zero / absent) + pillar 3 (the O-step RNG keys
on the ABSOLUTE step counter, never a window-relative one).  The M2 gate:
forall K in {1,5,25,50}, same binary, bitwise-equal trajectories.

C1 flag (ruling a): heavy-atom (mass >= 2) displacement from the window
start vs d_thresh = 0.14 nm; K=50 is the deliberate over-budget point
(200 fs heavy-atom displacement p50 0.16-0.18 > 0.14 at dt=4 fs).
"""
from __future__ import annotations

import numpy as np

D_THRESH = 0.14  # nm, ruling a


class StepCtx:
    """Immutable per-system constants + device-resident influence table."""

    def __init__(self, *, q, sig, eps, mass, box, ng, tc, alpha, rc, skin,
                 dt, gamma, kT, seed, r_oh, r_hh, shake_iters=12):
        import cupy as cp
        self.q = np.ascontiguousarray(q, dtype=np.float64)
        self.sig = np.ascontiguousarray(sig, dtype=np.float64)
        self.eps = np.ascontiguousarray(eps, dtype=np.float64)
        self.mass = np.asarray(mass, dtype=np.float64)
        self.invm = (1.0 / self.mass).astype(np.float64)
        self.box = np.asarray(box, dtype=np.float64)
        self.ng = int(ng)
        self.tc = int(tc)
        self.alpha = float(alpha)
        self.rc = float(rc)
        self.skin = float(skin)
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.kT = float(kT)
        self.seed = int(seed)
        self.r_oh = float(r_oh)
        self.r_hh = float(r_hh)
        self.shake_iters = int(shake_iters)
        self.n = int(self.q.shape[0])
        from opus.pme import PmeGrid
        g = PmeGrid(self.box, (self.ng, self.ng, self.ng), self.alpha,
                    order=4)
        self.inv_box = g.inv_box
        ch = np.ascontiguousarray(g.c[:, :, :self.ng // 2 + 1])
        self.c_half_dev = cp.asarray(ch)
        from gpu import force_q24, pme
        self.pme = pme
        self.force_q24 = force_q24
        self._dev = {}

    def dev_consts(self):
        import cupy as cp
        if "q" not in self._dev:
            self._dev["q"] = cp.asarray(self.q)
            self._dev["sig"] = cp.asarray(self.sig)
            self._dev["eps"] = cp.asarray(self.eps)
            self._dev["mass"] = cp.asarray(self.mass)
            self._dev["invm"] = cp.asarray(self.invm)
        return self._dev

    def lists(self, x_host, pairs):
        """Full per-atom lists from an undirected pair list (exclusions
        already removed by the caller)."""
        n = self.n
        maxnb = max(sum(1 for p in pairs if a in p) for a in range(n)) + 1
        nlist = np.full((n, maxnb), -1, dtype=np.int32)
        ncnt = np.zeros(n, dtype=np.int32)
        for (i, j) in pairs:
            nlist[i, ncnt[i]] = j
            ncnt[i] += 1
            nlist[j, ncnt[j]] = i
            ncnt[j] += 1
        return nlist, ncnt


def brute_pairs_vec(x0, box, rc):
    """Vectorized MIC UNION pair list within rc (host): a pair is listed
    if within rc in ANY replica (opus build_union_list semantics)."""
    n, R, _ = x0.shape
    L = np.diag(np.asarray(box, dtype=np.float64))
    mask = np.zeros((n, n), dtype=bool)
    for r in range(R):
        d = np.abs(x0[:, r, :] - x0[:, r, :][:, None, :])
        d = np.minimum(d, L - d)
        mask |= np.sqrt((d ** 2).sum(-1)) < rc
    iu, ju = np.where(np.triu(np.ones((n, n), dtype=bool), 1) & mask)
    return list(zip(iu.tolist(), ju.tolist()))


def md_window(ctx, x, v, pairs, step0, k_steps, x_snap=None,
              fq_snap=None):
    """Run ONE K-step window in place; x, v are device arrays (flat f64).

    The neighbor list is rebuilt at the window start.  Returns the C1
    flag count: heavy-atom displacement from the window start crossing
    D_THRESH (rule a; heavy = mass >= 2)."""
    import cupy as cp
    from gpu import integrate
    n, R = ctx.n, x.shape[0] // (3 * ctx.n)
    ctx.r_rep = R
    xh = cp.asnumpy(x).reshape(n, R, 3)
    x_win0 = xh.copy()
    nlist, ncnt = ctx.lists(xh, pairs)
    d = ctx.dev_consts()
    d_nlist = cp.asarray(nlist)
    d_ncnt = cp.asarray(ncnt)
    x3 = x.reshape(n, R, 3)  # device-side (N, R, 3) view for the kernels
    flags = 0
    for step in range(step0, step0 + k_steps):
        # PME: spread -> dequant -> rfft -> c(m) -> irfft -> phi
        grid = ctx.pme.spread(x3, d["q"], ctx.box, ng=ctx.ng,
                              mode="tile")
        gf = grid.astype(np.float64) * (2.0 ** -48)
        G = cp.fft.rfftn(gf, axes=(1, 2, 3))
        G *= ctx.c_half_dev
        pot = cp.fft.irfftn(G, s=(ctx.ng, ctx.ng, ctx.ng),
                            axes=(1, 2, 3)).real
        f_recip = ctx.pme.interp_forces(x3, d["q"],
                                        pot.reshape(ctx.r_rep, -1),
                                        ctx.box, ng=ctx.ng)
        # direct: Q24.40 per-pair accumulation
        Fq_dir, sticky = ctx.force_q24.direct_forces_q(
            x3, d["q"], d["sig"], d["eps"], d_nlist, d_ncnt,
            ctx.alpha, ctx.rc, box=ctx.box)
        if sticky:
            raise RuntimeError(f"Q24.40 overflow (sticky={sticky})")
        # combine: opus quantizes the recip contribution once per atom
        Fq = Fq_dir + cp.rint(f_recip * float(1 << 40)).astype(cp.int64)
        f64 = Fq * (2.0 ** -40)
        integrate.baoab_step(x, v, f64.reshape(-1), d["mass"],
                             d["invm"], n, R, dt=ctx.dt, gamma=ctx.gamma,
                             kT=ctx.kT, step=step, seed=ctx.seed,
                             first=(step == 0), r_oh=ctx.r_oh,
                             r_hh=ctx.r_hh, iters=ctx.shake_iters)
        if x_snap is not None:
            x_snap.append(cp.asnumpy(x).copy())
        if fq_snap is not None:
            fq_snap.append(cp.asnumpy(Fq).copy())
    # C1 flag: heavy-atom displacement vs window start (rule a)
    x_end = cp.asnumpy(x).reshape(n, R, 3)
    if (ctx.mass >= 2.0).any():
        disp = np.abs(x_end - x_win0)
        worst = float(disp[ctx.mass >= 2.0].max())
        if worst > D_THRESH:
            flags += 1
    return flags
