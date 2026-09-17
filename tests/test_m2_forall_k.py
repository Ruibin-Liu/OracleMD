"""M2 forall-K gate: K-window chunking must not perturb the trajectory.

Spec M2 (bitwise tier): for K in {1, 5, 25, 50}, the SAME binary running
the production step chain (spread -> rFFT -> c(m) -> irFFT -> interp ->
direct_q -> combine -> dequant -> BAOAB + SHAKE/RATTLE) with K-step
neighbor-list windows must produce bitwise-identical trajectories.
Foundation: pillar 5 (different lists -> identical forces) + pillar 3
(O-step RNG keys on the absolute step counter).

Also asserts rollback neutrality (a replayed window from its start state,
same list, reproduces the same end state bitwise) and run-to-run
determinism (M15 sanity).  The C1 flag counts are recorded (not gated:
window partitioning legitimately changes the displacement baselines).

Needs CUDA; skips otherwise.  K=50 is the deliberate over-budget point.
"""
import numpy as np
import pytest

from opus.dynamics import KB_KJ
from gpu.step import brute_pairs_vec


def water_cluster(nw, r, seed, box_len, jitter=0.02, min_sep=0.35):
    rng = np.random.default_rng(seed)
    ideal = np.array([[0.0, 0.0, 0.0], [0.09572, 0.0, 0.0],
                      [0.02377, 0.09272, 0.0]])
    x = np.empty((nw * 3, r, 3))
    placed = []
    for w in range(nw):
        for _ in range(300):
            c = rng.uniform(0.3, box_len - 0.3, 3)
            if all(np.linalg.norm(c - p) > min_sep for p in placed):
                break
        else:
            raise RuntimeError("water placement infeasible: lower nw or "
                               "min_sep (silent overlap corrupts Q24.40)")
        placed.append(c)
        qq, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        for k in range(3):
            base = c + ideal[k] @ qq + jitter * rng.normal(size=3)
            for rep in range(r):
                x[3 * w + k, rep] = base + 0.01 * rng.normal(size=3)
    x %= box_len
    return x


@pytest.fixture(scope="module")
def kctx():
    cupy = pytest.importorskip("cupy")
    from gpu.step import StepCtx
    # PHYSICAL TIP3P-FB-like parameters: the production skin (0.35, ruling
    # a) is calibrated for water-like forces -- a random-charge system
    # accelerates H atoms beyond the skin within a window and the forall-K
    # gate then measures the SKIN's validity condition, not the
    # implementation (spec: skin re-scaling triggers include K up).
    nw, r, L = 64, 2, 2.6
    x0 = water_cluster(nw, r, seed=17, box_len=L, jitter=0.01,
                       min_sep=0.42)
    rng = np.random.default_rng(18)
    n = nw * 3
    q = np.tile(np.array([-0.734, 0.367, 0.367]), nw)
    sig = np.tile(np.array([0.3186, 0.0087, 0.0087]), nw)
    eps = np.tile(np.array([0.6502, 0.0, 0.0]), nw)
    m = np.tile(np.array([15.999, 1.008, 1.008]), nw)
    v0 = rng.normal(0, 0.15, x0.shape)
    ctx = StepCtx(q=q, sig=sig, eps=eps, mass=m, box=np.diag([L, L, L]),
                  ng=64, tc=8, alpha=3.5, rc=0.9, skin=0.35,
                  dt=2.0e-3, gamma=1.0, kT=KB_KJ * 300.0, seed=77,
                  r_oh=0.09572, r_hh=0.15139)
    return dict(ctx=ctx, x0=x0, v0=v0, n=n, r=r, L=L, cupy=cupy)


def run_arm(kctx, K, nsteps=50, replay_last=False):
    import cupy as cp
    from gpu.step import md_window, brute_pairs_vec
    ctx, x0, v0 = kctx["ctx"], kctx["x0"], kctx["v0"]
    n, r, L = kctx["n"], kctx["r"], kctx["L"]
    x = cp.asarray(x0.reshape(-1))
    v = cp.asarray(v0.reshape(-1))
    flags = []
    starts = []
    step = 0
    while step < nsteps:
        k_steps = min(K, nsteps - step)
        xh = cp.asnumpy(x).reshape(n, r, 3)
        pairs = [(i, j) for (i, j) in brute_pairs_vec(
            xh, np.diag([L, L, L]), ctx.rc + ctx.skin)
            if i // 3 != j // 3]
        if replay_last and step + k_steps == nsteps:
            starts.append((x.copy(), v.copy(), list(pairs)))
        f = md_window(ctx, x, v, pairs, step, k_steps)
        flags.append(f)
        step += k_steps
    out = {"x": cp.asnumpy(x).copy(), "v": cp.asnumpy(v).copy(),
           "flags": flags, "starts": starts}
    return out


class TestForallK:
    def test_trajectory_bitwise_across_K(self, kctx):
        Ks = [1, 5, 25, 50]
        arms = {K: run_arm(kctx, K) for K in Ks}
        base = arms[50]
        for K in Ks:
            a = arms[K]
            assert (a["x"].view(np.int64)
                    == base["x"].view(np.int64)).all(), f"x diverged at K={K}"
            assert (a["v"].view(np.int64)
                    == base["v"].view(np.int64)).all(), f"v diverged at K={K}"
        # run-to-run determinism (M15 sanity) on the K=50 arm
        again = run_arm(kctx, 50)
        assert (again["x"].view(np.int64)
                == base["x"].view(np.int64)).all()
        # C1 flags recorded (informational)
        print("C1 flags per window:",
              {K: arms[K]["flags"] for K in Ks})

    def test_rollback_neutrality(self, kctx):
        """A replayed window from its start state (same list) reproduces
        the same end state bitwise -- C1 rollback is trajectory-neutral."""
        import cupy as cp
        from gpu.step import md_window
        ctx, x0, v0 = kctx["ctx"], kctx["x0"], kctx["v0"]
        n, r, L = kctx["n"], kctx["r"], kctx["L"]
        K, nsteps = 25, 50
        x = cp.asarray(x0.reshape(-1))
        v = cp.asarray(v0.reshape(-1))
        step = 0
        snap = None
        while step < nsteps:
            k_steps = min(K, nsteps - step)
            xh = cp.asnumpy(x).reshape(n, r, 3)
            pairs = brute_pairs_vec(xh, np.diag([L, L, L]),
                                    ctx.rc + ctx.skin)
            if step + k_steps == nsteps:
                snap = (x.copy(), v.copy(), list(pairs), step)
            md_window(ctx, x, v, pairs, step, k_steps)
            step += k_steps
        x_end, v_end = x.copy(), v.copy()
        xs, vs, pairs, step0 = snap
        f = md_window(ctx, xs, vs, pairs, step0, nsteps - step0)
        assert (xs.view(np.int64) == x_end.view(np.int64)).all(), \
            "replayed window diverged (rollback not neutral)"
        assert (vs.view(np.int64) == v_end.view(np.int64)).all()
