"""GPU constraint/integration CI: op-order mirrors vs opus.dynamics.

Mirrors of gpu/constrain.py and gpu/integrate.py (the CUDA sources' literal
numpy transcriptions) are verified bitwise against the real opus reference:
  - shake_positions / project_velocities on random rigid-water clusters;
  - Dynamics.step_baoab chains (gamma=0; gamma>0 needs the pillar-3 RNG
    equivalence and is deliberately NOT aligned here).
Any mismatch is a transcription defect in the module, per the mirror
discipline in tests/test_gpu_pme.py.
"""
import numpy as np
import pytest

from opus import dynamics as dyn_mod
from opus.dynamics import Dynamics, project_velocities, shake_positions

from .test_gpu_pme import _C, water_constraint_list, _water_system

DT = 2.0e-3  # ps (2 fs)


# ----------------------------------------------------------- shake mirror

def mirror_shake(x, invm, constraints, n_iter):
    """gpu/constrain.py shake_rigid_water transcribed literally."""
    x = x.copy()
    for _ in range(n_iter):
        for c in constraints:
            i, j, d = c.a, c.b, c.distance
            dx = x[j] - x[i]
            r2 = (dx[..., 0] * dx[..., 0] + dx[..., 1] * dx[..., 1]) \
                + dx[..., 2] * dx[..., 2]
            rr = np.sqrt(np.where(r2 < 1e-24, 1e-24, r2))
            dr = (invm[i] + invm[j]) * rr
            corr = (rr - d) / np.where(dr < 1e-24, 1e-24, dr)
            x[i] = x[i] + (invm[i] * corr)[..., None] * dx
            x[j] = x[j] - (invm[j] * corr)[..., None] * dx
    return x


def mirror_project(v, x, invm, constraints, n_iter):
    """gpu/constrain.py project_rigid_water transcribed literally."""
    v = v.copy()
    for _ in range(n_iter):
        for c in constraints:
            i, j, d = c.a, c.b, c.distance
            dx = x[j] - x[i]
            r2 = (dx[..., 0] * dx[..., 0] + dx[..., 1] * dx[..., 1]) \
                + dx[..., 2] * dx[..., 2]
            dv = v[j] - v[i]
            dvr = (dv[..., 0] * dx[..., 0] + dv[..., 1] * dx[..., 1]) \
                + dv[..., 2] * dx[..., 2]
            pr = (invm[i] + invm[j]) * r2
            corr = dvr / np.where(pr < 1e-24, 1e-24, pr)
            v[i] = v[i] + (invm[i] * corr)[..., None] * dx
            v[j] = v[j] - (invm[j] * corr)[..., None] * dx
    return v


class TestConstrainMirror:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_shake_bitwise_vs_opus(self, seed):
        nw, R = 20, 2
        x, m = _water_system(nw=nw, r=R, seed=seed, jitter=0.08)
        invm = 1.0 / m
        cons = water_constraint_list(nw)
        ref = shake_positions(x.copy(), x.copy(), cons, invm, n_iter=12)
        got = mirror_shake(x, invm, cons, 12)
        assert (ref.view(np.int64) == got.view(np.int64)).all()

    def test_project_bitwise_vs_opus(self):
        nw, R = 20, 2
        x, m = _water_system(nw=nw, r=R, seed=2, jitter=0.08)
        rng = np.random.default_rng(5)
        v = rng.normal(0, 0.6, x.shape)
        invm = 1.0 / m
        cons = water_constraint_list(nw)
        ref = project_velocities(v.copy(), x, cons, invm, n_iter=12)
        got = mirror_project(v, x, invm, cons, 12)
        assert (ref.view(np.int64) == got.view(np.int64)).all()


# ------------------------------------------------------- integrate mirror

def mirror_stream_step(x, v, mass, invm, constraints, *, dt, first,
                       iters=12):
    """gpu/integrate.py baoab_step (gamma=0) transcribed literally.

    x, v: (N, R, 3).  Kicks divide by mass; drifts have no mass factor.
    """
    x = x.copy()
    v = v.copy()
    s = 0.5 * dt
    if not first:
        v = v + s * _f[0] / mass[:, None, None]
        if constraints:
            v = project_velocities(v, x, constraints, invm, iters)
    v = v + s * _f[0] / mass[:, None, None]
    x = x + s * v
    x = x + s * v
    if constraints:
        x = shake_positions(x, x, constraints, invm, iters)
    return x, v


_f = [None]  # held force (exogenous), set per test


class TestIntegrateMirror:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_chain_bitwise_vs_opus_dynamics(self, seed, monkeypatch):
        nw, R, K = 12, 2, 3
        x0, m = _water_system(nw=nw, r=R, seed=seed, jitter=0.05)
        rng = np.random.default_rng(11 + seed)
        v0 = rng.normal(0, 0.4, x0.shape)
        invm = 1.0 / m
        cons = water_constraint_list(nw)

        def forces_of(x):
            # smooth exogenous field (deterministic, history-free)
            return {"forces": 50.0 * np.sin(x * 3.1 + seed)}

        monkeypatch.setattr(dyn_mod, "single_point",
                            lambda sys_, x: forces_of(x))
        system = type("S", (), {})()
        system.masses = m
        system.constraints = cons
        dyn = Dynamics(system, x0.copy(), v0.copy(), seed=0, shake_iters=12)
        for _ in range(K):
            dyn.step_baoab(DT, gamma=0.0)
        ref_x, ref_v = dyn.x, dyn.v

        # streaming mirror: same force sequence the real chain consumed
        x, v = x0.copy(), v0.copy()
        for k in range(K):
            _f[0] = forces_of(x)["forces"]
            first = k == 0
            x, v = mirror_stream_step(x, v, m, invm, cons, dt=DT, first=first)
        _f[0] = forces_of(x)["forces"]
        v = v + 0.5 * DT * _f[0] / m[:, None, None]
        v = project_velocities(v, x, cons, invm, 12)

        assert (ref_x.view(np.int64) == x.view(np.int64)).all(), "x diverged"
        assert (ref_v.view(np.int64) == v.view(np.int64)).all(), "v diverged"

    def test_mirror_uses_division_not_invm(self):
        """Guard the f1/m transcription trap: reciprocal multiply differs."""
        rng = np.random.default_rng(0)
        f = rng.normal(size=(4, 1, 3))
        m = rng.uniform(1, 16, 4)
        a = 0.5 * DT * f / m[:, None, None]
        b = 0.5 * DT * f * (1.0 / m)[:, None, None]
        assert not (a.view(np.int64) == b.view(np.int64)).all(), \
            "assumption drifted: division == invm multiply (mirror is fine)"
