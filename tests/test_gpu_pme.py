"""GPU collection CI: numpy mirrors of the production CUDA op-order vs opus.

The CUDA sources in gpu/{pme,constrain,integrate}.py are transcriptions of
the numpy mirrors verified here against the opus reference implementations
(bitwise).  On the pod, the kernels themselves are aligned against the same
opus references (E0m-pattern: local oracle -> npz -> GPU dual/bit gate);
this file catches transcription defects before the GPU window and keeps
them caught by CI forever.

Mirror discipline: each mirror replicates the CUDA operation order
literally (left-associated products, division not reciprocal-multiply,
sequential 3-sums, np.rint quantization); any mirror <-> opus mismatch here
is a real semantic defect in the module, not a tolerance question.
"""
import numpy as np
import pytest

from opus import dynamics as dyn_mod
from opus.dynamics import Dynamics, project_velocities, shake_positions
from opus.fxp import FixedPointAccumulator
from opus.pme import PmeGrid, bspline_weights, cardinal_bspline, spread
from gpu import constrain, integrate, pme


# --------------------------------------------------------------- helpers

def _water_system(nw=27, r=2, seed=3, box_len=2.0, jitter=0.05):
    """Rigid-water cluster: (O,H,H) layout, perturbed ideal geometry."""
    rng = np.random.default_rng(seed)
    ideal = np.array([[0.0, 0.0, 0.0],
                      [0.09572, 0.0, 0.0],
                      [0.02377, 0.09272, 0.0]])
    x = np.empty((nw * 3, r, 3))
    for w in range(nw):
        c = rng.uniform(0.3, box_len - 0.3, 3)
        rot = rng.normal(size=(3, 3))
        q, _ = np.linalg.qr(rot)
        for k in range(3):
            base = c + ideal[k] @ q + jitter * rng.normal(size=3)
            for rep in range(r):  # per-replica thermal jitter
                x[3 * w + k, rep] = base + 0.01 * rng.normal(size=3)
    x %= box_len
    m = np.tile(np.array([15.999, 1.008, 1.008]), nw)
    return x, m


def _water_constraints(nw, r_oh=0.09572, r_hh=0.15139):
    from opus.ir import Constraint  # noqa: F401  (may not exist; duck-typed)
    return None


class _C:
    __slots__ = ("a", "b", "distance")

    def __init__(self, a, b, d):
        self.a, self.b, self.distance = a, b, d


def water_constraint_list(nw, r_oh=0.09572, r_hh=0.15139):
    """Water-major list: per water (OH1, OH2, HH) -- the order both opus
    shake_positions and the CUDA kernel iterate."""
    out = []
    for w in range(nw):
        o, h1, h2 = 3 * w, 3 * w + 1, 3 * w + 2
        out += [_C(o, h1, r_oh), _C(o, h2, r_oh), _C(h1, h2, r_hh)]
    return out


# --------------------------------------------------------- gpu.pme source

class TestPmeSource:
    def test_scale_from_fxp(self):
        from opus.fxp import Q16_48
        assert pme.SCALE == 1 << Q16_48["frac_bits"]

    @pytest.mark.parametrize("ng,tc", [(128, 12), (32, 8)])
    def test_kernel_source_ascii(self, ng, tc):
        src = pme.kernel_source(ng, tc)
        assert "__global__" in src
        assert all(ord(c) < 128 for c in src)

    def test_ng_must_be_power_of_two(self):
        with pytest.raises(AssertionError):
            pme.kernel_source(100, 12)

    @pytest.mark.parametrize("path", [constrain, integrate, pme])
    def test_module_cuda_strings_ascii(self, path):
        import re
        text = open(path.__file__, encoding="utf-8").read()
        for m in re.finditer(r'r?"""(.*?)"""', text, re.S):
            if "__global__" not in m.group(1):
                continue
            bad = [c for c in m.group(1) if ord(c) > 127]
            assert not bad, f"{path.__name__}: non-ASCII CUDA SRC: {bad[:3]!r}"

    def test_component_base_offset_tripwire(self):
        """(a*R+r) is an ATOM-REPLICA index; the component base is *3.
        The 2026-09-10 pod alignment caught coords_u called without the
        *3 (replica 0 correct by luck, every other (a,r) reading shifted
        components).  This tripwire pins the corrected call sites."""
        src = open(pme.__file__, encoding="utf-8").read()
        assert src.count("coords_u(x, ((long long)a * R + r) * 3") == 2
        assert src.count("coords_u(xs, ((long long)e * R + r) * 3") == 1
        # constrain/integrate carry their own *3 (match direct.py style)
        csrc = open(constrain.__file__, encoding="utf-8").read()
        assert csrc.count("(3 * w) * R + r) * 3") == 2
        isrc = open(integrate.__file__, encoding="utf-8").read()
        assert "(long long)idx * 3" in isrc


# -------------------------------------------------- weights mirror (opus)

def mirror_cbs(p: int, x: float) -> float:
    """CUDA cbs() transcribed literally (see gpu/pme.py)."""
    if p == 1:
        return 1.0 if (0.0 <= x < 1.0) else 0.0
    l = x / (p - 1) * mirror_cbs(p - 1, x)
    r = (p - x) / (p - 1) * mirror_cbs(p - 1, x - 1.0)
    return l + r


class TestWeightsMirror:
    def test_cbs_bitwise_vs_opus(self):
        xs = np.linspace(-2.0, 6.0, 4001)
        for x in xs:
            assert mirror_cbs(4, float(x)) == cardinal_bspline(float(x), 4)

    def test_wts4_anchor_wrap_vs_opus(self):
        rng = np.random.default_rng(0)
        ng = 32
        for u in np.concatenate([rng.uniform(0, 1, 200),
                                 [0.0, 1e-16, 0.999999, 1.0 / ng]]):
            xg = u * ng
            a = int(np.floor(xg))
            anchors_op, w_op = bspline_weights(float(u), ng, 4)
            anchors_op = anchors_op % ng
            for t in range(4):
                gg = a - 3 + t
                assert gg % ng == anchors_op[t]
                assert mirror_cbs(4, xg - float(gg)) == w_op[t]


# -------------------------------------------------- spread mirror (opus)

def mirror_spread(x, q, box, ng):
    """gpu/pme.py spread_v1 transcribed to numpy (layout, order, quantize).

    Grid flat index within a replica: (ix*ng + iy)*ng + iz (opus C-order,
    dim0 slowest); u = fmod-wrap of x*inv_diag; val = ((q*w0)*w1)*w2;
    dep = rint(val * 2^48); exact int adds.
    """
    inv = np.linalg.inv(np.asarray(box, dtype=float))
    scale = float(pme.SCALE)
    N, R, _ = x.shape
    grid = np.zeros((R, ng, ng, ng), dtype=np.int64)
    for r in range(R):
        for a in range(N):
            u = [np.float64(x[a, r, d]) * inv[d, d] for d in range(3)]
            u = [np.float64(np.fmod(uu, 1.0) + 1.0 if np.fmod(uu, 1.0) < 0
                            else np.fmod(uu, 1.0)) for uu in u]
            ws = []
            for d in range(3):
                xg = u[d] * ng
                a0 = int(np.floor(xg))
                g = [a0 - 3 + t for t in range(4)]
                w = [mirror_cbs(4, xg - float(gg)) for gg in g]
                ws.append(([gg % ng for gg in g], w))
            qi = q[a]
            (ax, wx), (ay, wy), (az, wz) = ws
            for dz in range(4):
                for dy in range(4):
                    for dx in range(4):
                        val = ((qi * wx[dx]) * wy[dy]) * wz[dz]
                        dep = int(np.rint(val * scale))
                        grid[r, ax[dx], ay[dy], az[dz]] += dep
    return grid


class TestSpreadMirror:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_mirror_bitwise_vs_opus(self, seed):
        rng = np.random.default_rng(10 + seed)
        ng, L = 32, 2.0
        box = np.diag([L, L, L])
        x, _ = _water_system(nw=10, r=2, seed=seed, box_len=L)
        x = x.copy()
        if seed:  # seam stress: park some atoms across the ng boundary
            x[::7, :, :] = rng.uniform(0.0, 0.02, x[::7].shape)
            x[1::11, :, :] = rng.uniform(L - 0.02, L - 1e-9, x[1::11].shape)
        q = rng.uniform(-1, 1, x.shape[0])
        g = PmeGrid(box, (ng, ng, ng), alpha=2.0, order=4)
        acc = FixedPointAccumulator((ng, ng, ng), int_bits=16, frac_bits=48)
        for r in range(x.shape[1]):
            frac = x[:, r, :] @ np.linalg.inv(box)  # opus frac path
            spread(frac, q, g, acc)
            assert (acc.acc.reshape(ng, ng, ng)
                    == mirror_spread(x[:, r:r + 1], q, box, ng)[0]).all()
            acc.acc[:] = 0


# ------------------------------------------------- cell sort + tile logic

def _tile_sim_clean(x, box, ng, tc, shift, atom_e, cs, ce, org, ncell):
    inv = np.diag(np.linalg.inv(np.asarray(box, dtype=float)))
    R = x.shape[1]
    ts = tc + 3
    grid = np.zeros((R, ng, ng, ng), dtype=np.int64)
    for c in range(ncell):
        ox, oy, oz = (int(v) for v in org[c])
        for r in range(R):  # kernel: one block per (cell, replica), own tile
            tile = np.zeros((ts, ts, ts), dtype=np.int64)
            for e in range(cs[c], ce[c]):
                a = int(atom_e[e])
                au = []
                for d in range(3):
                    xg = np.float64(x[a, r, d]) * inv[d]
                    m = np.fmod(xg, 1.0)
                    if m < 0:
                        m += 1.0
                    au.append(int(np.floor(m * ng)))
                sx, sy, sz = (int(v) for v in shift[e, r])
                for dz in range(4):
                    tz = (au[2] - 3 + dz) + sz * ng - oz + 1
                    if not (0 <= tz < ts):
                        continue
                    for dy in range(4):
                        ty = (au[1] - 3 + dy) + sy * ng - oy + 1
                        if not (0 <= ty < ts):
                            continue
                        for dx in range(4):
                            tx = (au[0] - 3 + dx) + sx * ng - ox + 1
                            if not (0 <= tx < ts):
                                continue
                            tile[tz, ty, tx] += 1
            for tzi in range(1, tc + 1):
                for tyi in range(1, tc + 1):
                    for txi in range(1, tc + 1):
                        v = int(tile[tzi, tyi, txi])
                        if not v:
                            continue
                        # short last block: unwrapped points beyond ng are
                        # block 0's, not ours (kernel flush guard mirror)
                        if (ox + txi - 1 >= ng or oy + tyi - 1 >= ng
                                or oz + tzi - 1 >= ng):
                            continue
                        gx = (ox + txi - 1) % ng
                        gy = (oy + tyi - 1) % ng
                        gz = (oz + tzi - 1) % ng
                        grid[r, gx, gy, gz] += v
    return grid


def _v1_sim(x, box, ng):
    inv = np.diag(np.linalg.inv(np.asarray(box, dtype=float)))
    N, R, _ = x.shape
    grid = np.zeros((R, ng, ng, ng), dtype=np.int64)
    for r in range(R):
        for a in range(N):
            au = []
            for d in range(3):
                xg = np.float64(x[a, r, d]) * inv[d]
                m = np.fmod(xg, 1.0)
                if m < 0:
                    m += 1.0
                au.append(int(np.floor(m * ng)))
            for dz in range(4):
                for dy in range(4):
                    for dx in range(4):
                        grid[r, (au[0] - 3 + dx) % ng,
                             (au[1] - 3 + dy) % ng, (au[2] - 3 + dz) % ng] += 1
    return grid


class TestCellSort:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("tc", [8, 12])  # 12: tc does not divide 32
    def test_tile_placement_bitwise_v1(self, seed, tc):
        ng, L = 32, 2.0
        box = np.diag([L, L, L])
        x, _ = _water_system(nw=15, r=3, seed=seed, box_len=L, jitter=0.3)
        rng = np.random.default_rng(40 + seed)
        x = x.copy()
        x[::5] = rng.uniform(0.0, 0.03, x[::5].shape)      # seam head
        x[2::7] = rng.uniform(L - 0.03, L - 1e-12, x[2::7].shape)  # seam tail
        atom_e, shift, cs, ce, org, ncell = pme.cell_sort(x, box, ng, tc)
        assert ncell > 0
        g_tile = _tile_sim_clean(x, box, ng, tc, shift, atom_e, cs, ce,
                                 org, ncell)
        g_v1 = _v1_sim(x, box, ng)
        assert (g_tile == g_v1).all(), "tile placement != v1 placement"

    def test_owner_entries_complete_and_unique(self):
        """Every (atom, replica, stencil point)'s owner block has an entry;
        entries are unique per (atom, block)."""
        ng, tc, L = 32, 8, 2.0
        box = np.diag([L, L, L])
        x, _ = _water_system(nw=10, r=2, seed=9, box_len=L, jitter=0.3)
        x[0] = np.linspace(0.0, L - 1e-9,
                           x.shape[1] * 3).reshape(x.shape[1], 3)
        atom_e, shift, cs, ce, org, ncell = pme.cell_sort(x, box, ng, tc)
        inv = np.diag(np.linalg.inv(box))

        def cell_of(i):
            return int(np.searchsorted(cs, i, "right")) - 1

        atom_blocks: dict[int, set] = {}
        for i, a in enumerate(atom_e):
            o = org[cell_of(i)]
            atom_blocks.setdefault(int(a), set()).add(
                (int(o[0]) // tc, int(o[1]) // tc, int(o[2]) // tc))
        n_entries = len(atom_e)
        assert sum(len(v) for v in atom_blocks.values()) == n_entries
        for a, blocks in atom_blocks.items():
            for r in range(x.shape[1]):
                au = []
                for d in range(3):
                    xg = np.float64(x[a, r, d]) * inv[d]
                    m = np.fmod(xg, 1.0)
                    if m < 0:
                        m += 1.0
                    au.append(int(np.floor(m * ng)))
                for dx in range(4):
                    for dy in range(4):
                        for dz in range(4):
                            p = ((au[0] - 3 + dx) % ng,
                                 (au[1] - 3 + dy) % ng,
                                 (au[2] - 3 + dz) % ng)
                            owner = (p[0] // tc, p[1] // tc, p[2] // tc)
                            assert owner in blocks, \
                                f"atom {a} point {p} owner {owner} no entry"
