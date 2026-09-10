"""Q24.40 direct-space forces (pillar 1): CI for gpu/force_q24.py.

Three-layer proof structure (e0-pattern, see module docstring):
  - mirror_opus (here) replicates opus.nonbonded.direct_space op-for-op at
    alpha=0, where erfc(0)=1 and exp(-0)=1 are EXACT on every platform:
    compared BITWISE against the real opus ForceAccumulator -- this pins
    the quantization/accumulation/MIC/layout semantics layer;
  - gpu.force_q24.direct_q_mirror_kernel replicates the CUDA kernel's own
    arithmetic (k2 sr6 grouping, kernel-form erfc poly): the pod compares
    the kernel against it bitwise (e0n pattern);
  - kernel vs opus forces: dual gate (poly-erfc dominant, phase-1 style).
All CPU-runnable; the CUDA oracle test skips without cupy.
"""
import numpy as np
import pytest

from opus.engine import ForceAccumulator
from opus.nonbonded import NeighborList, direct_space
from opus.fxp import Q24_40

from gpu import force_q24


# ------------------------------------------------------------- helpers

class _Attr:
    def __init__(self, v):
        self.value = v


class _Atom:
    def __init__(self, q, s, e):
        self.q, self.sigma, self.epsilon = _Attr(q), _Attr(s), _Attr(e)


class _NB:
    def __init__(self, q, sig, eps):
        self.atoms = [_Atom(qq, s, e) for qq, s, e in zip(q, sig, eps)]


def water_cluster(nw, r, seed, box_len=2.0, jitter=0.02, min_sep=0.3):
    """Rigid waters, per-replica thermal jitter, min-distance rejection
    between waters (clean-envelope regime: no Q24.40 saturation; the
    saturation path is register-boundary territory -- opus add_to has a
    latent |scaled|==2^63 float-compare-vs-int64-cast edge, registered)."""
    rng = np.random.default_rng(seed)
    ideal = np.array([[0.0, 0.0, 0.0], [0.09572, 0.0, 0.0],
                      [0.02377, 0.09272, 0.0]])
    x = np.empty((nw * 3, r, 3))
    placed = []  # per-water base centers (pre-jitter), rejection domain
    for w in range(nw):
        for _ in range(200):
            c = rng.uniform(0.3, box_len - 0.3, 3)
            if all(np.linalg.norm(_mic(c - p, box_len)) > min_sep
                   for p in placed):
                break
        else:
            raise RuntimeError("could not place water")
        placed.append(c)
        qq, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        for k in range(3):
            base = c + ideal[k] @ qq + jitter * rng.normal(size=3)
            for rep in range(r):
                x[3 * w + k, rep] = base + 0.01 * rng.normal(size=3)
    x %= box_len
    return x


def _mic(d, L):
    return d - np.rint(d / L) * L


def mirror_opus(x, q, sig, eps, pairs, alpha, rc, box=None):
    """opus.nonbonded.direct_space transcribed op-for-op (alpha=0 path).

    Returns the raw int64 Q24.40 accumulator (N, R, 3)."""
    scale = float(1 << Q24_40["frac_bits"])
    vmax = (1 << 63) - 1
    from opus.nonbonded import KE
    N, R, _ = x.shape
    acc = np.zeros((N, R, 3), dtype=np.int64)
    inv_box = np.linalg.inv(box) if box is not None else None
    bbox = np.linalg.inv(inv_box) if box is not None else None
    ild = np.diag(inv_box) if box is not None else None
    ld = np.diag(bbox) if box is not None else None
    rc2 = rc * rc
    for (i, j) in pairs:
        diff = x[j] - x[i]
        if box is not None:
            u = diff * ild
            u -= np.rint(u)
            diff = u * ld
        r2 = (diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]) \
            + diff[:, 2] * diff[:, 2]
        inside = (r2 < rc2).astype(np.float64)
        r2s = np.where(r2 < 1e-12, 1e-12, r2)
        rr = np.sqrt(r2s)
        inv_r = 1.0 / rr
        inv_r2 = inv_r * inv_r
        sij = 0.5 * (sig[i] + sig[j])
        s6 = (np.array([sij]) ** 6)[0]          # opus sig_ij ** 6 (array ufunc)
        sr6 = s6 * (np.array([inv_r2]) ** 3)[0]  # opus sr6_base * inv_r2**3
        dulj = -24.0 * np.sqrt(eps[i] * eps[j]) * (2.0 * sr6 * sr6 - sr6) \
            * inv_r
        # alpha = 0: er = erfc(0) = 1.0, exp(-0) = 1.0 (exact everywhere)
        dcoul = -(1.0 * inv_r2 + 0.0)
        duc = KE * (q[i] * q[j]) * dcoul
        coef = (dulj + duc) * inside * inv_r
        fi = coef[:, None] * diff
        for at, f in ((i, fi), (j, -fi)):
            scaled = np.rint(f * scale)
            bad = ~np.isfinite(scaled) | (np.abs(scaled) > vmax)
            scaled = np.where(bad, np.sign(scaled) * vmax, scaled)
            # opus add_to: RUNNING saturation of the accumulator (object
            # arithmetic) -- raw int64 wrap diverges once multiple
            # saturated contributions stack
            ssum = acc[at].astype(object) + scaled.astype(np.int64) \
                .astype(object)
            over = (ssum > vmax) | (ssum < -vmax - 1)
            if over.any():
                ssum = np.where(ssum > vmax, vmax,
                                np.where(ssum < -vmax - 1, -vmax - 1, ssum))
            acc[at] = ssum.astype(np.int64)
    return acc


def min_atom_sep(x, box=None, same_molecule=None):
    """min INTER-molecular all-atom distance (envelope check; bonded
    intra-molecule pairs are excluded via same_molecule(i, j))."""
    n, r, _ = x.shape
    same = same_molecule or (lambda i, j: False)
    m = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if same(i, j):
                continue
            d = np.abs(x[j] - x[i])
            if box is not None:
                d = np.minimum(d, np.diag(box) - d)
            m = min(m, float(np.sqrt((d ** 2).sum(-1)).min()))
    return m


def full_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def run_opus(x, q, sig, eps, pairs, alpha, rc, box=None):
    nb = _NB(q, sig, eps)
    f_acc = ForceAccumulator(x.shape[0], x.shape[1])
    direct_space(nb, x, set(), f_acc, alpha, rc,
                 NeighborList(np.array(pairs, dtype=np.int64)), box=box)
    return f_acc.acc.acc


# ------------------------------------------------------- constants/SRC

class TestConstants:
    def test_scale_from_fxp(self):
        assert (1 << Q24_40["frac_bits"]) == 1 << 40

    def test_no_retyped_constants(self):
        """Trap class 8 tripwire: KE / 2/sqrt(pi) / 2**40 must be imported,
        never transcribed into the source."""
        src = open(force_q24.__file__, encoding="utf-8").read()
        for bad in ("138.93", "1.1283", "1099511627776", "0.5641895"):
            assert bad not in src, f"retyped constant {bad!r}"

    def test_kernel_source_ascii(self):
        src = force_q24.kernel_source()
        assert "__global__" in src
        assert all(ord(c) < 128 for c in src)
        assert "138.93545764438198" in src  # KE via repr(opus.nonbonded.KE)


# ---------------------------------------------- mirror vs opus (bitwise)

class TestMirrorOpusBitwise:
    def test_two_atom_aperiodic(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.3, 1.8, (2, 2, 3))
        q, sig, eps = rng.uniform(-1, 1, 2), rng.uniform(.2, .4, 2), \
            rng.uniform(.1, .6, 2)
        pairs = full_pairs(2)
        ref = run_opus(x, q, sig, eps, pairs, 0.0, 5.0)
        got = mirror_opus(x, q, sig, eps, pairs, 0.0, 5.0)
        assert (ref.view(np.int64) == got.view(np.int64)).all()

    def test_cluster_with_cutoff_mask(self):
        rng = np.random.default_rng(1)
        n = 8
        x = rng.uniform(0.0, 2.0, (n, 2, 3))
        q, sig, eps = rng.uniform(-1, 1, n), rng.uniform(.2, .4, n), \
            rng.uniform(.1, .6, n)
        pairs = full_pairs(n)
        ref = run_opus(x, q, sig, eps, pairs, 0.0, 0.9)  # mask fires
        got = mirror_opus(x, q, sig, eps, pairs, 0.0, 0.9)
        assert (ref.view(np.int64) == got.view(np.int64)).all()
        assert (got == 0).any(), "cutoff mask did not zero anything"

    def test_water_periodic_mic(self):
        nw, r, L = 12, 2, 2.0
        x = water_cluster(nw, r, seed=3, box_len=L, jitter=0.02,
                          min_sep=0.43)
        rng = np.random.default_rng(4)
        n = nw * 3
        q = rng.uniform(-1, 1, n)
        sig = rng.uniform(.2, .4, n)
        eps = rng.uniform(.1, .6, n)
        box = np.diag([L, L, L])
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        ref = run_opus(x, q, sig, eps, pairs, 0.0, 0.7, box=box)
        got = mirror_opus(x, q, sig, eps, pairs, 0.0, 0.7, box=box)
        assert (ref.view(np.int64) == got.view(np.int64)).all()


# ------------------------------------------- mirror_kernel vs mirror_opus

class TestMirrorKernel:
    def test_dual_gate_vs_mirror_opus_alpha0(self):
        """Inter-molecular pairs only (production: bonded pairs are
        exclusions); clean envelope => sticky == 0 on both mirrors."""
        nw, r, L = 8, 2, 2.0
        x = water_cluster(nw, r, seed=5, box_len=L, jitter=0.005,
                          min_sep=0.45)
        rng = np.random.default_rng(6)
        n = nw * 3
        q, sig, eps = rng.uniform(-1, 1, n), rng.uniform(.2, .4, n), \
            rng.uniform(.1, .6, n)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if i // 3 != j // 3]
        box = np.diag([L, L, L])
        assert min_atom_sep(x, box, lambda i, j: i // 3 == j // 3) > 0.18, \
            "test system left the envelope"
        ref = mirror_opus(x, q, sig, eps, pairs, 0.0, 0.8, box=box)
        nlist, ncnt = _lists(n, pairs)
        got, sticky = force_q24.direct_q_mirror_kernel(
            x, q, sig, eps, nlist, ncnt, 0.0, 0.8, box=box)
        assert sticky == 0
        f_ref = ref.astype(np.float64) * 2.0 ** -40
        f_got = got.astype(np.float64) * 2.0 ** -40
        da = np.abs(f_ref - f_got)
        scale = np.maximum(np.abs(f_ref), 1e-30)
        assert np.all((da < 1e-8) | (da / scale < 1e-9)), \
            f"max abs {da.max():.3e}"

    def test_newton3_bitwise_two_atom(self):
        """Full lists + odd-symmetric quantizer => Fq[i] == -Fq[j] exactly
        (two-atom system: each total IS the single pair contribution)."""
        rng = np.random.default_rng(7)
        x = rng.uniform(0.0, 1.5, (2, 2, 3))
        q, sig, eps = rng.uniform(-1, 1, 2), rng.uniform(.2, .4, 2), \
            rng.uniform(.1, .6, 2)
        nlist = np.array([[1, -1], [0, -1]], dtype=np.int32)
        ncnt = np.array([1, 1], dtype=np.int32)
        fq, sticky = force_q24.direct_q_mirror_kernel(
            x, q, sig, eps, nlist, ncnt, 0.0, 5.0)
        assert sticky == 0
        for rep in range(2):
            assert (fq[0, rep].view(np.int64)
                    == (-fq[1, rep]).view(np.int64)).all()


def _lists(n, pairs):
    maxnb = max(sum(1 for p in pairs if a in p) for a in range(n)) + 1
    nlist = np.full((n, maxnb), -1, dtype=np.int32)
    ncnt = np.zeros(n, dtype=np.int32)
    for (i, j) in pairs:
        nlist[i, ncnt[i]] = j
        ncnt[i] += 1
        nlist[j, ncnt[j]] = i
        ncnt[j] += 1
    return nlist, ncnt


# --------------------------------------------------- CUDA (pod, skipped)

class TestDirectQKernel:
    def test_bitwise_vs_mirror_alpha0(self):
        cupy = pytest.importorskip("cupy")
        nw, r, L = 10, 2, 2.0
        x = water_cluster(nw, r, seed=9, box_len=L)
        rng = np.random.default_rng(10)
        n = nw * 3
        q, sig, eps = rng.uniform(-1, 1, n), rng.uniform(.2, .4, n), \
            rng.uniform(.1, .6, n)
        pairs = full_pairs(n)
        nlist, ncnt = _lists(n, pairs)
        ref, _ = force_q24.direct_q_mirror_kernel(
            x, q, sig, eps, nlist, ncnt, 0.0, 0.6, box=np.diag([L, L, L]))
        got, sticky = force_q24.direct_forces_q(
            x, q, sig, eps, nlist, ncnt, 0.0, 0.6, box=np.diag([L, L, L]))
        got = got.get() if hasattr(got, "get") else got
        assert sticky == 0
        assert (got.view(np.int64) == ref.view(np.int64)).all(), \
            "CUDA direct_q != kernel mirror (transcription defect)"
