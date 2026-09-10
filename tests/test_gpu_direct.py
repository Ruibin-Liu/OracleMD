"""GPU module CI tests (erfc generation on CPU; kernel oracle needs CUDA).

The direct-space kernel oracle alignment (vs opus reference forces) ran
2026-09-10 on A100: 543-atom TIP3P-FB water, max rel 1.07e-11 / abs 4.5e-10
(A1 dual gate PASS). The alignment hunt exposed five defects invisible to
self-consistent prototypes (MIC, force sign, Newton 3rd law half-list,
missing 1/r in Coulomb, KE truncation) -- all fixed in gpu/direct.py;
the KE constant is now imported from opus, never retyped.
"""
import numpy as np
import pytest

from gpu import erfc_poly


class TestErfcPolyGeneration:
    def test_max_rel_le_1e13(self):
        assert erfc_poly.validate(1e-13) <= 3.5e-14

    def test_stored_matches_refit(self):
        assert erfc_poly.max_rel_stored() < 1e-13

    def test_emit_cuda_ascii(self):
        src = erfc_poly.emit_cuda()
        assert "erfc_poly" in src
        assert all(ord(c) < 128 for c in src)

    def test_piece_mapping_constants_exact(self):
        """The 4/9-class bug: per-piece t-scale must be exactly 2/(hi-lo)."""
        import re
        src = erfc_poly.emit_cuda()
        m = re.search(r"double t = x \* ([0-9.e+-]+) - 1.0", src)
        assert m and float(m.group(1)) == pytest.approx(2.0 / 1.5, rel=1e-15)
        m = re.search(r"double t = \(x - 1.5\) \* ([0-9.e+-]+) - 1.0", src)
        assert m and float(m.group(1)) == pytest.approx(2.0 / 1.5, rel=1e-15)
        m = re.search(r"double t = \(xc - 3.0\) \* ([0-9.e+-]+) - 1.0", src)
        assert m and float(m.group(1)) == pytest.approx(2.0 / 3.5, rel=1e-15)


class TestDirectKernelOracle:
    """Kernel-vs-exact two-atom oracle (CUDA required; skips otherwise)."""

    def test_two_atom_exact(self):
        cupy = pytest.importorskip("cupy")
        import math
        from gpu.direct import direct_forces
        from opus.nonbonded import KE  # never retype (alignment defect #5)

        r = 0.25
        x = np.array([[[0., 0., 0.]], [[r, 0., 0.]]])
        q = np.array([0.8, -0.4])
        sig = np.array([0.33, 0.29])
        eps = np.array([0.35, 0.60])
        nlist = np.array([[1], [0]], dtype=np.int32)
        ncnt = np.array([1, 1], dtype=np.int32)
        box = [10.0, 10.0, 10.0]
        Fl = direct_forces(x, np.zeros_like(q), sig, eps, nlist, ncnt,
                           3.5, 1.0, box_diag=box)
        Fc = direct_forces(x, q, sig, np.full_like(eps, 1e-300), nlist, ncnt,
                           3.5, 1.0, box_diag=box)
        Fl = Fl.get() if hasattr(Fl, "get") else Fl
        Fc = Fc.get() if hasattr(Fc, "get") else Fc
        F = Fl[:, 0, :] + KE * Fc[:, 0, :]
        er = math.erfc(3.5 * r)
        e = math.exp(-(3.5 * r) ** 2)
        sr6 = (0.31 / r) ** 6
        lj = -24 * (0.35 * 0.60) ** 0.5 * (2 * sr6 ** 2 - sr6) / r
        co = -KE * 0.8 * (-0.4) * (er / r ** 2
                                   + 2 * 3.5 * 0.5641895835477563 * e / r)
        assert abs(F[0, 0] - (lj + co)) < 1e-10 * abs(lj + co)
        assert np.allclose(F[0, 1:], 0.0, atol=1e-12)
        # Newton's third law
        assert np.allclose(F[1], -F[0], rtol=1e-12, atol=1e-12)
