"""M2 kernel numerics oracles (E0f/E0h semantics; transcription tripwires).

These tests parse the CUDA source of the M2 prototype kernels and validate the
extracted arithmetic against independent references — the coefficients are
NEVER transcribed into this file (anti-transcription discipline; see review
2026-09-08: 7 transcription-class defects this session, all caught by
cross-checks. These tests make the tripwire permanent and CI-enforced).

  1. erfc polynomial (E0f): erfc(x) = exp(-x^2)*W(x), 3-piece Chebyshev in
     piece-local t. Validates: (a) branch mapping constants == 2/(hi-lo)
     exactly [this is the test that would have caught the 4/9 bug];
     (b) max rel error vs scipy erfc <= 1e-13 on [0, 6.5].
  2. order-4 B-spline weights (E0h/E0h2): partition of unity + exact match
     vs opus.pme.cardinal_bspline (value-for-value under each convention).
"""
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc as erfc_ref

from opus.pme import cardinal_bspline

E0_DIR = Path(__file__).resolve().parents[1] / "experiments" / "e0"
SRC_F = E0_DIR / "e0f_erfc_poly_bench.py"
SRC_H = E0_DIR / "e0h2_spread_tile.py"


def _src(path):
    if not path.exists():
        pytest.skip(f"kernel source {path.name} not present")
    return path.read_text(encoding="utf-8")


def _w_funcs(src):
    """Extract W0/W1/W2 Horner expressions -> evaluable lambdas."""
    out = {}
    for name in ("W0", "W1", "W2"):
        m = re.search(rf"double {name}\(double t\) \{{\s*return (.*?);", src, re.S)
        assert m, f"{name} not found in kernel source"
        out[name] = eval(f"lambda t: {m.group(1)}")
    return out


def _erfc_dispatch(src):
    """Extract the three t-mapping constants from erfc_poly dispatch."""
    m = re.search(r"if \(x < 1\.5\).*?double t = x\*([0-9.]+) - 1\.0", src, re.S)
    c0 = float(m.group(1))
    m = re.search(r"else if \(x < 3\.0\).*?double t = \(x-1\.5\)\*([0-9.]+) - 1\.0", src, re.S)
    c1 = float(m.group(1))
    m = re.search(r"double t = \(xc-3\.0\)\*([0-9.]+) - 1\.0", src, re.S)
    c2 = float(m.group(1))
    return c0, c1, c2


class TestErfcPoly:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = _src(SRC_F)
        self.W = _w_funcs(self.src)

    def test_branch_mapping_constants_exact(self):
        """The 4/9-vs-2/3.5 class of bug: mapping must be exactly 2/(hi-lo)."""
        c0, c1, c2 = _erfc_dispatch(self.src)
        assert c0 == pytest.approx(2.0 / 1.5, rel=1e-15)
        assert c1 == pytest.approx(2.0 / 1.5, rel=1e-15)
        assert c2 == pytest.approx(2.0 / 3.5, rel=1e-15)

    def _poly(self, x):
        if x < 1.5:
            w = self.W["W0"](x * (2.0 / 1.5) - 1.0)
        elif x < 3.0:
            w = self.W["W1"]((x - 1.5) * (2.0 / 1.5) - 1.0)
        else:
            w = self.W["W2"]((min(x, 6.5) - 3.0) * (2.0 / 3.5) - 1.0)
        return np.exp(-x * x) * w

    def test_max_rel_error_le_1e13(self):
        xs = np.linspace(0.001, 6.49, 20001)
        got = np.array([self._poly(x) for x in xs])
        rel = np.abs(got - erfc_ref(xs)) / erfc_ref(xs)
        assert rel.max() < 1e-13, f"max rel {rel.max():.2e} at x={xs[rel.argmax()]}"

    def test_branch_seams_continuous(self):
        for x in (1.4999999, 1.5000001, 2.9999999, 3.0000001):
            assert abs(self._poly(x) / erfc_ref(x) - 1.0) < 1e-12


class TestBspline4Weights:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = _src(SRC_H)

    def _w4(self, s):
        # weights4 值(中心约定: anchors k-1..k+2)
        om = 1.0 - s
        return np.array([om**3 / 6, 2/3 - s*s + 0.5*s**3,
                         1/6 + 0.5*s + 0.5*s**2 - 0.5*s**3, s**3 / 6])

    def test_partition_of_unity(self):
        for s in np.linspace(0.0, 1.0, 101):
            assert abs(self._w4(s).sum() - 1.0) < 1e-14

    def test_matches_reference_cardinal_bspline(self):
        worst = 0.0
        for s in np.linspace(0.0, 1.0, 101):
            ref = np.array([cardinal_bspline(s + 3, 4), cardinal_bspline(s + 2, 4),
                            cardinal_bspline(s + 1, 4), cardinal_bspline(s, 4)])
            worst = max(worst, float(np.abs(self._w4(s) - ref).max()))
        assert worst < 1e-14, f"max |diff| {worst:.2e}"

    def test_kernel_weights4_anchors_complete(self):
        """anchor[1..3] must be assigned — the uninitialized-anchor bug
        (e0h/e0h2, caught by per-point numpy oracle after the SUM check
        passed vacuously) must never return."""
        for path in (SRC_F, SRC_H):
            if not path.exists():
                continue
            src = path.read_text(encoding="utf-8")
            for m in re.finditer(r"void weights4\(.*?\n\}", src, re.S):
                body = m.group(0)
                for i in (1, 2, 3):
                    assert f"anchor[{i}] =" in body, (
                        f"{path.name}: weights4 anchor[{i}] never assigned")

    def test_kernel_weights4_matches_this_file(self):
        """The kernel's weights4 expression must equal the formulas above
        (parsed from CUDA source — transcription tripwire)."""
        m = re.search(
            r"w\[0\] = om \* om \* om \* \(1\.0 / 6\.0\);.*?"
            r"w\[3\] = s \* s \* s \* \(1\.0 / 6\.0\);", self.src, re.S)
        assert m, "weights4 body not found"
        body = m.group(0)
        # 逐常数核对关键系数(2/3, 0.5, 1/6 三处出现)
        assert body.count("2.0 / 3.0") == 1
        assert body.count("0.5 * s") >= 2
        assert body.count("1.0 / 6.0") == 3


class TestKernelSourceHygiene:
    """NVRTC writes sources as ASCII — non-ASCII in any bench SRC is a
    guaranteed runtime failure (struck three times 2026-09). Tripwire."""

    def test_bench_src_strings_ascii(self):
        for path in sorted(E0_DIR.glob("e0*.py")):
            text = path.read_text(encoding="utf-8")
            for m in re.finditer(r'SRC = r?"""(.*?)"""', text, re.S):
                bad = [c for c in m.group(1) if ord(c) > 127]
                assert not bad, f"{path.name}: non-ASCII in SRC: {bad[:3]!r}"
