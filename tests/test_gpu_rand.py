"""Pillar-3 RNG CI: philox4x64_10 + ziggurat mirrors vs numpy / opus.

The gpu O-step (gpu/integrate.py) calls the CUDA transcription of these
mirrors (gpu/rand.py emit_cuda); local bitwise validation against numpy's
own Generator and opus.rng.gauss_stream is the semantic layer; the pod
probes device log1p/exp parity before claiming gamma>0 trajectory bitwise.
"""
import numpy as np
import pytest

from gpu import rand


class TestPhiloxMirror:
    @pytest.mark.parametrize("key", [1, 7, 12345, 0xDEADBEEFCAFEF00D])
    def test_raw_stream_bitwise_vs_numpy(self, key):
        ref = np.random.Philox(key=key).random_raw(16)
        stream = rand.PhiloxStream(key)
        got = [stream.next_u64() for _ in range(16)]
        assert got == [int(v) for v in ref]

    def test_counter_carry(self):
        """Counter carry chain: burn 4*3 blocks and keep matching."""
        key = 99
        ref = np.random.Philox(key=key).random_raw(4 * 3 + 2)
        stream = rand.PhiloxStream(key)
        got = [stream.next_u64() for _ in range(4 * 3 + 2)]
        assert got == [int(v) for v in ref]


class TestZigguratMirror:
    @pytest.mark.parametrize("key", [1, 7, 12345, 999999937])
    def test_standard_normal_bitwise_vs_numpy(self, key):
        gen = np.random.Generator(np.random.Philox(key=key))
        ref = gen.standard_normal(64)
        ns = rand.NormalStream(key)
        got = np.array([ns.standard_normal() for _ in range(64)])
        assert (got.view(np.int64) == ref.view(np.int64)).all()

    def test_draw_budget_bounded(self):
        """Tail/wedge paths consume extra doubles; consumption stays
        plausible (sanity: 128 draws never need > 8 raw blocks per draw)."""
        ns = rand.NormalStream(31337)
        for _ in range(128):
            ns.standard_normal()
        blocks, draws = ns.phil.pos + 4 * (ns.phil.ctr[0] - 1) or 4, 128
        assert blocks / draws < 8


class TestGaussStreamMirror:
    @pytest.mark.parametrize("tuple_", [
        (42, 0, 1, 0, 0), (42, 7, 33, 0, 2), (7, 1000, 59, 0, 1),
        (123456789, 5, 0, 0, 0), (1, 1, 1, 1, 1),
    ])
    def test_bitwise_vs_opus(self, tuple_):
        from opus.rng import gauss_stream
        gs, st, at, sl, dof = tuple_
        ref = gauss_stream(gs, st, at, sl, dof, 8)
        got = rand.gauss_stream_mirror(gs, st, at, sl, dof, 8)
        assert (got.view(np.int64) == ref.view(np.int64)).all()

    def test_independence(self):
        """Different tuples => independent streams (composition contract)."""
        a = rand.gauss_stream_mirror(42, 0, 1, 0, 0, 1)
        b = rand.gauss_stream_mirror(42, 0, 1, 0, 1, 1)
        c = rand.gauss_stream_mirror(42, 1, 1, 0, 0, 1)
        assert a[0] != b[0] and a[0] != c[0]


class TestCudaSource:
    def test_emit_ascii_and_structure(self):
        src = rand.emit_cuda()
        assert "__device__" in src
        assert all(ord(ch) < 128 for ch in src)

    def test_constants_not_retyped_in_json(self):
        """The tables must come from the generated json, not literals."""
        src = open(rand.__file__, encoding="utf-8").read()
        assert "0x000EF33D" not in src  # ki[0] literal would be a retype
        assert "8.68362706080130616677e-16" not in src  # wi[0]
        json_data = rand._TABLES
        assert len(json_data["ki"]) == 256 and len(json_data["wi"]) == 256 \
            and len(json_data["fi"]) == 256
