"""Counter RNG (pillar 3): numpy-Philox/ziggurat mirrors + CUDA emitter.

Target semantics: opus.rng.gauss_stream = fresh numpy Generator(Philox(key))
per (global_seed, step, atom_id, slot, dof) tuple, standard_normal(n).
This module reproduces that stream BITWISE:

  - philox4x64_10: Random123 round function per numpy/random/src/philox/
    philox.h (10 rounds, M0/M1/W0/W1 constants; stream counter starts at
    0 and increments BEFORE the block, buffer of 4 uint64 per block);
  - normal: numpy random_standard_normal ziggurat (256-level, r = e3n52sb8
    decomposition: idx = r & 0xff, sign = (r >> 8) & 1, rabs = (r >> 9) &
    0xfffffffffffff; tail strip idx==0 via log1p; wedge via fi ladder);
  - keying: opus.rng._mix(global_seed, step+1, atom+1, slot+1, dof+1)
    (64-bit splitmix-style finalizer) -- same tuple => same stream.

Tables come from gpu/ziggurat_constants.json (programmatic parse of
numpy's ziggurat_constants.h -- gpu/ziggurat_gen.py; never hand-copied).

Cross-platform note: the ziggurat tail/wedge use log1p/exp.  Python and
CUDA implementations are validated against their own libms; the pod
alignment probes log1p/exp parity on device before any gamma>0 bitwise
claim (e0-pattern report records the probe result).
"""
from __future__ import annotations

import json
import os

import numpy as np

_M64 = (1 << 64) - 1
_K0B, _K1B = 0x9E3779B97F4A7C15, 0xBB67AE8584CAA73B
_PM0, _PM1 = 0xD2E7470EE14C6C93, 0xCA5A826395121157
_NP53 = 1.0 / 9007199254740992.0  # 2^-53


def _load_tables() -> dict:
    path = os.path.join(os.path.dirname(__file__), "ziggurat_constants.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_TABLES = _load_tables()


# --------------------------------------------------------- philox mirror

def _mulhilo(a: int, b: int) -> tuple[int, int]:
    p = a * b
    return p & _M64, p >> 64


def _philox_round(ctr, key):
    lo0, hi0 = _mulhilo(_PM0, ctr[0])
    lo1, hi1 = _mulhilo(_PM1, ctr[2])
    return (hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0)


def philox4x64_10(ctr, key):
    """One block: 10 rounds, first with the raw key (Random123 layout)."""
    for rnd in range(10):
        if rnd:
            key = ((key[0] + _K0B) & _M64, (key[1] + _K1B) & _M64)
        ctr = _philox_round(ctr, key)
    return ctr


class PhiloxStream:
    """numpy Philox state machine mirror (counter starts at 0; each
    philox_next increments counter[0] with carry, then emits buffer)."""

    def __init__(self, key: int):
        self.ctr = [0, 0, 0, 0]
        self.key = (key & _M64, 0)
        self.buf = [0, 0, 0, 0]
        self.pos = 4  # force generation on first next()

    def next_u64(self) -> int:
        if self.pos < 4:
            out = self.buf[self.pos]
            self.pos += 1
            return out
        c = self.ctr
        c[0] = (c[0] + 1) & _M64
        if c[0] == 0:
            c[1] = (c[1] + 1) & _M64
            if c[1] == 0:
                c[2] = (c[2] + 1) & _M64
                if c[2] == 0:
                    c[3] = (c[3] + 1) & _M64
        blk = philox4x64_10(tuple(c), self.key)
        self.buf = list(blk)
        self.pos = 1
        return self.buf[0]


# -------------------------------------------------------- ziggurat mirror

class NormalStream:
    """numpy random_standard_normal mirror over a PhiloxStream (1 uint64
    per draw on the 99.3% path; strip-0 tail and fi-wedge consume extra
    doubles exactly as distributions.c)."""

    def __init__(self, key: int):
        self.phil = PhiloxStream(key)
        self.wi = _TABLES["wi"]
        self.ki = _TABLES["ki"]
        self.fi = _TABLES["fi"]
        self.nor_r = _TABLES["nor_r"]
        self.nor_inv_r = _TABLES["nor_inv_r"]

    def _next_double(self) -> float:
        return (self.phil.next_u64() >> 11) * _NP53

    def standard_normal(self) -> float:
        wi, ki, fi = self.wi, self.ki, self.fi
        while True:
            r = self.phil.next_u64()
            idx = r & 0xFF
            r >>= 8
            sign = r & 0x1
            rabs = (r >> 1) & 0x000FFFFFFFFFFFFF
            x = rabs * wi[idx]
            if sign & 0x1:
                x = -x
            if rabs < ki[idx]:
                return x
            if idx == 0:
                while True:
                    xx = -self.nor_inv_r * np.log1p(-self._next_double())
                    yy = -np.log1p(-self._next_double())
                    if yy + yy > xx * xx:
                        if (rabs >> 8) & 0x1:
                            return -(self.nor_r + xx)
                        return self.nor_r + xx
            else:
                if (fi[idx - 1] - fi[idx]) * self._next_double() + fi[idx] \
                        < np.exp(-0.5 * x * x):
                    return x


# ------------------------------------------------------- opus composition

def gauss_stream_mirror(global_seed: int, step: int, atom_id: int,
                        slot: int, dof: int, n: int) -> np.ndarray:
    """opus.rng.gauss_stream mirror: fresh stream per tuple, n draws."""
    from opus.rng import _mix
    key = _mix(global_seed, step + 1, atom_id + 1, slot + 1, dof + 1)
    ns = NormalStream(key)
    return np.array([ns.standard_normal() for _ in range(n)])


# ------------------------------------------------------------ CUDA source

def emit_cuda() -> str:
    """Device functions: philox4x64_10 + ziggurat normal (distribution.c
    transcription).  Tables are NOT baked in -- the caller passes wi/ki/fi
    device pointers and nor_r/nor_inv_r scalars."""
    return r"""
// numpy Philox4x64-10 (Random123 layout; numpy/random/src/philox/philox.h)
__device__ __forceinline__ void philox4x64_round(
    unsigned long long c0, unsigned long long c1,
    unsigned long long c2, unsigned long long c3,
    unsigned long long k0, unsigned long long k1,
    unsigned long long* out)
{
    unsigned long long lo0 = (0xD2E7470EE14C6C93ULL) * c0;
    unsigned long long hi0 = __umul64hi(0xD2E7470EE14C6C93ULL, c0);
    unsigned long long lo1 = (0xCA5A826395121157ULL) * c2;
    unsigned long long hi1 = __umul64hi(0xCA5A826395121157ULL, c2);
    out[0] = hi1 ^ c1 ^ k0;
    out[1] = lo1;
    out[2] = hi0 ^ c3 ^ k1;
    out[3] = lo0;
}

// one 4x uint64 block; ctr is advanced by the caller (buffer discipline
// lives in the per-thread stream struct below)
__device__ __forceinline__ void philox4x64_10(
    unsigned long long* ctr, unsigned long long k0, unsigned long long k1,
    unsigned long long* out)
{
    // numpy passes the counter BY VALUE (ct = *state->ctr): the caller's
    // ctr must stay the RAW counter.  The first transcription mutated it
    // in place -- block 2+ silently consumed block(1) as the counter
    // (E0o hunt 2026-09-10; pointer-vs-value semantics trap).
    unsigned long long c[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
    unsigned long long kk0 = k0, kk1 = k1, t[4];
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        if (r) {
            kk0 += 0x9E3779B97F4A7C15ULL;
            kk1 += 0xBB67AE8584CAA73BULL;
        }
        philox4x64_round(c[0], c[1], c[2], c[3], kk0, kk1, t);
        c[0] = t[0]; c[1] = t[1]; c[2] = t[2]; c[3] = t[3];
    }
    out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; out[3] = c[3];
}

// per-thread stream state (fresh counter per gauss tuple; buffer of 4)
struct philox_rng {
    unsigned long long ctr[4];
    unsigned long long buf[4];
    int pos;

    __device__ void init(unsigned long long key) {
        ctr[0] = 0ULL; ctr[1] = 0ULL; ctr[2] = 0ULL; ctr[3] = 0ULL;
        pos = 4;
        key_ = key;
    }
    unsigned long long key_;

    __device__ unsigned long long next_u64() {
        if (pos < 4) return buf[pos++];
        ctr[0] += 1ULL;
        if (ctr[0] == 0ULL) {
            ctr[1] += 1ULL;
            if (ctr[1] == 0ULL) {
                ctr[2] += 1ULL;
                if (ctr[2] == 0ULL) ctr[3] += 1ULL;
            }
        }
        unsigned long long out[4];
        philox4x64_10(ctr, key_, 0ULL, out);
        buf[0] = out[0]; buf[1] = out[1]; buf[2] = out[2]; buf[3] = out[3];
        pos = 1;
        return buf[0];
    }
};

// numpy random_standard_normal (ziggurat, distributions.c transcription)
__device__ __noinline__ double ziggurat_normal(
    philox_rng* rng,
    const double* __restrict__ wi, const unsigned long long* __restrict__ ki,
    const double* __restrict__ fi,
    double nor_r, double nor_inv_r)
{
    for (;;) {
        unsigned long long r = rng->next_u64();
        int idx = (int)(r & 0xFFULL);
        r >>= 8;
        int sign = (int)(r & 0x1ULL);
        unsigned long long rabs = (r >> 1) & 0x000FFFFFFFFFFFFFULL;
        double x = (double)rabs * wi[idx];
        if (sign & 0x1) x = -x;
        if (rabs < ki[idx]) return x;  // 99.3%
        if (idx == 0) {
            for (;;) {
                double xx = -nor_inv_r * log1p(-(rng->next_u64() >> 11)
                                               * (1.0 / 9007199254740992.0));
                double yy = -log1p(-(rng->next_u64() >> 11)
                                   * (1.0 / 9007199254740992.0));
                if (yy + yy > xx * xx)
                    return ((rabs >> 8) & 0x1) ? -(nor_r + xx)
                                               : (nor_r + xx);
            }
        } else {
            double u = (rng->next_u64() >> 11) * (1.0 / 9007199254740992.0);
            if ((fi[idx - 1] - fi[idx]) * u + fi[idx]
                < exp(-0.5 * x * x))
                return x;
        }
    }
}

// opus gauss_stream: fresh stream per (seed, step, atom, slot, dof)
__device__ __forceinline__ double gauss_stream1(
    unsigned long long global_seed, unsigned long long step,
    unsigned long long atom_id, unsigned long long slot,
    unsigned long long dof,
    const double* __restrict__ wi, const unsigned long long* __restrict__ ki,
    const double* __restrict__ fi,
    double nor_r, double nor_inv_r)
{
    // opus.rng._mix (splitmix-style finalizer), all mod 2^64
    unsigned long long x = global_seed;
    x ^= (step + 1ULL) * 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31)
        ^ ((atom_id + 1ULL) * 0x8B72C5AF1A3F1E2DULL)
        ^ (((slot + 1ULL) << 21))
        ^ ((dof + 1ULL) * 0xC2B2AE3D27D4EB4FULL);
    philox_rng rng;
    rng.init(x);
    // ziggurat body INLINED (see note above): out-of-line call + caller
    // loop miscompiles at NVRTC (-fmad=false) -- E0o 2026-09-10.
    // The accept loop is BOUNDED (NVRTC also miscompiles the unbounded
    // for(;;) shape at >=2 blocks for specific streams, grid-quiet hang):
    // 100000 consecutive rejections has probability ~0.007^100000 = 0;
    // exhaustion returns 0.0 (deterministic garbage the opus alignment
    // would flag) instead of hanging the grid.
    for (int zzt = 0; zzt < 100000; ++zzt) {
        unsigned long long r = rng.next_u64();
        int idx = (int)(r & 0xFFULL);
        r >>= 8;
        int sign = (int)(r & 0x1ULL);
        unsigned long long rabs = (r >> 1) & 0x000FFFFFFFFFFFFFULL;
        double x = (double)rabs * wi[idx];
        if (sign & 0x1) x = -x;
        if (rabs < ki[idx]) return x;
        if (idx == 0) {
            for (;;) {
                double xx = -nor_inv_r * log1p(-(rng.next_u64() >> 11)
                                               * (1.0 / 9007199254740992.0));
                double yy = -log1p(-(rng.next_u64() >> 11)
                                   * (1.0 / 9007199254740992.0));
                if (yy + yy > xx * xx)
                    return ((rabs >> 8) & 0x1) ? -(nor_r + xx)
                                               : (nor_r + xx);
            }
        } else {
            double u = (rng.next_u64() >> 11) * (1.0 / 9007199254740992.0);
            if ((fi[idx - 1] - fi[idx]) * u + fi[idx]
                < exp(-0.5 * x * x))
                return x;
        }
    }
    return 0.0;  // exhaustion: unreachable; deterministic sentinel
}
"""


# ------------------------------------------------------- device probe

def probe_kernel_source() -> str:
    """Standalone kernel: one stream per thread (key = threadIdx), n draws
    each -- device-vs-host ziggurat parity probe + log1p/exp discipline."""
    return emit_cuda() + r"""
// NOTE kernel-arg discipline: all POINTERS first, then scalars grouped by
// alignment -- CUDA packs params at natural alignment, so an int32 placed
// between pointers shifts every later arg by 4 bytes (cupy does not
// auto-align); the misread table pointers hang ziggurat_normal in its
// accept loop (caught by E0o probe 2026-09-10, grid-quiet hang).
extern "C" __global__ void ziggurat_probe(
    double* __restrict__ out,
    const double* __restrict__ wi,
    const unsigned long long* __restrict__ ki,
    const double* __restrict__ fi,
    unsigned long long base_key, int draws,
    double nor_r, double nor_inv_r)
{
    // ziggurat body INLINED per-thread: the out-of-line ziggurat_normal
    // call inside a caller loop miscompiles at NVRTC for specific streams
    // (grid-quiet hang; E0o 2026-09-10) -- inline form verified working
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    philox_rng rng;
    rng.init(base_key + (unsigned long long)t);
    double acc = 0.0;
    for (int i = 0; i < draws; ++i) {
        for (;;) {
            unsigned long long r = rng.next_u64();
            int idx = (int)(r & 0xFFULL);
            r >>= 8;
            int sign = (int)(r & 0x1ULL);
            unsigned long long rabs = (r >> 1) & 0x000FFFFFFFFFFFFFULL;
            double x = (double)rabs * wi[idx];
            if (sign & 0x1) x = -x;
            if (rabs < ki[idx]) { acc += x; break; }
            if (idx == 0) {
                for (;;) {
                    double xx = -nor_inv_r * log1p(-(rng.next_u64() >> 11)
                                                   * (1.0 / 9007199254740992.0));
                    double yy = -log1p(-(rng.next_u64() >> 11)
                                       * (1.0 / 9007199254740992.0));
                    if (yy + yy > xx * xx) {
                        acc += ((rabs >> 8) & 0x1) ? -(nor_r + xx)
                                                    : (nor_r + xx);
                        break;
                    }
                }
            } else {
                double u = (rng.next_u64() >> 11)
                    * (1.0 / 9007199254740992.0);
                if ((fi[idx - 1] - fi[idx]) * u + fi[idx]
                    < exp(-0.5 * x * x)) {
                    acc += x;
                    break;
                }
            }
        }
    }
    out[t] = acc;
}
"""


def device_probe(n_streams: int = 65536, draws: int = 8):
    """Device ziggurat vs host mirror over n_streams independent streams.

    Returns (n_bitwise_mismatch, n_streams, host_out, dev_out)."""
    import cupy as cp
    src = probe_kernel_source()
    mod = cp.RawModule(code=src, options=("-fmad", "false"))
    k = mod.get_function("ziggurat_probe")
    t = _rng_tables_pub()
    out = cp.zeros(n_streams)
    k(((n_streams + 255) // 256,), (256,),
      (out, t["wi"], t["ki"], t["fi"],
       np.uint64(0x5EED_0000_0000_0001), np.int32(draws),
       t["nor_r"], t["nor_inv_r"]))
    cp.cuda.Stream.null.synchronize()
    dev = cp.asnumpy(out)
    # reference = numpy itself (Generator, C speed); the python mirror's
    # bitwise parity with numpy is already CI-proven, so this compares the
    # DEVICE against the same ground truth
    host = np.empty(n_streams)
    for sidx in range(n_streams):
        gen = np.random.Generator(
            np.random.Philox(key=0x5EED_0000_0000_0001 + sidx))
        host[sidx] = gen.standard_normal(draws).sum()
    mism = int((host.view(np.int64) != dev.view(np.int64)).sum())
    return mism, n_streams, host, dev


_TABLES_DEV: dict = {}


def _rng_tables_pub():
    import cupy as cp
    if "wi" not in _TABLES_DEV:
        _TABLES_DEV["wi"] = cp.asarray(np.array(_TABLES["wi"], dtype=np.float64))
        _TABLES_DEV["ki"] = cp.asarray(np.array(_TABLES["ki"], dtype=np.uint64))
        _TABLES_DEV["fi"] = cp.asarray(np.array(_TABLES["fi"], dtype=np.float64))
        _TABLES_DEV["nor_r"] = float(_TABLES["nor_r"])
        _TABLES_DEV["nor_inv_r"] = float(_TABLES["nor_inv_r"])
    return _TABLES_DEV
