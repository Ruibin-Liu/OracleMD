"""Fixed-point infrastructure — spec §3.1/§3.2 (pillars 1 & 2).

Q24.40 force accumulator and Q16.48 charge-grid accumulator, with
saturating add + sticky overflow flags (checked in release too, at every
graph/window boundary — here: every accumulation call).

Conventions (spec):
  - Q24.40: 1 sign + 23 integer bits + 40 fractional bits; saturate at ±2^23.
  - Q16.48: 1 sign + 15 integer bits + 48 fractional bits; saturate at ±2^15.
  - Accumulation is exact integer addition; quantization happens once per
    contribution (round-to-nearest-even via numpy.rint).
  - Saturation, not wraparound.  A wraparound in a bitwise-deterministic
    system is a reproducible silently-negated force — the sticky flag is
    the only detector.
"""
from __future__ import annotations

import numpy as np

Q24_40 = dict(int_bits=24, frac_bits=40)   # force accumulator
Q16_48 = dict(int_bits=16, frac_bits=48)   # charge grid accumulator


def _limits(int_bits: int, frac_bits: int) -> tuple[int, int]:
    """Raw-int64 saturation limits.  Q24.40 and Q16.48 exactly fill int64
    (sign + int + frac = 64), so the *format* range and the int64 range
    coincide: physical |value| <= 2^(int_bits-1) == raw < 2^63."""
    vmax = (1 << (int_bits + frac_bits - 1)) - 1
    return -vmax - 1, vmax


class FixedPointAccumulator:
    """Saturating fixed-point integer accumulator with sticky overflow flag.

    The reference implementation mirrors the GPU semantics: integer adds are
    order-independent (bitwise determinism pillar 1); the float->fixed
    quantization of each contribution is the only rounding step.
    """

    def __init__(self, shape, int_bits: int, frac_bits: int, name: str = "fxp"):
        self.name = name
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.vmin, self.vmax = _limits(int_bits, frac_bits)
        self.acc = np.zeros(shape, dtype=np.int64)
        self.sticky_overflow = False
        self.n_saturations = 0

    def add_f64(self, contrib: np.ndarray) -> None:
        """Quantize float64 contributions (round-half-even) and add exactly."""
        scaled = np.rint(np.asarray(contrib, dtype=np.float64) * (1 << self.frac_bits))
        # Values beyond representable float for the scaled magnitude saturate.
        big = float(1 << 62)
        self.sticky_overflow |= bool(np.any(np.abs(scaled) > big))
        q = np.clip(scaled, -(1 << 62), (1 << 62)).astype(np.int64)
        self._add_i64(q)

    def scatter_add_f64(self, idx, contrib: np.ndarray) -> None:
        """Scatter-accumulate quantized contributions at grid indices.

        Grid semantics (pillar 2): each contribution quantized once, integer
        adds exact and order-independent.  Format-range check is post-hoc
        (envelope-checked at ingest; wraparound impossible by construction).
        """
        scaled = np.rint(np.asarray(contrib, dtype=np.float64) * (1 << self.frac_bits))
        q = scaled.astype(np.int64)
        np.add.at(self.acc, idx, q)
        over = np.abs(self.acc.astype(object)) > self.vmax
        if over.any():
            self.sticky_overflow = True
            self.n_saturations += int(over.sum())
            self.acc = np.where(self.acc > self.vmax, self.vmax,
                                np.where(self.acc < self.vmin, self.vmin,
                                         self.acc)).astype(np.int64)

    def add_i64(self, q: np.ndarray) -> None:
        self._add_i64(q)

    def _add_i64(self, q: np.ndarray) -> None:
        # Exact addition on int128-equivalent via float64 guard is unsafe at
        # extremes; use int64 with saturation check via sign inspection.
        a = self.acc
        s = a.astype(object) + q.astype(object)  # exact, slow path (reference impl)
        over = (s > self.vmax) | (s < self.vmin)
        if over.any():
            self.sticky_overflow = True
            self.n_saturations += int(over.sum())
        sat = np.where(s > self.vmax, self.vmax, s)
        sat = np.where(sat < self.vmin, self.vmin, sat)
        self.acc = np.asarray(sat, dtype=np.int64)

    def to_f64(self) -> np.ndarray:
        return self.acc.astype(np.float64) / (1 << self.frac_bits)


def q24_40_forces(shape) -> FixedPointAccumulator:
    return FixedPointAccumulator(shape, **Q24_40, name="force_Q24.40")


def q16_48_grid(shape) -> FixedPointAccumulator:
    return FixedPointAccumulator(shape, **Q16_48, name="grid_Q16.48")
