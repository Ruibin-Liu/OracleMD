"""Energy accumulation — spec §3 'energy_accum: f64 固定形状树形归约 或 Q34.30'.

The reference implementation uses Q34.30 fixed-point per-component
per-replica registers: every energy contribution is quantized once
(round-half-even) and summed exactly, so component energies are
order-independent (atom reorder / pair-order invariant — M6 energies
bitwise) and deterministic.

Resolution 2^-30 kJ/mol ~ 9.3e-10 — passes A1's abs gate (1e-8) for small
terms and rel gate for large ones.
"""
from __future__ import annotations

import numpy as np

Q34_30 = dict(int_bits=34, frac_bits=30)


class EnergyAccumulator:
    """Per-component, per-replica Q34.30 registers with sticky overflow."""

    def __init__(self, n_replicas: int):
        self.frac_bits = Q34_30["frac_bits"]
        vmax = (1 << (Q34_30["int_bits"] + Q34_30["frac_bits"] - 1)) - 1
        self.vmin, self.vmax = -vmax - 1, vmax
        self.reg: dict[str, np.ndarray] = {}   # component -> int64 (R,)
        self.R = n_replicas
        self.sticky_overflow = False

    def add(self, component: str, values) -> None:
        """values: (R,) float vector or scalar (broadcast over replicas)."""
        v = np.atleast_1d(np.asarray(values, dtype=np.float64))
        if v.shape[0] == 1 and self.R > 1:
            v = np.repeat(v, self.R)
        scaled = np.rint(v * (1 << self.frac_bits))
        bad = ~np.isfinite(scaled) | (np.abs(scaled) > self.vmax)
        if bad.any():
            self.sticky_overflow = True
            scaled = np.where(bad, np.sign(scaled) * self.vmax, scaled)
        q = scaled.astype(np.int64)
        if component not in self.reg:
            self.reg[component] = np.zeros(self.R, dtype=np.int64)
        s = self.reg[component].astype(object) + q.astype(object)
        over = (s > self.vmax) | (s < self.vmin)
        if over.any():
            self.sticky_overflow = True
            s = np.where(s > self.vmax, self.vmax, s)
            s = np.where(s < self.vmin, self.vmin, s)
        self.reg[component] = np.asarray(s, dtype=np.int64)

    def energies(self) -> dict[str, float]:
        """Component sums over replicas, as float (exact int->f64 casts)."""
        return {k: float(np.sum(reg)) / (1 << self.frac_bits)
                for k, reg in self.reg.items()}
