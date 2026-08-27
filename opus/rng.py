"""Counter-based RNG — spec pillar 3.

Seed = (global_seed, step, stable_atom_id, slot, dof) for the Langevin O-step;
host-side decisions use (global_seed, step, move_type, pair_id).

Implementation: numpy Philox with counter composed from the seed tuple.
Counter-based means:
  - no state carried between draws (run-to-run and cross-config reproducible);
  - any (seed-tuple) -> stream mapping is stable;
  - atom reorderings remap streams by *stable atom id*, not array index (M6b).

The reference implementation exposes the stream composition so the GPU
kernel (later) reproduces bit-identical streams only up to Philox output
mapping — the M0 contract is the *composition* semantics: same tuple =>
same stream; different tuple => independent stream.
"""
from __future__ import annotations

import numpy as np


def _mix(seed: int, a: int, b: int, c: int, d: int) -> int:
    """64-bit mixer combining counter fields (splitmix-style finalizer)."""
    x = (seed & 0xFFFFFFFFFFFFFFFF)
    x = (x ^ (a * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 31) ^ (b * 0x8B72C5AF1A3F1E2D)
            ^ ((c << 21) & 0xFFFFFFFFFFFFFFFF)
            ^ (d * 0xC2B2AE3D27D4EB4F)) & 0xFFFFFFFFFFFFFFFF
    return x


def gauss_stream(global_seed: int, step: int, atom_id: int, slot: int,
                 dof: int, n: int) -> np.ndarray:
    """n standard-normal doubles for one (atom, slot, dof) at one step."""
    key = _mix(global_seed, step + 1, atom_id + 1, slot + 1, dof + 1)
    rng = np.random.Generator(np.random.Philox(key=key))
    return rng.standard_normal(n)


def uniform_stream(global_seed: int, step: int, move_type: int, pair_id: int,
                   n: int = 1) -> np.ndarray:
    """Host-side decision stream (HREX/barostat/ALF judgments)."""
    key = _mix(global_seed, step + 1, move_type + 1, pair_id + 1, 0x5DEECE66D)
    rng = np.random.Generator(np.random.Philox(key=key))
    return rng.random(n)
