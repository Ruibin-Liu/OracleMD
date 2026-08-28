"""HREX machinery (M3, reference implementation).

Spec §8 semantics:
  - 2 x N independent repeat networks (two chains of λ states); Gibbs
    all-pairs WITHIN a network, never across (spec §8.2);
  - exchange λ (state), never coordinates (§8.1): slots keep x, v; the
    state assignment permutes;
  - exchange step == energy step (B9): swap decisions consume energies of
    the CURRENT conformations under candidate states, from the counter RNG
    (deterministic given (seed, step));
  - u_kn: state-indexed rows, sample-indexed columns (spec §4.4: state
    owns the row label; permutation log records which slot produced each
    sample);
  - MSλD is mutually exclusive with HREX at the config level (§8.2).
"""
from __future__ import annotations

import numpy as np

from . import rng

KB_KJ = 0.00831446261815324


class SlotStateTable:
    """slot = coordinate carrier (never moves); state = λ value (permutes).

    spec §4.4: x, v belong to slot; λ belongs to state; the u_kn row label
    is the state; RNG's replica dimension is the SLOT.
    """

    def __init__(self, lams: np.ndarray, n_networks: int = 2):
        self.lams = np.asarray(lams, dtype=float)
        self.n_states = len(lams)
        self.n_networks = n_networks
        assert self.n_states % n_networks == 0
        # network membership: states are split into contiguous blocks
        self.state_network = np.repeat(
            np.arange(n_networks), self.n_states // n_networks)
        self.slot_to_state = np.arange(self.n_states)   # identity start
        self.permutation_log: list[tuple[int, int, int]] = []
        # (step, slot_i, slot_j) for every accepted swap

    @property
    def slot_lams(self) -> np.ndarray:
        return self.lams[self.slot_to_state]

    def states_in_network(self, net: int) -> np.ndarray:
        return np.where(self.state_network == net)[0]

    def swap(self, slot_i: int, slot_j: int, step: int) -> None:
        si, sj = self.slot_to_state[slot_i], self.slot_to_state[slot_j]
        self.slot_to_state[slot_i] = sj
        self.slot_to_state[slot_j] = si
        self.permutation_log.append((step, slot_i, slot_j))

    def network_of_slot(self, slot: int) -> int:
        return int(self.state_network[self.slot_to_state[slot]])


def gibbs_pairwise_swap(table: SlotStateTable, U: np.ndarray, beta: float,
                        step: int, seed: int) -> int:
    """Gibbs sampling over all slot pairs within each network.

    U[k, s] = energy of slot-s conformation under state-k λ (kT-free,
    kJ/mol).  Deterministic order (network asc, slot pairs asc), counter
    RNG per (step, pair).
    """
    n_swaps = 0
    R = U.shape[1]
    for net in range(table.n_networks):
        slots = [s for s in range(R) if table.network_of_slot(s) == net]
        for a in range(len(slots)):
            for b in range(a + 1, len(slots)):
                si, sj = slots[a], slots[b]
                ki, kj = table.slot_to_state[si], table.slot_to_state[sj]
                # Δ = [U_kj(x_si) + U_ki(x_sj)] − [U_ki(x_si) + U_kj(x_sj)]
                delta = (U[kj, si] + U[ki, sj]) - (U[ki, si] + U[kj, sj])
                u01 = rng.uniform_stream(seed, step, move_type=1,
                                         pair_id=si * R + sj)[0]
                if u01 < np.exp(-beta * min(delta, 0.0)):
                    table.swap(si, sj, step)
                    n_swaps += 1
    return n_swaps


# ---------------------------------------------------------------- u_kn

def ukn_matrix(alch, x: np.ndarray, lams: np.ndarray,
               energy_fn=None) -> np.ndarray:
    """u[k, r] = U(x_r; λ_k) in kJ/mol for every (state, slot-conformation).

    Reference implementation: K x R single-point evaluations (the GPU
    engine uses the C3 quadratic expansion — Q-007; here we take the naive
    path, which doubles as its oracle).
    """
    fn = energy_fn or (lambda lam: lambda xx: _E(alch, xx, lam))
    R = x.shape[1]
    K = len(lams)
    U = np.empty((K, R))
    for k, lam in enumerate(lams):
        f = fn(lam)
        for r in range(R):
            U[k, r] = f(x[:, r:r + 1, :])
    return U


def _E(alch, x, lam):
    from .alchem import single_point_lambda
    return single_point_lambda(alch, x, lam)["E"]


def ukn_schema_save(path, U: np.ndarray, lams: np.ndarray,
                    table: SlotStateTable, step: int, beta: float) -> None:
    """u_kn 落库 schema (spec §8.2/M3): NPZ with
       u_kn[kJ/mol]  lams  state_network  slot_to_state  permutation_log
       step  beta — everything MBAR + variance analysis + provenance need.
    """
    log = np.array(table.permutation_log, dtype=np.int64).reshape(-1, 3)
    np.savez(path, u_kn=U, lams=lams, state_network=table.state_network,
             slot_to_state=table.slot_to_state, permutation_log=log,
             step=np.int64(step), beta=np.float64(beta))


def ukn_schema_load(path):
    d = np.load(path)
    return dict(u_kn=d["u_kn"], lams=d["lams"],
                state_network=d["state_network"],
                slot_to_state=d["slot_to_state"],
                permutation_log=d["permutation_log"],
                step=int(d["step"]), beta=float(d["beta"]))
