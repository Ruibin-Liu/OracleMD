"""Stub for mpiplus (distributed-I/O helpers openmmtools' storage module
imports at module load; the A2 oracle harness never calls them)."""
import sys
import types


def install() -> None:
    if "mpiplus" in sys.modules:
        return
    stub = types.ModuleType("mpiplus")
    stub.__path__ = []
    stub.mpiplus_available = lambda: False
    stub.on_single_node = lambda rank=0, comm=None: True
    stub.get_mpicomm = lambda: None
    stub.share_with_nonmpi_nodes = lambda *a, **k: None
    stub.size = 1
    stub.rank = 0
    stub.initialize = lambda: None
    stub.finalize = lambda: None
    def _universal(*a, **k):
        if a and callable(a[0]):
            return a[0]     # decorator pass-through: never execute
        return _universal

    for _name in ("run_single", "distribute", "collect",
                  "share_with_nonmpi_nodes", "on_single_node",
                  "broadcast_result", "share_state", "delayed_termination",
                  "node_synchronize", "_mpiabort_on_exception"):
        setattr(stub, _name, _universal)
    stub.get_local_rank = lambda comm=None: 0
    sys.modules["mpiplus"] = stub
