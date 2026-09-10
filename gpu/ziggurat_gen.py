"""Generate gpu/ziggurat_constants.json from numpy's ziggurat_constants.h.

Constant discipline (trap class 8, same institution as erfc_coeffs.json):
the 256-level ziggurat tables are PARSED programmatically from numpy's
source header (pinned version tag) -- never transcribed by hand.  The
CUDA/Python RNG implementations (gpu/rand.py) consume this json.

Source: numpy/random/src/distributions/ziggurat_constants.h @ the numpy
version pinned in pyproject (numpy>=2.0); regenerate with:
    uv run python gpu/ziggurat_gen.py <path-to-ziggurat_constants.h>

The tables are algorithm constants (Marsaglia-Tsang 256-level 64-bit
ziggurat, randgen lineage) -- invariant across numpy releases; the parse
is validated by count + monotonicity + a bitwise stream check in
tests/test_gpu_rand.py.
"""
from __future__ import annotations

import json
import re
import sys


def parse_header(text: str) -> dict:
    def u64_array(name: str) -> list[int]:
        m = re.search(rf"static const uint64_t {name}\[\] = \{{(.*?)\}};",
                      text, re.S)
        vals = re.findall(r"0x([0-9A-Fa-f]+)ULL", m.group(1))
        assert len(vals) == 256, f"{name}: {len(vals)} entries"
        return [int(v, 16) for v in vals]

    def f64_array(name: str) -> list[float]:
        m = re.search(rf"static const double {name}\[\] = \{{(.*?)\}};",
                      text, re.S)
        vals = re.findall(r"(-?\d+\.\d+e[+-]\d+)", m.group(1))
        assert len(vals) == 256, f"{name}: {len(vals)} entries"
        return [float(v) for v in vals]

    m = re.search(r"static const double ziggurat_nor_r = ([0-9.]+);", text)
    nor_r = float(m.group(1))
    m = re.search(r"static const double ziggurat_nor_inv_r =\s*\n?\s*"
                  r"([0-9.]+);", text)
    nor_inv_r = float(m.group(1))
    return {
        "source": "numpy/random/src/distributions/ziggurat_constants.h",
        "ki": u64_array("ki_double"),
        "wi": f64_array("wi_double"),
        "fi": f64_array("fi_double"),
        "nor_r": nor_r,
        "nor_inv_r": nor_inv_r,
    }


def validate(data: dict) -> None:
    ki, wi, fi = data["ki"], data["wi"], data["fi"]
    # structural invariants (ki is bell-shaped over strips; wi[0] is the
    # special base-strip entry scaled by 2^-53 -- do not over-assume)
    assert ki[1] == 0, "ki[1] must be 0 (strip 1 accepts everything)"
    assert all(wi[i] < wi[i + 1] for i in range(1, 255)), \
        "wi increases over strips 1..255"
    assert all(fi[i] > fi[i + 1] for i in range(255)), \
        "fi strictly decreasing (Gaussian strip heights)"
    assert fi[0] == 1.0 and fi[255] < 0.01
    assert abs(data["nor_inv_r"] * data["nor_r"] - 1.0) < 1e-15
    assert 3.0 < data["nor_r"] < 4.0


def main() -> None:
    header_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        __file__.replace("ziggurat_gen.py", "ziggurat_constants.json")
    text = open(header_path, encoding="utf-8").read()
    data = parse_header(text)
    validate(data)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print(f"wrote {out_path}: ki/wi/fi x256, nor_r={data['nor_r']!r}")


if __name__ == "__main__":
    main()
