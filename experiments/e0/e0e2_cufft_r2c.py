import numpy as np, cupy as cp

def compare(a, b, label):
    an, bn = cp.asnumpy(a), cp.asnumpy(b)
    bit_eq = np.array_equal(an.view(np.int64), bn.view(np.int64))
    diff = np.abs(an - bn); denom = np.maximum(np.abs(an), np.abs(bn))
    rel = np.where(denom > 0, diff/np.maximum(denom,1e-300), 0.0)
    nd = np.count_nonzero(an.view(np.int64)!=bn.view(np.int64))
    print(f"{label:52s} bitwise={'EQ ' if bit_eq else 'DIFF'} maxrel={rel.max():.3e} diff={nd}/{a.size}")
    return bit_eq

rng = np.random.default_rng(123)
for n in (96, 100, 120, 128):
    R = 24
    base = [rng.standard_normal((n,n,n)) for _ in range(R)]
    singles = np.stack([cp.asnumpy(cp.fft.rfftn(cp.asarray(b))) for b in base])
    bat = cp.asnumpy(cp.fft.rfftn(cp.asarray(np.stack(base))))
    compare(singles, bat, f"r2c fwd {n}^3: 24 singles vs batch=24")
    isingles = np.stack([cp.asnumpy(cp.fft.irfftn(cp.asarray(s), s=(n,n,n))) for s in singles])
    ibat = cp.asnumpy(cp.fft.irfftn(cp.asarray(singles), s=(n,n,n)))
    compare(isingles, ibat, f"c2r inv {n}^3: 24 singles vs batch=24")
    cp.get_default_memory_pool().free_all_blocks()
print("done.")
