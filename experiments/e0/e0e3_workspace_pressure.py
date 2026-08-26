import numpy as np, cupy as cp

def compare(a, b, label):
    an, bn = cp.asnumpy(a), cp.asnumpy(b)
    bit_eq = np.array_equal(an.view(np.int64), bn.view(np.int64))
    nd = np.count_nonzero(an.view(np.int64)!=bn.view(np.int64))
    print(f"{label:56s} bitwise={'EQ ' if bit_eq else 'DIFF'} diff={nd}/{a.size}")

rng = np.random.default_rng(7)

def run_pair(tag):
    # 128^3 batch=24 vs 24 singles (c2c), plus 64^3 batch=48 vs singles
    for n, R in ((128, 24), (64, 48)):
        base = [rng.standard_normal((n,n,n)) + 1j*rng.standard_normal((n,n,n)) for _ in range(R)]
        singles = np.stack([cp.asnumpy(cp.fft.fftn(cp.asarray(b[None]), axes=(1,2,3))[0]) for b in base])
        bat = cp.asnumpy(cp.fft.fftn(cp.asarray(np.stack(base)), axes=(1,2,3)))
        compare(singles, bat, f"{tag} c2c {n}^3: {R} singles vs batch={R}")
        cp.get_default_memory_pool().free_all_blocks()

free0, _ = cp.cuda.Device().mem_info
print(f"free at start: {free0/2**30:.2f} GiB")
run_pair("[no-pressure]")

# squeeze: leave only ~1.5 GiB free
free, _ = cp.cuda.Device().mem_info
hog_bytes = int(free - int(1.5 * 2**30))
hog = cp.empty(hog_bytes // 8, dtype=cp.float64)
free2, _ = cp.cuda.Device().mem_info
print(f"free under pressure: {free2/2**30:.2f} GiB")
cp.get_default_memory_pool().free_all_blocks()  # release cached blocks but keep hog
free3, _ = cp.cuda.Device().mem_info
print(f"free after pool purge (hog kept): {free3/2**30:.2f} GiB")
run_pair("[mem-pressure]")

# drop cufft plan cache and rerun under pressure (force re-plan with low mem)
try:
    from cupy.cuda import cufft
    cufft.clear_plan_cache() if hasattr(cufft, "clear_plan_cache") else None
except Exception as e:
    print("plan cache clear:", e)
run_pair("[mem-pressure, replanned]")
print("done.")
