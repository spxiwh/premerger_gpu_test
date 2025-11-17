import sys
import time
import numpy as np
import cupy as cp

# Import reference and optimized implementations
import run_single_likelihood_batch as ref
import run_single_likelihood_batch_optimized as opt
import run_single_likelihood_batch_optimized2 as opt2


def build_shared_context():
    ctx = {}
    ctx['tlen'] = 2592000
    ctx['sample_rate'] = 0.2
    ctx['delta_f'] = 1.0 / ctx['tlen']
    ctx['delta_t'] = 5
    ctx['flen'] = ctx['tlen'] // 2 + 1
    ctx['cutoff_time'] = 86400 * 7
    ctx['kernel_length'] = 17280
    ctx['extra_forward_zeroes'] = 8640
    ctx['data_file'] = 'signal_0.hdf'
    ctx['psd_file'] = 'model_AE_TDI1_SMOOTH_optimistic.txt.gz'

    # Initialize data/PSDs/FD data
    ref.initialization(ctx)

    # Waveform/global params
    ctx['tdi'] = '1.5'
    ctx['t_obs_start'] = ctx['tlen']
    ctx['f_final'] = ctx['sample_rate'] / 2
    ctx['approximant'] = 'BBHX_PhenomD'
    ctx['t_offset'] = 7365189.431698299
    ctx['mode_array'] = [(2, 2)]
    return ctx


def main():
    B = 100
    N_RUNS = 50

    # Shared context and params
    shared_ctx = build_shared_context()

    # Use the same parameter builder as optimized to ensure identical batches
    params = opt.build_params(B)

    # Run reference N_RUNS times
    ctx_ref = dict(shared_ctx)
    t0 = time.perf_counter()
    for i in range(N_RUNS):
        ref.log_likelihood(params, ctx_ref)
    cp.cuda.Stream.null.synchronize()
    t_ref = (time.perf_counter() - t0) * 1000.0

    # Run optimized N_RUNS times
    ctx_opt = dict(shared_ctx)
    t1 = time.perf_counter()
    for i in range(N_RUNS):
        opt.log_likelihood_optimized(params, ctx_opt)
    cp.cuda.Stream.null.synchronize()
    t_opt = (time.perf_counter() - t1) * 1000.0

    # Run optimized2 N_RUNS times
    ctx_opt2 = dict(shared_ctx)
    t2 = time.perf_counter()
    for i in range(N_RUNS):
        opt2.log_likelihood_optimized2(params, ctx_opt2)
    cp.cuda.Stream.null.synchronize()
    t_opt2 = (time.perf_counter() - t2) * 1000.0

    # Compare
    ll_ref = cp.asnumpy(ctx_ref['log_likelihoods'])
    ll_opt = cp.asnumpy(ctx_opt['log_likelihoods'])
    ll_opt2 = cp.asnumpy(ctx_opt2['log_likelihoods'])

    ll_ref = cp.asnumpy(ctx_ref['log_likelihoods'])
    ll_opt = cp.asnumpy(ctx_opt['log_likelihoods'])
    ll_opt2 = cp.asnumpy(ctx_opt2['log_likelihoods'])

    abs_diff = np.abs(ll_opt - ll_ref)
    denom = np.maximum(np.abs(ll_ref), 1.0)
    rel_diff = abs_diff / denom

    abs_diff2 = np.abs(ll_opt2 - ll_ref)
    rel_diff2 = abs_diff2 / denom

    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())
    mean_rel = float(rel_diff.mean())

    max_abs2 = float(abs_diff2.max())
    max_rel2 = float(rel_diff2.max())
    mean_rel2 = float(rel_diff2.mean())

    print(f"Reference total time ({N_RUNS} runs): {t_ref:.1f} ms ({t_ref/N_RUNS:.2f} ms/run)")
    print(f"Optimized total time ({N_RUNS} runs): {t_opt:.1f} ms ({t_opt/N_RUNS:.2f} ms/run)")
    print(f"Optimized2 total time ({N_RUNS} runs): {t_opt2:.1f} ms ({t_opt2/N_RUNS:.2f} ms/run)")
    print(f"Speedup (opt1): {t_ref/t_opt:.2f}x")
    print(f"Speedup (opt2): {t_ref/t_opt2:.2f}x")
    print()
    print("Optimized1 vs Reference:")
    print(f"  Max abs diff: {max_abs:.3e}")
    print(f"  Max rel diff: {max_rel:.3e}")
    print(f"  Mean rel diff: {mean_rel:.3e}")
    print()
    print("Optimized2 vs Reference:")
    print(f"  Max abs diff: {max_abs2:.3e}")
    print(f"  Max rel diff: {max_rel2:.3e}")
    print(f"  Mean rel diff: {mean_rel2:.3e}")

    # Tolerances (float64 equivalence)
    atol = 1e-12
    rtol = 1e-12

    ok1 = np.allclose(ll_ref, ll_opt, rtol=rtol, atol=atol)
    ok2 = np.allclose(ll_ref, ll_opt2, rtol=rtol, atol=atol)
    
    print()
    print("RESULT opt1:", "PASS" if ok1 else "FAIL")
    print("RESULT opt2:", "PASS" if ok2 else "FAIL")

    if not (ok1 and ok2):
        if not ok1:
            idx = np.where(~np.isclose(ll_ref, ll_opt, rtol=rtol, atol=atol))[0]
            print("Opt1 mismatch indices (first 10):", idx[:10])
        if not ok2:
            idx = np.where(~np.isclose(ll_ref, ll_opt2, rtol=rtol, atol=atol))[0]
            print("Opt2 mismatch indices (first 10):", idx[:10])
        sys.exit(1)


if __name__ == '__main__':
    main()
