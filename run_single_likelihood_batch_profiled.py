import time
import cupy as cp
import run_single_likelihood_batch as orig
import run_single_likelihood_batch_optimized as opt
import run_single_likelihood_batch_optimized2 as opt2
from BBHX_Phenom_GPU import _bbhx_fd


def build_params(n: int):
    field_names = [
        'mass1','mass2','spin1z','spin2z',
        'distance','eclipticlongitude','eclipticlatitude',
        'inclination','polarization','coa_phase','tc'
    ]
    p = {name: cp.zeros(n, dtype=cp.float64) for name in field_names}
    p['mass1'][:] = 1000000.0 + cp.arange(n) * 10000.0
    p['mass2'][:] = 1000000.0
    p['spin1z'][:] = cp.linspace(0, 0.1, n)
    p['spin2z'][:] = 0
    p['distance'][:] = 27658.011507544677
    p['eclipticlongitude'][:] = 3.448296944257913
    p['eclipticlatitude'][:] = 0.44491231446252155
    p['inclination'][:] = 0.9238365050097769
    p['polarization'][:] = 3.4236020095353483
    p['coa_phase'][:] = 2.661901610522322
    p['tc'][:] = int(30*86400)
    return p


def gpu_timer():
    start = cp.cuda.Event(); end = cp.cuda.Event()
    def start_rec():
        start.record()
    def stop_ms():
        end.record(); end.synchronize(); return cp.cuda.get_elapsed_time(start, end)
    return start_rec, stop_ms


def log_likelihood_profiled(params, shared_context):
    timings = {}

    # 0) Derived quantities
    dt_end_samples = ((shared_context['tlen'] - (params['tc'] - shared_context['epoch'])) * shared_context['sample_rate']).astype(cp.int32)
    forward_zeroes = dt_end_samples + shared_context['extra_forward_zeroes'] + shared_context['kernel_length']

    # 1) Waveform generation (_bbhx_fd)
    s, e = gpu_timer(); s()
    waveforms = _bbhx_fd(
        ifos=['LISA_A', 'LISA_E'],
        tdi=shared_context['tdi'],
        t_obs_start=shared_context['t_obs_start'],
        delta_f=shared_context['delta_f'],
        f_final=shared_context['f_final'],
        mode_array=shared_context['mode_array'],
        t_offset=shared_context['t_offset'],
        **params
    )
    timings['wfgen_ms'] = e()

    # 2) Whitening and IRFFT to TD
    fd_A = cp.asarray(waveforms['LISA_A'])  # (B, Nf)
    fd_E = cp.asarray(waveforms['LISA_E'])
    if fd_A.ndim == 1: fd_A = fd_A[cp.newaxis, :]
    if fd_E.ndim == 1: fd_E = fd_E[cp.newaxis, :]

    whiten_A = cp.asarray(shared_context['whitening_psds']['LISA_A'].data)
    whiten_E = cp.asarray(shared_context['whitening_psds']['LISA_E'].data)

    Nfreq = fd_A.shape[-1]
    n_time = 2 * (Nfreq - 1)
    df = shared_context['delta_f']
    scale = n_time * df

    s, e = gpu_timer(); s()
    fd_A_whitened = fd_A * whiten_A.conj()
    fd_E_whitened = fd_E * whiten_E.conj()
    fd_stacked = cp.stack([fd_A_whitened, fd_E_whitened], axis=1)
    td_stacked = cp.fft.irfft(fd_stacked * scale, n=n_time, axis=-1)
    timings['whiten_irfft_ms'] = e()

    # 3) Zeroing (forward + cutoff) — original uses Python loop; time it as-is
    s, e = gpu_timer(); s()
    nfz_vec = forward_zeroes.astype(cp.int32)
    nctf = int(shared_context['cutoff_time'] * shared_context['sample_rate'])
    Nt = td_stacked.shape[-1]
    B = td_stacked.shape[0]
    for b in range(B):
        nfz = int(nfz_vec[b].item())
        if nfz > 0:
            td_stacked[b, :, :nfz] = 0
    if nctf > 0 and nctf < Nt:
        td_stacked[:, :, -nctf:] = 0
    timings['zeroing_ms'] = e()

    # 4) RFFT back to FD
    s, e = gpu_timer(); s()
    waveforms_fd = cp.fft.rfft(td_stacked, axis=-1) / scale
    timings['rfft_ms'] = e()

    # 5) Overlaps + sigmasq
    s, e = gpu_timer(); s()
    data_A_fd = cp.asarray(shared_context['lisa_a_strain_fd'].data)[cp.newaxis, :]
    data_E_fd = cp.asarray(shared_context['lisa_e_strain_fd'].data)[cp.newaxis, :]

    wf_A_fd = waveforms_fd[:, 0, :]
    wf_E_fd = waveforms_fd[:, 1, :]
    inner_A = cp.sum(data_A_fd.conj() * wf_A_fd, axis=-1)
    inner_E = cp.sum(data_E_fd.conj() * wf_E_fd, axis=-1)
    overlap_A = 4.0 * df * inner_A
    overlap_E = 4.0 * df * inner_E
    sigmasq_A = 4.0 * df * cp.sum(cp.abs(wf_A_fd)**2, axis=-1)
    sigmasq_E = 4.0 * df * cp.sum(cp.abs(wf_E_fd)**2, axis=-1)
    log_likelihood_values = (overlap_A + overlap_E).real - 0.5 * (sigmasq_A + sigmasq_E)
    timings['overlaps_ms'] = e()

    timings['total_ms'] = sum(timings[k] for k in ['wfgen_ms','whiten_irfft_ms','zeroing_ms','rfft_ms','overlaps_ms'])

    return log_likelihood_values, timings


def main():
    shared_context = {}
    shared_context['tlen'] = 2592000
    shared_context['sample_rate'] = 0.2
    shared_context['delta_f'] = 1./shared_context['tlen']
    shared_context['delta_t'] = 5
    shared_context['flen'] = shared_context['tlen']//2 + 1
    shared_context['cutoff_time'] = 86400*7
    shared_context['kernel_length'] = 17280
    shared_context['extra_forward_zeroes'] = 8640
    shared_context['data_file'] = 'signal_0.hdf'
    shared_context['psd_file'] = 'model_AE_TDI1_SMOOTH_optimistic.txt.gz'

    orig.initialization(shared_context)

    shared_context['tdi'] = '1.5'
    shared_context['t_obs_start'] = shared_context['tlen']
    shared_context['f_final'] = shared_context['sample_rate'] / 2
    shared_context['approximant'] = 'BBHX_PhenomD'
    shared_context['t_offset'] = 7365189.431698299
    shared_context['mode_array'] = [(2,2)]

    B = 100
    params = build_params(B)

    # Warm-up calls to eliminate setup costs
    print("Running warm-up calls...")
    ctx_warmup = dict(shared_context)
    _ = log_likelihood_profiled(params, ctx_warmup)
    ctx_warmup = dict(shared_context)
    _ = opt.log_likelihood_optimized(params, ctx_warmup)
    ctx_warmup = dict(shared_context)
    _ = opt2.log_likelihood_optimized2(params, ctx_warmup)
    cp.cuda.Stream.null.synchronize()
    print("Warm-up complete.\n")

    print("=" * 80)
    print("REFERENCE IMPLEMENTATION")
    print("=" * 80)
    
    # Measure CPU wall time too
    ctx_ref = dict(shared_context)
    t0 = time.perf_counter()
    ll_ref, timings_ref = log_likelihood_profiled(params, ctx_ref)
    cp.cuda.Stream.null.synchronize()
    wall_ms_ref = (time.perf_counter() - t0) * 1000.0

    print(f"Wall time: {wall_ms_ref:.2f} ms")
    print("Stage timings (ms):")
    for k in ['wfgen_ms', 'whiten_irfft_ms', 'zeroing_ms', 'rfft_ms', 'overlaps_ms']:
        print(f"  {k:20s}: {timings_ref[k]:7.2f}")
    print(f"  {'total_ms':20s}: {timings_ref['total_ms']:7.2f}")
    print(f"Log-likelihood stats: min={cp.min(ll_ref):.2f}, max={cp.max(ll_ref):.2f}, mean={cp.mean(ll_ref):.2f}")
    
    print()
    print("=" * 80)
    print("OPTIMIZED IMPLEMENTATION")
    print("=" * 80)
    
    ctx_opt = dict(shared_context)
    t1 = time.perf_counter()
    ll_opt = opt.log_likelihood_optimized(params, ctx_opt)
    cp.cuda.Stream.null.synchronize()
    wall_ms_opt = (time.perf_counter() - t1) * 1000.0
    
    print(f"Wall time: {wall_ms_opt:.2f} ms")
    if 'timing_ms' in ctx_opt:
        print("Stage timings (ms):")
        for k in ['whiten_irfft', 'zeroing', 'rfft', 'overlaps']:
            print(f"  {k:20s}: {ctx_opt['timing_ms'][k]:7.2f}")
        print(f"  {'total':20s}: {ctx_opt['timing_ms']['total']:7.2f}")
    print(f"Log-likelihood stats: min={cp.min(ll_opt):.2f}, max={cp.max(ll_opt):.2f}, mean={cp.mean(ll_opt):.2f}")
    print(f"Speedup vs reference: {wall_ms_ref/wall_ms_opt:.2f}x")
    
    print()
    print("=" * 80)
    print("OPTIMIZED2 IMPLEMENTATION")
    print("=" * 80)
    
    ctx_opt2 = dict(shared_context)
    t2 = time.perf_counter()
    ll_opt2 = opt2.log_likelihood_optimized2(params, ctx_opt2)
    cp.cuda.Stream.null.synchronize()
    wall_ms_opt2 = (time.perf_counter() - t2) * 1000.0
    
    print(f"Wall time: {wall_ms_opt2:.2f} ms")
    if 'timing_ms' in ctx_opt2:
        print("Stage timings (ms):")
        for k in ['whiten_irfft', 'zeroing', 'rfft', 'overlaps']:
            print(f"  {k:20s}: {ctx_opt2['timing_ms'][k]:7.2f}")
        print(f"  {'total':20s}: {ctx_opt2['timing_ms']['total']:7.2f}")
    print(f"Log-likelihood stats: min={cp.min(ll_opt2):.2f}, max={cp.max(ll_opt2):.2f}, mean={cp.mean(ll_opt2):.2f}")
    print(f"Speedup vs reference: {wall_ms_ref/wall_ms_opt2:.2f}x")
    print(f"Speedup vs optimized: {wall_ms_opt/wall_ms_opt2:.2f}x")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Implementation':<20s} {'Wall (ms)':>12s} {'Speedup':>10s}")
    print("-" * 80)
    print(f"{'Reference':<20s} {wall_ms_ref:12.2f} {'1.00x':>10s}")
    print(f"{'Optimized':<20s} {wall_ms_opt:12.2f} {wall_ms_ref/wall_ms_opt:10.2f}x")
    print(f"{'Optimized2':<20s} {wall_ms_opt2:12.2f} {wall_ms_ref/wall_ms_opt2:10.2f}x")


if __name__ == '__main__':
    main()
