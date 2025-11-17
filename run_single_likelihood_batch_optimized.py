import time
import cupy as cp
from BBHX_Phenom_GPU import _bbhx_fd
import run_single_likelihood_batch as orig  # reuse initialization


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


def _get_buffers(shared_context, B, Nfreq, Nt):
    key = (B, Nfreq, Nt, 'f64')
    buf = shared_context.get('optimized_buffers')
    if buf is None or buf.get('key') != key:
        shared_context['optimized_buffers'] = buf = {
            'key': key,
            'fd_stacked': cp.empty((B, 2, Nfreq), dtype=cp.complex128),
            'td_stacked': cp.empty((B, 2, Nt), dtype=cp.float64),
            'waveforms_fd': cp.empty((B, 2, Nfreq), dtype=cp.complex128),
        }
    return shared_context['optimized_buffers']


_ZERO_TD_F64 = cp.ElementwiseKernel(
    'float64 td, raw int32 nfz, int32 Nt, int32 C, int32 nctf',
    'float64 out',
    '''
    int idx = i;
    int t = idx % Nt;
    int bc = idx / Nt;
    int b = bc / C;
    int start = nfz[b];
    int end_zero_start = Nt - nctf;
    out = (t < start || t >= end_zero_start) ? 0.0 : td;
    ''',
    'zero_td_f64_opt'
)


def log_likelihood_optimized(params, shared_context):
    # GPU timer helpers
    def gpu_timer_start():
        start = cp.cuda.Event(); end = cp.cuda.Event()
        start.record()
        return start, end
    def gpu_timer_end(start, end):
        end.record(); end.synchronize(); return cp.cuda.get_elapsed_time(start, end)  # ms

    # Compute forward zeros per-batch
    dt_end_samples = ((shared_context['tlen'] - (params['tc'] - shared_context['epoch'])) * shared_context['sample_rate']).astype(cp.int32)
    forward_zeroes = dt_end_samples + shared_context['extra_forward_zeroes'] + shared_context['kernel_length']

    # Generate FD waveforms (batched) from BBHx wrapper
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

    fd_A = cp.asarray(waveforms['LISA_A'])  # (B, Nf)
    fd_E = cp.asarray(waveforms['LISA_E'])
    if fd_A.ndim == 1: fd_A = fd_A[cp.newaxis, :]
    if fd_E.ndim == 1: fd_E = fd_E[cp.newaxis, :]

    whiten_A = cp.asarray(shared_context['whitening_psds']['LISA_A'].data)
    whiten_E = cp.asarray(shared_context['whitening_psds']['LISA_E'].data)

    # Precompute constants
    Nfreq = fd_A.shape[-1]
    n_time = 2 * (Nfreq - 1)
    df = shared_context['delta_f']
    scale = n_time * df  # 1/dt

    # Buffers
    bufs = _get_buffers(shared_context, fd_A.shape[0], Nfreq, n_time)

    # Timed: FD whitening + scaling + IFFT to TD
    s0,e0 = gpu_timer_start()
    fd_stacked = bufs['fd_stacked']
    fd_stacked[:, 0, :] = fd_A * whiten_A.conj()
    fd_stacked[:, 1, :] = fd_E * whiten_E.conj()
    td_stacked = bufs['td_stacked']
    td_stacked[...] = cp.fft.irfft(fd_stacked * scale, n=n_time, axis=-1)  # (B,2,Nt)
    t_whiten_irfft_ms = gpu_timer_end(s0,e0)

    # Timed: Zeroing via fused elementwise kernel (avoids huge boolean masks)
    s1,e1 = gpu_timer_start()
    Nt = td_stacked.shape[-1]
    nctf = int(shared_context['cutoff_time'] * shared_context['sample_rate'])
    nfz_vec = forward_zeroes.astype(cp.int32)

    td_stacked = _ZERO_TD_F64(td_stacked.reshape(-1), nfz_vec, Nt, 2, nctf).reshape(td_stacked.shape)
    t_zero_ms = gpu_timer_end(s1,e1)

    # Timed: FFT back to FD
    s2,e2 = gpu_timer_start()
    waveforms_fd = bufs['waveforms_fd']
    waveforms_fd[...] = cp.fft.rfft(td_stacked, axis=-1) / scale  # (B,2,Nf)
    t_rfft_ms = gpu_timer_end(s2,e2)

    # Timed: Overlaps and sigmasq (batched, both channels together) via elementwise + reduction
    s3,e3 = gpu_timer_start()
    data_stack = cp.stack([
        cp.asarray(shared_context['lisa_a_strain_fd'].data),
        cp.asarray(shared_context['lisa_e_strain_fd'].data)
    ], axis=0)  # (2,Nf)
    data_stack_conj = data_stack.conj()  # (2,Nf)
    inner = cp.sum(waveforms_fd * data_stack_conj[cp.newaxis, :, :], axis=-1)  # (B,2)
    overlap = 4.0 * df * inner
    sigmasq = 4.0 * df * cp.sum((waveforms_fd.conj() * waveforms_fd).real, axis=-1)  # (B,2) real
    log_likelihood_values = overlap.real.sum(axis=-1) - 0.5 * sigmasq.sum(axis=-1)
    t_overlaps_ms = gpu_timer_end(s3,e3)

    # Save results and timings
    shared_context['waveforms_td'] = td_stacked
    shared_context['log_likelihoods'] = log_likelihood_values
    shared_context['timing_ms'] = {
        'whiten_irfft': float(t_whiten_irfft_ms),
        'zeroing': float(t_zero_ms),
        'rfft': float(t_rfft_ms),
        'overlaps': float(t_overlaps_ms),
        'total': float(t_whiten_irfft_ms + t_zero_ms + t_rfft_ms + t_overlaps_ms),
    }

    return log_likelihood_values

def main():
    # Build shared context via original initializer
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

    # Waveform/global params
    shared_context['tdi'] = '1.5'
    shared_context['t_obs_start'] = shared_context['tlen']
    shared_context['f_final'] = shared_context['sample_rate'] / 2
    shared_context['approximant'] = 'BBHX_PhenomD'
    shared_context['t_offset'] = 7365189.431698299
    shared_context['mode_array'] = [(2,2)]

    # Params and run
    B = 100
    params = build_params(B)

    t1 = time.perf_counter()
    ll_opt = log_likelihood_optimized(params, shared_context)
    cp.cuda.Stream.null.synchronize()
    t_opt = (time.perf_counter() - t1) * 1000.0

    print("Optimized total time: %.1f ms" % t_opt)
    if 'timing_ms' in shared_context:
        print("Breakdown (ms):", shared_context['timing_ms'])
    print("LIKELIHOODS ARE", cp.asnumpy(ll_opt))

    # Plot a handful of TD waveforms for verification (mirrors reference)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    td_waveforms = shared_context['waveforms_td']  # Shape: (B, 2, Nt)
    Bv, num_channels, Nt = td_waveforms.shape
    dt = shared_context['delta_t']
    t = cp.arange(Nt) * dt
    t_np = cp.asnumpy(t)

    outdir = 'plots_opt'
    os.makedirs(outdir, exist_ok=True)
    num_to_plot = min(3, Bv)

    for b in range(num_to_plot):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        wf_A = cp.asnumpy(td_waveforms[b, 0, :])
        plt.plot(t_np / 86400.0, wf_A, linewidth=0.5)
        plt.xlabel('Time [days]')
        plt.ylabel('Whitened strain')
        plt.title(f'Waveform {b}: LISA_A (TD, whitened+zeroed)')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        wf_E = cp.asnumpy(td_waveforms[b, 1, :])
        plt.plot(t_np / 86400.0, wf_E, linewidth=0.5)
        plt.xlabel('Time [days]')
        plt.ylabel('Whitened strain')
        plt.title(f'Waveform {b}: LISA_E (TD, whitened+zeroed)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'waveform_td_{b}.png'), dpi=150)
        plt.close()
        print(f"Saved waveform {b} plot: max(A)={wf_A.max():.3e}, max(E)={wf_E.max():.3e}")

    # Zoomed region
    nfz_min = 25920
    nctf = int(shared_context['cutoff_time'] * shared_context['sample_rate'])
    zoom_start = nfz_min
    zoom_end = Nt - nctf
    if zoom_end > zoom_start:
        t_zoom = t_np[zoom_start:zoom_end]
        plt.figure(figsize=(14, 5))
        for b in range(num_to_plot):
            plt.subplot(1, 2, 1)
            wf_A_zoom = cp.asnumpy(td_waveforms[b, 0, zoom_start:zoom_end])
            plt.plot((t_zoom - t_zoom[0]) / 3600.0, wf_A_zoom, label=f'Wf {b}', linewidth=0.7)
        plt.xlabel('Time in window [hours]')
        plt.ylabel('Whitened strain')
        plt.title('LISA_A (zoomed to non-zero region)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        for b in range(num_to_plot):
            plt.subplot(1, 2, 2)
            wf_E_zoom = cp.asnumpy(td_waveforms[b, 1, zoom_start:zoom_end])
            plt.plot((t_zoom - t_zoom[0]) / 3600.0, wf_E_zoom, label=f'Wf {b}', linewidth=0.7)
        plt.xlabel('Time in window [hours]')
        plt.ylabel('Whitened strain')
        plt.title('LISA_E (zoomed to non-zero region)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'waveforms_td_zoom.png'), dpi=150)
        plt.close()
        print(f"Saved zoomed waveform plot (samples {zoom_start}:{zoom_end})")

if __name__ == '__main__':
    main()
