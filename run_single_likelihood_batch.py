from pycbc.types import timeseries as pycbc_ts
import pycbc.psd as pycbc_psd
from pycbc.strain.strain import execute_cached_fft
import cupy as cp

from pre_merger_utils import generate_pre_merger_psds
from pre_merger_utils import pre_process_data_lisa_pre_merger
from BBHX_Phenom_GPU import _bbhx_fd

# We agreed not to use our own inference toolkits, but PyCBC (and it's
# standard utilities, like timeseries) I think are still fair game.

def initialization(shared_context):
    # This stuff would all be done once. In theory a bunch of it probably
    # could go on the GPU (and could be extracted from PyCBC here) but it's
    # simpler to just do what we did for the premerger paper.
    whitening_psds = {}
    whitening_psds['LISA_A'] = generate_pre_merger_psds(
        shared_context['psd_file'],
        sample_rate=shared_context['sample_rate'],
        duration=shared_context['tlen'],
        kernel_length=shared_context['kernel_length'],
    )["FD"]
    whitening_psds['LISA_E'] = generate_pre_merger_psds(
        shared_context['psd_file'],
        sample_rate=shared_context['sample_rate'],
        duration=shared_context['tlen'],
        kernel_length=shared_context['kernel_length'],
    )["FD"]
    shared_context['whitening_psds'] = whitening_psds

    # Read in data
    # Both this data and the PSD come from the premerger paper data release
    data_A = pycbc_ts.load_timeseries(
        shared_context['data_file'],
        group="/LISA_A",
    )
    data_A._delta_t = 5 # Apparently it is not exactly this in the files
    data_E = pycbc_ts.load_timeseries(
        shared_context['data_file'],
        group="/LISA_E",
    )
    data_E._delta_t = 5
    pre_merger_data = pre_process_data_lisa_pre_merger(
        {'LISA_A': data_A, 'LISA_E': data_E},
        sample_rate=shared_context['sample_rate'],
        psds_for_whitening=shared_context['whitening_psds'],
        window_length=0, 
        cutoff_time=shared_context['cutoff_time'],
        forward_zeroes=shared_context['kernel_length'],
    )
    shared_context['pre_merger_data'] = pre_merger_data

    # Frequency-domain data for computing log-likelihood
    lisa_a_strain_fd = execute_cached_fft(
        pre_merger_data["LISA_A"],
        copy_output=True,
        uid=3223965
    )

    lisa_e_strain_fd = execute_cached_fft(
        pre_merger_data["LISA_E"],
        copy_output=True,
        uid=3223967
    )

    lisa_a_strain_fd._epoch = 0
    lisa_e_strain_fd._epoch = 0

    shared_context['lisa_a_strain_fd'] = lisa_a_strain_fd
    shared_context['lisa_e_strain_fd'] = lisa_e_strain_fd
    shared_context['epoch'] = float(lisa_a_strain_fd._epoch) # Start time of array


def log_likelihood(params, shared_context):
    # Identify how much zeroing on the front is needed first
    dt_end_samples = ((shared_context['tlen'] - (params['tc'] - shared_context['epoch'])) * shared_context['sample_rate']).astype(cp.int32)
    forward_zeroes = dt_end_samples +shared_context['extra_forward_zeroes'] + shared_context['kernel_length']

    # Call _bbhx_fd with parameters from shared_context and params
    waveforms = _bbhx_fd(
        ifos=['LISA_A', 'LISA_E'],
        tdi=shared_context['tdi'],
        t_obs_start=shared_context['t_obs_start'],
        delta_f=shared_context['delta_f'],
        f_final=shared_context['f_final'],
        mode_array=shared_context['mode_array'],
        **params
    )

    # Retrieve FD waveforms (CuPy arrays) and ensure batch dimension
    fd_A = cp.asarray(waveforms['LISA_A'])
    fd_E = cp.asarray(waveforms['LISA_E'])
    

    if fd_A.ndim == 1:
        fd_A = fd_A[cp.newaxis, :]
    if fd_E.ndim == 1:
        fd_E = fd_E[cp.newaxis, :]

    if fd_A.shape[-1] != fd_E.shape[-1]:
        raise ValueError(f"A/E frequency lengths differ: {fd_A.shape[-1]} vs {fd_E.shape[-1]}")

    # Whiten FD waveforms using whitening PSDs (conjugate multiply, following apply_pre_merger_kernel)
    # Extract whitening kernels from PyCBC FrequencySeries and convert to CuPy
    whiten_A = cp.asarray(shared_context['whitening_psds']['LISA_A'].data)
    whiten_E = cp.asarray(shared_context['whitening_psds']['LISA_E'].data)
    
    # Apply whitening: multiply by conjugate of whitening kernel
    fd_A_whitened = fd_A * whiten_A.conj()
    fd_E_whitened = fd_E * whiten_E.conj()

    # Batch both channels together: shape (B, 2, Nfreq)
    fd_stacked = cp.stack([fd_A_whitened, fd_E_whitened], axis=1)

    # Compute inverse FFT to time domain along frequency axis with correct scaling
    # Discrete mapping: X_k (DFT coeffs) = S(f_k) * (n * df) = S(f_k) / dt
    Nfreq = fd_stacked.shape[-1]
    n_time = 2 * (Nfreq - 1)
    df = shared_context['delta_f']
    scale = n_time * df  # equals 1/dt

    td_stacked = cp.fft.irfft(fd_stacked * scale, n=n_time, axis=-1)


    # Apply pre-merger kernel operations in TD (following apply_pre_merger_kernel):
    # 1. Zero initial samples (forward_zeroes per-batch)
    # 2. Apply window if specified (window_length samples after nfz)
    # 3. Zero final samples (cutoff_time -> nctf samples)
    
    # Use the computed per-batch forward_zeroes (now corrected upstream)
    nfz_vec = cp.asarray(forward_zeroes).astype(cp.int32)
    window_length = 0  # from pre_process_data_lisa_pre_merger call in initialization
    nctf = int(shared_context['cutoff_time'] * shared_context['sample_rate'])
    
    Nt = td_stacked.shape[-1]
    B = td_stacked.shape[0]

    # Zero forward samples per-batch
    for b in range(B):
        nfz = int(nfz_vec[b].item())
        if nfz > 0:
            td_stacked[b, :, :nfz] = 0

    # Apply window if window_length > 0
    if window_length > 0 and nfz + window_length < Nt:
        from scipy import signal
        window = signal.windows.hann(window_length * 2 + 1)[:window_length]
        window_cp = cp.asarray(window)
        td_stacked[:, :, nfz:nfz+window_length] *= window_cp

    # Zero final samples (cutoff region, same for all batches)
    if nctf > 0 and nctf < Nt:
        td_stacked[:, :, -nctf:] = 0

    nfz_min = int(cp.min(nfz_vec).item()) if nfz_vec.size else 0
    nfz_max = int(cp.max(nfz_vec).item()) if nfz_vec.size else 0

    # Store for downstream use
    shared_context['waveforms_td'] = td_stacked
    
    # ===== Likelihood Calculation =====
    # 1. Convert waveforms back to frequency domain
    # Inverse of the IFFT scaling: we multiplied by (n_time * df) = 1/dt before irfft,
    # so we divide by that factor after rfft to get back to S(f_k)
    waveforms_fd = cp.fft.rfft(td_stacked, axis=-1) / scale  # Shape: (B, 2, Nfreq)
    
    
    # 2. Extract data FD arrays (already whitened in initialization)
    data_A_fd = cp.asarray(shared_context['lisa_a_strain_fd'].data)  # Shape: (Nfreq,)
    data_E_fd = cp.asarray(shared_context['lisa_e_strain_fd'].data)

    # Broadcast data to match batch dimension: (1, Nfreq) for broadcasting
    data_A_fd = data_A_fd[cp.newaxis, :]  # (1, Nfreq)
    data_E_fd = data_E_fd[cp.newaxis, :]

    # 3. Compute data-waveform overlaps (batched overlap_cplx)
    # overlap = 4 * df * sum(data*.conj() * waveform)
    # Extract waveform FD for each channel: (B, Nfreq)
    wf_A_fd = waveforms_fd[:, 0, :]  # (B, Nfreq)
    wf_E_fd = waveforms_fd[:, 1, :]

    # Compute inner products per batch element
    inner_A = cp.sum(data_A_fd.conj() * wf_A_fd, axis=-1)  # (B,) complex
    inner_E = cp.sum(data_E_fd.conj() * wf_E_fd, axis=-1)

    overlap_A = 4.0 * df * inner_A  # (B,) complex
    overlap_E = 4.0 * df * inner_E


    # 4. Compute waveform self-overlaps (batched sigmasq)
    # sigmasq = 4 * df * sum(|waveform|^2) - should be real
    sigmasq_A = 4.0 * df * cp.sum(cp.abs(wf_A_fd)**2, axis=-1)  # (B,) real
    sigmasq_E = 4.0 * df * cp.sum(cp.abs(wf_E_fd)**2, axis=-1)


    # 5. Compute log-likelihood for each waveform
    # logL = Re(overlap_A + overlap_E) - (sigmasq_A + sigmasq_E)/2
    log_likelihood_values = (overlap_A + overlap_E).real - 0.5 * (sigmasq_A + sigmasq_E)
    
    
    # Store results
    shared_context['log_likelihoods'] = log_likelihood_values
    shared_context['overlaps'] = {'A': overlap_A, 'E': overlap_E}
    shared_context['sigmasqs'] = {'A': sigmasq_A, 'E': sigmasq_E}

def main() -> None:
    shared_context = {}

    # Set some top-level parameters
    shared_context['tlen'] = 2592000
    shared_context['sample_rate'] = 0.2
    shared_context['delta_f'] = 1./shared_context['tlen']
    shared_context['delta_t'] = 5
    shared_context['flen'] = shared_context['tlen']//2 + 1
    shared_context['cutoff_time'] = 86400*7
    shared_context['kernel_length'] = 17280
    shared_context['extra_forward_zeroes'] = 8640
    shared_context['data_file'] = 'signal_0_new.hdf'
    shared_context['psd_file'] = 'model_AE_TDI1_SMOOTH_optimistic.txt.gz'

    initialization(shared_context)

    # Desired batch size; will fallback to 50 if we hit GPU OOM
    batch_size = 100

    # Cupy doesn't appear to support structured arrays, so per-field arrays seem needed.
    field_names = [
        'mass1','mass2','spin1z','spin2z',
        'distance','eclipticlongitude','eclipticlatitude',
        'inclination','polarization','coa_phase','tc'
    ]

    def build_params(n):
        p = {name: cp.zeros(n, dtype=cp.float64) for name in field_names}
        # Simple variation across the batch
        p['mass1'][:] = 1000000.0 + cp.arange(n) * 10000.0  # 1e6 .. 1e6+ (n-1)*1e4
        p['mass2'][:] = 1000000.0
        p['spin1z'][:] = cp.linspace(0, 0.1, n)
        p['spin2z'][:] = 0
        p['distance'][:] = 27658.011507544677
        p['eclipticlongitude'][:] = 3.448296944257913
        p['eclipticlatitude'][:] = 0.44491231446252155
        p['inclination'][:] = 0.9238365050097769
        p['polarization'][:] = 3.4236020095353483
        p['coa_phase'][:] = 2.661901610522322
        p['tc'][:] = int(30*86400) # Using same tc for all rows
        return p

    # Some top-level constant waveform parameters
    shared_context['tdi'] = '1.5'
    shared_context['t_obs_start'] = shared_context['tlen']
    shared_context['f_final'] = shared_context['sample_rate'] / 2
    # cutoff_deltat, are they needed?
    shared_context['approximant'] = 'BBHX_PhenomD'
    shared_context['mode_array'] = [(2,2)]
    
    # Always use 100-waveform batch
    params = build_params(batch_size)
    log_likelihood(params, shared_context)

    print("LIKELIHOODS ARE", shared_context['log_likelihoods'])

    # Plot a handful of TD waveforms for verification
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    
    td_waveforms = shared_context['waveforms_td']  # Shape: (B, 2, Nt)
    B, num_channels, Nt = td_waveforms.shape
    dt = shared_context['delta_t']
    t = cp.arange(Nt) * dt  # Time axis in seconds
    t_np = cp.asnumpy(t)
    
    # Plot directory
    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)
    
    # Plot first 3 waveforms (or fewer if B < 3)
    num_to_plot = min(3, B)
    
    for b in range(num_to_plot):
        plt.figure(figsize=(14, 5))
        
        # LISA_A
        plt.subplot(1, 2, 1)
        wf_A = cp.asnumpy(td_waveforms[b, 0, :])
        plt.plot(t_np / 86400.0, wf_A, linewidth=0.5)
        plt.xlabel('Time [days]')
        plt.ylabel('Whitened strain')
        plt.title(f'Waveform {b}: LISA_A (TD, whitened+zeroed)')
        plt.grid(True, alpha=0.3)
        
        # LISA_E
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
    
    # Zoom into non-zero region (between nfz and -nctf)
    nfz_min = 25920  # From debug output
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
    
    print(f"\nPlots saved in {outdir}/")
    print("DONE")


if __name__ == "__main__":
    main()
