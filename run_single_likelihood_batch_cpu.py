from pycbc.types import timeseries as pycbc_ts
import pycbc.psd as pycbc_psd
from pycbc.strain.strain import execute_cached_fft
import numpy as np

from pre_merger_utils import generate_pre_merger_psds
from pre_merger_utils import pre_process_data_lisa_pre_merger
from BBHX_Phenom_GPU import _bbhx_fd

# CPU version: identical logic to the GPU reference, but using NumPy arrays.


def initialization(shared_context):
    # Generate whitening PSDs (pre-merger pipeline convention)
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

    # Read in data (from pre-merger paper data release)
    data_A = pycbc_ts.load_timeseries(
        shared_context['data_file'],
        group="/LISA_A",
    )
    data_A._delta_t = 5  # File metadata slightly off, enforce 5 s
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
        uid=3223965,
    )

    lisa_e_strain_fd = execute_cached_fft(
        pre_merger_data["LISA_E"],
        copy_output=True,
        uid=3223967,
    )

    lisa_a_strain_fd._epoch = 0
    lisa_e_strain_fd._epoch = 0

    shared_context['lisa_a_strain_fd'] = lisa_a_strain_fd
    shared_context['lisa_e_strain_fd'] = lisa_e_strain_fd
    shared_context['epoch'] = float(lisa_a_strain_fd._epoch)


def log_likelihood(params, shared_context):
    # Determine forward zeroing per batch element
    dt_end_samples = (
        (shared_context['tlen'] - (params['tc'] - shared_context['epoch']))
        * shared_context['sample_rate']
    ).astype(np.int32)
    forward_zeroes = dt_end_samples + shared_context['extra_forward_zeroes'] + shared_context['kernel_length']

    # Generate FD waveforms with BBHx
    waveforms = _bbhx_fd(
        ifos=['LISA_A', 'LISA_E'],
        tdi=shared_context['tdi'],
        t_obs_start=shared_context['t_obs_start'],
        delta_f=shared_context['delta_f'],
        f_final=shared_context['f_final'],
        mode_array=shared_context['mode_array'],
        t_offset=shared_context['t_offset'],
        **params,
    )

    # Ensure batch dimension and NumPy arrays
    fd_A = np.asarray(waveforms['LISA_A'])
    fd_E = np.asarray(waveforms['LISA_E'])

    if fd_A.ndim == 1:
        fd_A = fd_A[np.newaxis, :]
    if fd_E.ndim == 1:
        fd_E = fd_E[np.newaxis, :]

    if fd_A.shape[-1] != fd_E.shape[-1]:
        raise ValueError(f"A/E frequency lengths differ: {fd_A.shape[-1]} vs {fd_E.shape[-1]}")

    # Whiten FD waveforms using whitening PSDs (conjugate multiply)
    whiten_A = np.asarray(shared_context['whitening_psds']['LISA_A'].data)
    whiten_E = np.asarray(shared_context['whitening_psds']['LISA_E'].data)

    fd_A_whitened = fd_A * whiten_A.conj()
    fd_E_whitened = fd_E * whiten_E.conj()

    # Stack channels: (B, 2, Nfreq)
    fd_stacked = np.stack([fd_A_whitened, fd_E_whitened], axis=1)

    # IFFT to time domain with correct scaling
    Nfreq = fd_stacked.shape[-1]
    n_time = 2 * (Nfreq - 1)
    df = shared_context['delta_f']
    scale = n_time * df  # equals 1/dt

    td_stacked = np.fft.irfft(fd_stacked * scale, n=n_time, axis=-1)

    # Apply pre-merger kernel in TD: zeroing and optional window
    nfz_vec = np.asarray(forward_zeroes).astype(np.int32)
    window_length = 0
    nctf = int(shared_context['cutoff_time'] * shared_context['sample_rate'])

    Nt = td_stacked.shape[-1]
    B = td_stacked.shape[0]

    # Zero forward samples per-batch
    for b in range(B):
        nfz = int(nfz_vec[b].item())
        if nfz > 0:
            td_stacked[b, :, :nfz] = 0

    # Apply window if requested (currently disabled)
    if window_length > 0:
        from scipy import signal
        window = signal.windows.hann(window_length * 2 + 1)[:window_length]
        td_stacked[:, :, nfz:nfz+window_length] *= window

    # Zero final samples (cutoff region)
    if nctf > 0 and nctf < Nt:
        td_stacked[:, :, -nctf:] = 0

    # Store TD waveforms
    shared_context['waveforms_td'] = td_stacked

    # Back to FD for likelihood (undo scaling)
    waveforms_fd = np.fft.rfft(td_stacked, axis=-1) / scale  # (B, 2, Nfreq)

    # Data FD (already whitened in initialization)
    data_A_fd = np.asarray(shared_context['lisa_a_strain_fd'].data)[np.newaxis, :]
    data_E_fd = np.asarray(shared_context['lisa_e_strain_fd'].data)[np.newaxis, :]

    # Overlaps (per batch)
    wf_A_fd = waveforms_fd[:, 0, :]
    wf_E_fd = waveforms_fd[:, 1, :]

    inner_A = np.sum(data_A_fd.conj() * wf_A_fd, axis=-1)
    inner_E = np.sum(data_E_fd.conj() * wf_E_fd, axis=-1)

    overlap_A = 4.0 * df * inner_A
    overlap_E = 4.0 * df * inner_E

    # Self-overlaps
    sigmasq_A = 4.0 * df * np.sum(np.abs(wf_A_fd) ** 2, axis=-1)
    sigmasq_E = 4.0 * df * np.sum(np.abs(wf_E_fd) ** 2, axis=-1)

    # Log-likelihood
    log_likelihood_values = (overlap_A + overlap_E).real - 0.5 * (sigmasq_A + sigmasq_E)

    # Store results
    shared_context['log_likelihoods'] = log_likelihood_values
    shared_context['overlaps'] = {'A': overlap_A, 'E': overlap_E}
    shared_context['sigmasqs'] = {'A': sigmasq_A, 'E': sigmasq_E}


def main() -> None:
    shared_context = {}

    # Top-level parameters
    shared_context['tlen'] = 2592000
    shared_context['sample_rate'] = 0.2
    shared_context['delta_f'] = 1.0 / shared_context['tlen']
    shared_context['delta_t'] = 5
    shared_context['flen'] = shared_context['tlen'] // 2 + 1
    shared_context['cutoff_time'] = 86400 * 7
    shared_context['kernel_length'] = 17280
    shared_context['extra_forward_zeroes'] = 8640
    shared_context['data_file'] = 'signal_0.hdf'
    shared_context['psd_file'] = 'model_AE_TDI1_SMOOTH_optimistic.txt.gz'

    initialization(shared_context)

    batch_size = 100

    field_names = [
        'mass1','mass2','spin1z','spin2z',
        'distance','eclipticlongitude','eclipticlatitude',
        'inclination','polarization','coa_phase','tc'
    ]

    def build_params(n):
        p = {name: np.zeros(n, dtype=np.float64) for name in field_names}
        # Simple variation across the batch
        p['mass1'][:] = 1000000.0 + np.arange(n) * 10000.0
        p['mass2'][:] = 1000000.0
        p['spin1z'][:] = np.linspace(0, 0.1, n)
        p['spin2z'][:] = 0
        p['distance'][:] = 27658.011507544677
        p['eclipticlongitude'][:] = 3.448296944257913
        p['eclipticlatitude'][:] = 0.44491231446252155
        p['inclination'][:] = 0.9238365050097769
        p['polarization'][:] = 3.4236020095353483
        p['coa_phase'][:] = 2.661901610522322
        p['tc'][:] = int(30 * 86400)  # same tc for all rows
        return p

    # Waveform/global settings
    shared_context['tdi'] = '1.5'
    shared_context['t_obs_start'] = shared_context['tlen']
    shared_context['f_final'] = shared_context['sample_rate'] / 2
    shared_context['approximant'] = 'BBHX_PhenomD'
    shared_context['t_offset'] = 7365189.431698299
    shared_context['mode_array'] = [(2, 2)]

    params = build_params(batch_size)
    log_likelihood(params, shared_context)

    print("LIKELIHOODS ARE", shared_context['log_likelihoods'])

    # Plot a handful of TD waveforms for verification
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    td_waveforms = shared_context['waveforms_td']  # (B, 2, Nt)
    B, num_channels, Nt = td_waveforms.shape
    dt = shared_context['delta_t']
    t = np.arange(Nt) * dt

    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)

    num_to_plot = min(3, B)

    for b in range(num_to_plot):
        plt.figure(figsize=(14, 5))

        # LISA_A
        plt.subplot(1, 2, 1)
        wf_A = td_waveforms[b, 0, :]
        plt.plot(t / 86400.0, wf_A, linewidth=0.5)
        plt.xlabel('Time [days]')
        plt.ylabel('Whitened strain')
        plt.title(f'Waveform {b}: LISA_A (TD, whitened+zeroed)')
        plt.grid(True, alpha=0.3)

        # LISA_E
        plt.subplot(1, 2, 2)
        wf_E = td_waveforms[b, 1, :]
        plt.plot(t / 86400.0, wf_E, linewidth=0.5)
        plt.xlabel('Time [days]')
        plt.ylabel('Whitened strain')
        plt.title(f'Waveform {b}: LISA_E (TD, whitened+zeroed)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'waveform_td_cpu_{b}.png'), dpi=150)
        plt.close()

        print(f"Saved CPU waveform {b} plot: max(A)={wf_A.max():.3e}, max(E)={wf_E.max():.3e}")

    # Zoom into non-zero region (between nfz and -nctf)
    nfz_min = 25920  # from prior debug context
    nctf = int(shared_context['cutoff_time'] * shared_context['sample_rate'])
    zoom_start = nfz_min
    zoom_end = Nt - nctf

    if zoom_end > zoom_start:
        t_zoom = t[zoom_start:zoom_end]

        plt.figure(figsize=(14, 5))
        for b in range(num_to_plot):
            plt.subplot(1, 2, 1)
            wf_A_zoom = td_waveforms[b, 0, zoom_start:zoom_end]
            plt.plot((t_zoom - t_zoom[0]) / 3600.0, wf_A_zoom, label=f'Wf {b}', linewidth=0.7)
        plt.xlabel('Time in window [hours]')
        plt.ylabel('Whitened strain')
        plt.title('LISA_A (zoomed to non-zero region)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        for b in range(num_to_plot):
            plt.subplot(1, 2, 2)
            wf_E_zoom = td_waveforms[b, 1, zoom_start:zoom_end]
            plt.plot((t_zoom - t_zoom[0]) / 3600.0, wf_E_zoom, label=f'Wf {b}', linewidth=0.7)
        plt.xlabel('Time in window [hours]')
        plt.ylabel('Whitened strain')
        plt.title('LISA_E (zoomed to non-zero region)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'waveforms_td_zoom_cpu.png'), dpi=150)
        plt.close()
        print(f"Saved CPU zoomed waveform plot (samples {zoom_start}:{zoom_end})")

    print(f"\nCPU plots saved in {outdir}/")
    print("DONE (CPU)")


if __name__ == "__main__":
    main()
