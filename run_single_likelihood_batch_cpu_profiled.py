import time
import numpy as np
from pycbc.types import timeseries as pycbc_ts
from pycbc.strain.strain import execute_cached_fft

from pre_merger_utils import generate_pre_merger_psds
from pre_merger_utils import pre_process_data_lisa_pre_merger
from BBHX_Phenom_GPU import _bbhx_fd


def initialization(shared_context):
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

    data_A = pycbc_ts.load_timeseries(
        shared_context['data_file'],
        group="/LISA_A",
    )
    data_A._delta_t = 5
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


def build_params(n):
    field_names = [
        'mass1','mass2','spin1z','spin2z',
        'distance','eclipticlongitude','eclipticlatitude',
        'inclination','polarization','coa_phase','tc'
    ]
    p = {name: np.zeros(n, dtype=np.float64) for name in field_names}
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
    p['tc'][:] = int(30 * 86400)
    return p


def log_likelihood_cpu_profiled(params, shared_context):
    timings = {
        'waveform_ms': 0.0,
        'whiten_irfft_ms': 0.0,
        'zero_td_ms': 0.0,
        'rfft_ms': 0.0,
        'overlaps_ms': 0.0,
        'total_ms': 0.0,
    }

    t0_total = time.perf_counter()

    # forward zero calculation
    dt_end_samples = (
        (shared_context['tlen'] - (params['tc'] - shared_context['epoch']))
        * shared_context['sample_rate']
    ).astype(np.int32)
    forward_zeroes = dt_end_samples + shared_context['extra_forward_zeroes'] + shared_context['kernel_length']

    # Waveforms
    t0 = time.perf_counter()
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
    timings['waveform_ms'] += (time.perf_counter() - t0) * 1000.0

    fd_A = np.asarray(waveforms['LISA_A'])
    fd_E = np.asarray(waveforms['LISA_E'])
    if fd_A.ndim == 1:
        fd_A = fd_A[np.newaxis, :]
    if fd_E.ndim == 1:
        fd_E = fd_E[np.newaxis, :]

    # Whiten + iRFFT
    t0 = time.perf_counter()
    whiten_A = np.asarray(shared_context['whitening_psds']['LISA_A'].data)
    whiten_E = np.asarray(shared_context['whitening_psds']['LISA_E'].data)
    fd_A_whitened = fd_A * whiten_A.conj()
    fd_E_whitened = fd_E * whiten_E.conj()
    fd_stacked = np.stack([fd_A_whitened, fd_E_whitened], axis=1)

    Nfreq = fd_stacked.shape[-1]
    n_time = 2 * (Nfreq - 1)
    df = shared_context['delta_f']
    scale = n_time * df
    td_stacked = np.fft.irfft(fd_stacked * scale, n=n_time, axis=-1)
    timings['whiten_irfft_ms'] += (time.perf_counter() - t0) * 1000.0

    # Zeroing in TD
    t0 = time.perf_counter()
    nfz_vec = np.asarray(forward_zeroes).astype(np.int32)
    nctf = int(shared_context['cutoff_time'] * shared_context['sample_rate'])
    Nt = td_stacked.shape[-1]
    B = td_stacked.shape[0]
    for b in range(B):
        nfz = int(nfz_vec[b].item())
        if nfz > 0:
            td_stacked[b, :, :nfz] = 0
    if nctf > 0 and nctf < Nt:
        td_stacked[:, :, -nctf:] = 0
    timings['zero_td_ms'] += (time.perf_counter() - t0) * 1000.0

    # rFFT back
    t0 = time.perf_counter()
    waveforms_fd = np.fft.rfft(td_stacked, axis=-1) / scale
    timings['rfft_ms'] += (time.perf_counter() - t0) * 1000.0

    # Overlaps
    t0 = time.perf_counter()
    data_A_fd = np.asarray(shared_context['lisa_a_strain_fd'].data)[np.newaxis, :]
    data_E_fd = np.asarray(shared_context['lisa_e_strain_fd'].data)[np.newaxis, :]
    wf_A_fd = waveforms_fd[:, 0, :]
    wf_E_fd = waveforms_fd[:, 1, :]
    inner_A = np.sum(data_A_fd.conj() * wf_A_fd, axis=-1)
    inner_E = np.sum(data_E_fd.conj() * wf_E_fd, axis=-1)
    overlap_A = 4.0 * df * inner_A
    overlap_E = 4.0 * df * inner_E
    sigmasq_A = 4.0 * df * np.sum(np.abs(wf_A_fd) ** 2, axis=-1)
    sigmasq_E = 4.0 * df * np.sum(np.abs(wf_E_fd) ** 2, axis=-1)
    log_likelihood_values = (overlap_A + overlap_E).real - 0.5 * (sigmasq_A + sigmasq_E)
    timings['overlaps_ms'] += (time.perf_counter() - t0) * 1000.0

    timings['total_ms'] += (time.perf_counter() - t0_total) * 1000.0

    # Persist for optional downstream checks
    shared_context['log_likelihoods'] = log_likelihood_values
    shared_context['waveforms_td'] = td_stacked

    return timings


def main():
    shared_context = {}
    # Config
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

    # Waveform settings
    shared_context['tdi'] = '1.5'
    shared_context['t_obs_start'] = shared_context['tlen']
    shared_context['f_final'] = shared_context['sample_rate'] / 2
    shared_context['approximant'] = 'BBHX_PhenomD'
    shared_context['t_offset'] = 7365189.431698299
    shared_context['mode_array'] = [(2, 2)]

    initialization(shared_context)

    batch_size = 100
    params = build_params(batch_size)

    # Warm-up (loads code paths, caches plans, etc.)
    _ = log_likelihood_cpu_profiled(params, shared_context)

    # Timed runs
    runs = 50
    accum = {k: 0.0 for k in ['waveform_ms','whiten_irfft_ms','zero_td_ms','rfft_ms','overlaps_ms','total_ms']}
    last_ll = None
    for i in range(runs):
        print("run", i)
        t = log_likelihood_cpu_profiled(params, shared_context)
        for k in accum:
            accum[k] += t[k]
        last_ll = shared_context['log_likelihoods']

    # Report
    print("\nCPU profile ({} runs, batch size {}):".format(runs, batch_size))
    for k in ['waveform_ms','whiten_irfft_ms','zero_td_ms','rfft_ms','overlaps_ms','total_ms']:
        print("- {:17s}: {:8.2f} ms/run".format(k, accum[k] / runs))
    print("\nExample likelihoods (first 5):", np.asarray(last_ll)[:5])


if __name__ == "__main__":
    main()
