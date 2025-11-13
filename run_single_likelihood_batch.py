from pycbc.types import timeseries as pycbc_ts
import pycbc.psd as pycbc_psd
from pycbc.strain.strain import execute_cached_fft
import cupy as cp

from pre_merger_utils import generate_pre_merger_psds
from pre_merger_utils import pre_process_data_lisa_pre_merger

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

    shared_context['lisa_a_strain_fd'] = lisa_a_strain_fd
    shared_context['lisa_e_strain_fd'] = lisa_e_strain_fd
    shared_context['epoch'] = float(lisa_a_strain_fd._epoch) # Start time of array


def log_likelihood(params, shared_context):
    # Identify how much zeroing on the front is needed first
    print(params['tc'])
    dt_end_samples = ((shared_context['tlen'] - (params['tc'] - shared_context['epoch'])) * shared_context['sample_rate']).astype(cp.int32)
    forward_zeroes = dt_end_samples +shared_context['extra_forward_zeroes'] + shared_context['kernel_length']

    waveforms_A, waveforms_E = generate_waveform(params)

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
    shared_context['data_file'] = 'signal_0.hdf'
    shared_context['psd_file'] = 'model_AE_TDI1_SMOOTH_optimistic.txt.gz'

    initialization(shared_context)

    # Cupy doesn't appear to support structured arrays, so per-field arrays
    # seem needed.
    field_names = [
        'mass1','mass2','spin1z','spin2z',
        'distance','eclipticlongitude','eclipticlatitude',
        'inclination','polarization','coa_phase','tc'
    ]

    # create a dict of per-field CuPy arrays (10 rows each)
    params = {name: cp.zeros(10, dtype=cp.float64) for name in field_names}
    
    # Set values for each row (currently all rows have the same values)
    params['mass1'][:] = 1000000.0
    params['mass2'][:] = 1000000.0
    params['spin1z'][:] = 0
    params['spin2z'][:] = 0
    params['distance'][:] = 27658.011507544677
    params['eclipticlongitude'][:] = 3.448296944257913
    params['eclipticlatitude'][:] = 0.44491231446252155
    params['inclination'][:] = 0.9238365050097769
    params['polarization'][:] = 3.4236020095353483
    params['coa_phase'][:] = 2.661901610522322
    params['tc'][:] = 1931852406.9997194

    # Some top-level constant waveform parameters
    shared_context['tdi'] = '1.5'
    shared_context['t_obs_start'] = shared_context['tlen']
    shared_context['f_final'] = shared_context['sample_rate'] / 2
    # cutoff_deltat, are they needed?
    shared_context['approximant'] = 'BBHX_PhenomD'
    shared_context['t_offset'] = 7365189.431698299
    shared_context['mode_array'] = [(2,2)]
    
    # Call log_likelihood
    log_likelihood(params, shared_context)

    print(shared_context)


if __name__ == "__main__":
    main()
