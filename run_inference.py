"""Run inference for LISA SMBH binary signals using SMC with GPU acceleration.

Requires installing sequince package from: https://github.com/sequince-dev/sequince

You may also need to install emcee to get it working (its meant to be optional
but we missed an import).
"""

from run_single_likelihood_batch_optimized import _bbhx_fd, _get_buffers, _ZERO_TD_F64
import run_single_likelihood_batch as orig  # reuse initialization

import matplotlib.pyplot as plt
import cupy as cp

from sequince import ParticleState
from sequince.logging_utils import configure_logging
from sequince.prebuilt_samplers import make_standard_bayesian_smc
from orng import ArrayRNG

def _log_likelihood_batch(params, shared_context):
    # Batched likelihood copied and modified from run_single_likelihood_batch_optimized.py
    # to remove the timing parts
    dt_end_samples = ((shared_context['tlen'] - (params['tc'] - shared_context['epoch'])) * shared_context['sample_rate']).astype(cp.int32)
    forward_zeroes = dt_end_samples + shared_context['extra_forward_zeroes'] + shared_context['kernel_length']
    waveforms = _bbhx_fd(
        ifos=['LISA_A', 'LISA_E'],
        tdi=shared_context['tdi'],
        t_obs_start=shared_context['t_obs_start'],
        delta_f=shared_context['delta_f'],
        f_final=shared_context['f_final'],
        mode_array=shared_context['mode_array'],
        t_offset=shared_context['t_offset'],
        cache_generator=shared_context['cache_generator'],
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
    fd_stacked = bufs['fd_stacked']
    fd_stacked[:, 0, :] = fd_A * whiten_A.conj()
    fd_stacked[:, 1, :] = fd_E * whiten_E.conj()
    td_stacked = bufs['td_stacked']
    td_stacked[...] = cp.fft.irfft(fd_stacked * scale, n=n_time, axis=-1)  # (B,2,Nt)

    Nt = td_stacked.shape[-1]
    nctf = int(shared_context['cutoff_time'] * shared_context['sample_rate'])
    nfz_vec = forward_zeroes.astype(cp.int32)

    td_stacked = _ZERO_TD_F64(td_stacked.reshape(-1), nfz_vec, Nt, 2, nctf).reshape(td_stacked.shape)

    waveforms_fd = bufs['waveforms_fd']
    waveforms_fd[...] = cp.fft.rfft(td_stacked, axis=-1) / scale  # (B,2,Nf)

    data_stack = cp.stack([
        cp.asarray(shared_context['lisa_a_strain_fd'].data),
        cp.asarray(shared_context['lisa_e_strain_fd'].data)
    ], axis=0)  # (2,Nf)
    data_stack_conj = data_stack.conj()  # (2,Nf)
    inner = cp.sum(waveforms_fd * data_stack_conj[cp.newaxis, :, :], axis=-1)  # (B,2)
    overlap = 4.0 * df * inner
    sigmasq = 4.0 * df * cp.sum((waveforms_fd.conj() * waveforms_fd).real, axis=-1)  # (B,2) real
    log_likelihood_values = overlap.real.sum(axis=-1) - 0.5 * sigmasq.sum(axis=-1)

    # Save results and timings
    return log_likelihood_values



def log_likelihood(particles: cp.ndarray, context) -> cp.ndarray:
    # Compute the log-likelihood in batches to avoid memory issues
    batch_size = context.shared["batch_size"]
    # Check prior bounds
    valid = cp.all((particles >= context.shared["bounds"][:, 0]) & (particles <= context.shared["bounds"][:, 1]), axis=1)
    # Initialize logl array
    logl = cp.full(particles.shape[0], -cp.inf)
    # If no valid particles, return early
    if not cp.any(valid):
        return logl
    
    # Smaller helper to process a single batch
    def logl_fn(batch, valid_mask):
        logl_batch = cp.full(batch.shape[0], -cp.inf)
        # Convert to a parameter dictionary
        params = convert_to_parameters(batch[valid_mask], context.shared)
        logl_batch[valid_mask] = _log_likelihood_batch(params, context.shared)
        return logl_batch
    
    # Process in batches
    n_particles = particles.shape[0]
    n_batches = (n_particles + batch_size - 1) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_particles)
        logl[start:end] = logl_fn(particles[start:end], valid[start:end])

    return logl

def log_prior(particles: cp.ndarray, context) -> cp.ndarray:
    """Uniform prior over a bounded region."""
    lower_bound = context.shared["bounds"][:, 0]
    upper_bound = context.shared["bounds"][:, 1]
    valid = (particles >= lower_bound) & (particles <= upper_bound)
    # sum over dimensions; assign 0 if within bounds else -inf
    return cp.where(valid, 0, -cp.inf).sum(axis=-1)


# Mass conversion utilities
def eta_from_mass1_mass2(mass1, mass2):
    """Returns the symmetric mass ratio from mass1 and mass2."""
    return mass1*mass2 / (mass1 + mass2)**2.

def mass1_from_mchirp_q(mchirp, q):
    """Returns the primary mass from the given chirp mass and mass ratio."""
    mass1 = q**(2./5.) * (1.0 + q)**(1./5.) * mchirp
    return mass1

def mass2_from_mchirp_q(mchirp, q):
    """Returns the secondary mass from the given chirp mass and mass ratio."""
    mass2 = q**(-3./5.) * (1.0 + q)**(1./5.) * mchirp
    return mass2

def chirp_mass_mass_ratio_to_component_masses(chirp_mass: cp.ndarray, mass_ratio: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Convert chirp mass and mass ratio to component masses."""
    mass1 = mass1_from_mchirp_q(chirp_mass, mass_ratio)
    mass2 = mass2_from_mchirp_q(chirp_mass, mass_ratio)
    return mass1, mass2

def component_masses_to_chirp_mass_mass_ratio(mass1: cp.ndarray, mass2: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Convert component masses to chirp mass and mass ratio."""
    mass_ratio = mass2 / mass1
    chirp_mass = eta_from_mass1_mass2(mass1, mass2)**(3./5) * (mass1 + mass2)
    return chirp_mass, mass_ratio


def convert_to_parameters(particles, shared_context) -> dict:
    """Convert from particle array to parameter dictionary."""
    field_names = [
        'mass1','mass2','spin1z','spin2z',
        'distance','eclipticlongitude','eclipticlatitude',
        'inclination','polarization','coa_phase','tc'
    ]
    n = len(particles)
    p = {name: cp.zeros(n, dtype=cp.float64) for name in field_names}
    # Hardcoded for now but this can be made more general later
    if "chirp_mass" in shared_context["parameters"] and "mass_ratio" in shared_context["parameters"]:
        chirp_mass = particles[:, shared_context["parameters"].index("chirp_mass")]
        mass_ratio = particles[:, shared_context["parameters"].index("mass_ratio")]
        p['mass1'][:], p['mass2'][:] = chirp_mass_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
    # Set fixed parameters
    for k, v in shared_context["fixed_parameters"].items():
        p[k][:] = v
    return p


def main() -> None:
    # Create a CuPy RNG
    rng = ArrayRNG(backend='cupy', seed=42)

    # Enable logging
    configure_logging()

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
    shared_context['cache_generator'] = False

    orig.initialization(shared_context)

    # Waveform/global params
    shared_context['tdi'] = '1.5'
    shared_context['t_obs_start'] = shared_context['tlen']
    shared_context['f_final'] = shared_context['sample_rate'] / 2
    shared_context['approximant'] = 'BBHX_PhenomD'
    shared_context['t_offset'] = 7365189.431698299
    shared_context['mode_array'] = [(2,2)]

    chirp_mass_bounds = (8e5, 9e5)  # in solar masses
    mass_ratio_bounds = (0.5, 1.0)   # dimensionless
    shared_context["bounds"] = cp.array([
        chirp_mass_bounds,
        mass_ratio_bounds
    ], dtype=cp.float64)


    shared_context["parameters"] = [
        "chirp_mass",
        "mass_ratio"
    ]
    # Fixed parameters set to true values
    shared_context["fixed_parameters"] = {
        "spin1z": 0.0,
        "spin2z": 0.0,
        "distance": 27658.011507544677,
        "eclipticlongitude": 3.448296944257913,
        "eclipticlatitude": 0.44491231446252155,
        "inclination": 0.9238365050097769,
        "polarization": 3.4236020095353483,
        "coa_phase": 2.661901610522322,
        "tc": int(30*86400),    # 1931852406.9997194 is the true coalescence time in the original data
    }

    # Print true chirp mass and mass ratio
    mass1 = mass2 = 1e6
    true_chirp_mass, true_mass_ratio = component_masses_to_chirp_mass_mass_ratio(mass1, mass2)
    print(f"True chirp mass: {true_chirp_mass:.6e}, True mass ratio: {true_mass_ratio:.6e}")

    # Set batch size for likelihood evaluations
    shared_context["batch_size"] = 50

    # Test logl with out-of-bounds particles
    x_test = cp.array([
        [8.6e5, 0.6],    # valid
        [9.1e5, 0.7],    # invalid (chirp mass too high)
        [8.7e5, -4.0],    # invalid (mass ratio too high)
        [8.8e5, 0.8]     # valid
    ], dtype=cp.float64)
    context = type('Context', (), {'shared': shared_context})
    log_likelihood(x_test, context=context)

    dims = len(shared_context["parameters"])

    # Initialize particles from the prior
    # Real runs will need more
    n_particles = 200
    initial_particles = rng.uniform(low=shared_context["bounds"][:, 0], high=shared_context["bounds"][:, 1], size=(n_particles, dims))
    initial_log_weights = cp.full((n_particles,), -cp.log(cp.array(n_particles)))
    initial_state = ParticleState(
        particles=initial_particles,
        log_weights=initial_log_weights
    )

    # Use the prebuilt SMC with MiniPCN mutation
    # MiniPCN is the only sampler that supports the array API backends
    smc, _, base_shared_context = make_standard_bayesian_smc(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        target_ess_ratio=0.7,
        max_temperature=1.0,
        min_delta=1e-8,
        always_resample=True,
        mutation="minipcn",
        mutation_n_steps=20,    # May need more steps
        particle_transform="affine",
        auto_normalize=True,
    )

    shared_context.update(base_shared_context)

    result = smc.run(initial_state, rng=rng, shared_context=shared_context)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    result.plot_log_evidence(ax=axs[0])
    result.plot_metadata_history("temperature", ax=axs[1])
    fig.savefig("state.png")

    fig = result.plot_corner(
        truths=[true_chirp_mass, true_mass_ratio], bins=30
    )
    fig.savefig("posterior.png")


if __name__ == "__main__":
    main()