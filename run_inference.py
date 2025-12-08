"""Run inference for LISA SMBH binary signals using SMC with GPU acceleration.

Requires installing sequince package from: https://github.com/sequince-dev/sequince

You may also need to install emcee to get it working (its meant to be optional
but we missed an import).
"""
import argparse
from pathlib import Path
from run_single_likelihood_batch_optimized import _bbhx_fd, _get_buffers, _ZERO_TD_F64, log_likelihood_optimized
import run_single_likelihood_batch as orig  # reuse initialization
from config import load_config

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import h5py

from sequince import ParticleState
from sequince.steps.base import StepContext
from sequince.logging_utils import configure_logging
from sequince.prebuilt_samplers import make_standard_bayesian_smc
from sequince.callbacks import StatePlot, TracePlot
from sequince.posterior import draw_additional_samples
from orng import ArrayRNG


def create_parser():
    parser = argparse.ArgumentParser(description="Run LISA SMBH binary inference with SMC on GPU.")
    parser.add_argument("config", type=Path, help="Path to the configuration YAML file.")
    return parser


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
    if "tc" in shared_context["parameters"]:
        p['tc'][:] = particles[:, shared_context["parameters"].index("tc")]
    if "eclipticlongitude" in shared_context["parameters"]:
        p['eclipticlongitude'][:] = particles[:, shared_context["parameters"].index("eclipticlongitude")]
    if "sin_eclipticlatitude" in shared_context["parameters"]:
        sin_elat = particles[:, shared_context["parameters"].index("sin_eclipticlatitude")]
        p['eclipticlatitude'][:] = cp.arcsin(sin_elat)
    elif "eclipticlatitude" in shared_context["parameters"]:
        p['eclipticlatitude'][:] = particles[:, shared_context["parameters"].index("eclipticlatitude")]
    if "cos_inclination" in shared_context["parameters"]:
        cos_inc = particles[:, shared_context["parameters"].index("cos_inclination")]
        p['inclination'][:] = cp.arccos(cos_inc)
    elif "inclination" in shared_context["parameters"]:
        p['inclination'][:] = particles[:, shared_context["parameters"].index("inclination")]
    if "polarization" in shared_context["parameters"]:
        p['polarization'][:] = particles[:, shared_context["parameters"].index("polarization")]
    if "coa_phase" in shared_context["parameters"]:
        p['coa_phase'][:] = particles[:, shared_context["parameters"].index("coa_phase")]
    if "spin1z" in shared_context["parameters"]:
        p['spin1z'][:] = particles[:, shared_context["parameters"].index("spin1z")]
    if "spin2z" in shared_context["parameters"]:
        p['spin2z'][:] = particles[:, shared_context["parameters"].index("spin2z")]
    if "distance" in shared_context["parameters"]:
        p['distance'][:] = particles[:, shared_context["parameters"].index("distance")]
    # Set fixed parameters
    for k, v in shared_context["fixed_parameters"].items():
        p[k][:] = v
    return p


def initialize_shared_context(cutoff_days: float) -> dict:
    shared_context = {}
    shared_context['tlen'] = 2592000
    shared_context['sample_rate'] = 0.2
    shared_context['delta_f'] = 1./shared_context['tlen']
    shared_context['delta_t'] = 5
    shared_context['flen'] = shared_context['tlen']//2 + 1
    # Cutoff time for pre-merger
    shared_context['cutoff_time'] = 86400*cutoff_days
    shared_context['kernel_length'] = 17280
    shared_context['extra_forward_zeroes'] = 8640
    shared_context['data_file'] = 'signal_0_new.hdf'
    shared_context['psd_file'] = 'model_AE_TDI1_SMOOTH_optimistic.txt.gz'
    shared_context['cache_generator'] = False

    print("Initializing data and PSDs...")
    orig.initialization(shared_context)

    # Waveform/global params
    shared_context['tdi'] = '1.5'
    shared_context['t_obs_start'] = shared_context['tlen']
    shared_context['f_final'] = shared_context['sample_rate'] / 2
    shared_context['approximant'] = 'BBHX_PhenomD'
    shared_context['mode_array'] = [(2,2)]
    return shared_context


def get_true_parameters() -> dict:
    true_parameters = {
        "mass1": 1e6,
        "mass2": 1e6,
        "spin1z": 0.0,
        "spin2z": 0.0,
        "distance": 27658.011507544677,
        "eclipticlongitude": 3.448296944257913,
        "eclipticlatitude": 0.44491231446252155,
        "inclination": 0.9238365050097769,
        "polarization": 3.4236020095353483,
        "coa_phase": 2.661901610522322,
        "tc": int(30*86400),    # True coalescence time in the original data
    }
    return true_parameters


def get_prior_bounds() -> dict:
    prior_bounds = {
        "chirp_mass": (8e5, 9e5),
        "mass_ratio": (0.5, 1.0),
        "tc": (30*86400 - 1800, 30*86400 + 1800),
        "eclipticlongitude": (0, 2*cp.pi),
        "sin_eclipticlatitude": (-1, 1),
        "cos_inclination": (-1, 1),
        "polarization": (0, cp.pi),
        "coa_phase": (0, 2*cp.pi),
        "spin1z": (-0.99, 0.99),
        "spin2z": (-0.99, 0.99),
        "distance": (1e4, 5e4),
    }
    return prior_bounds


def get_shared_context_and_true_parameters(cfg) -> tuple[dict, dict]:
    shared_context = initialize_shared_context(cfg.analysis.cutoff_days)
    true_parameters = get_true_parameters()
    prior_bounds = get_prior_bounds()
    known_parameters = [
        "chirp_mass",
        "mass_ratio",
        "tc",
        "spin1z",
        "spin2z",
        "eclipticlongitude",
        "sin_eclipticlatitude",
        "cos_inclination",
        "polarization",
        "coa_phase",
        "distance",
    ]

    if len(cfg.analysis.parameters) == 0:
        parameters = known_parameters
    else:
        parameters = cfg.analysis.parameters
    shared_context["parameters"] = parameters

    for param in shared_context["parameters"]:
        if param not in prior_bounds:
            raise ValueError(f"No prior bounds specified for parameter: {param}")
    shared_context["bounds"] = cp.array([prior_bounds[param] for param in shared_context["parameters"]], dtype=cp.float64)
    print(f"Prior bounds:\n{shared_context['bounds']}")
    # Create bounds array
    print(f"Sampling parameters: {shared_context['parameters']}")
    # Fixed parameters set to true values
    shared_context["fixed_parameters"] = {}
    for k, v in true_parameters.items():
        if k not in shared_context["parameters"]:
            if "mass" in k and ("chirp_mass" in shared_context["parameters"] or "mass_ratio" in shared_context["parameters"]):
                continue  # skip mass1/mass2 if sampling in mc/q
            if "eclipticlatitude" == k and "sin_eclipticlatitude" in shared_context["parameters"]:
                continue
            if "inclination" == k and "cos_inclination" in shared_context["parameters"]:
                continue
            shared_context["fixed_parameters"][k] = v

    print(f"Fixed parameters:\n{shared_context['fixed_parameters']}")

    # Compute derived true parameters
    true_parameters["chirp_mass"], true_parameters["mass_ratio"] = component_masses_to_chirp_mass_mass_ratio(
        true_parameters["mass1"], true_parameters["mass2"]
    )
    true_parameters["sin_eclipticlatitude"] = np.sin(true_parameters["eclipticlatitude"])
    true_parameters["cos_inclination"] = np.cos(true_parameters["inclination"])
    return shared_context, true_parameters


def main(args) -> None:
    # Create a CuPy RNG
    rng = ArrayRNG(backend='cupy', seed=42)

    cfg = load_config(args.config)

    # Enable logging
    configure_logging()

    shared_context, true_parameters = get_shared_context_and_true_parameters(cfg)

    # Set batch size for likelihood evaluations
    shared_context["batch_size"] = cfg.analysis.batch_size

    dims = len(shared_context["parameters"])

    label = f"sampling_parameters_" + "_".join(shared_context["parameters"]) + cfg.label_suffix
    outdir = Path(cfg.outdir) / f"{cfg.analysis.cutoff_days}days" / label
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {outdir}")
    # Save config for reference
    with open(outdir / "config_used.yaml", "w") as f:
        import yaml
        yaml.dump(cfg.model_dump(), f)

    # Plot the likelihood surface for chirp_mass with all other params fixed
    chirp_mass_values = cp.linspace(8e5, 9e5, 200, dtype=cp.float64)
    context = StepContext(0, shared=shared_context)
    particles = cp.full((len(chirp_mass_values), len(shared_context["parameters"])), cp.nan, dtype=cp.float64)
    for i, param in enumerate(shared_context["parameters"]):
        particles[:, i] = true_parameters[param]
    particles[:, shared_context["parameters"].index("chirp_mass")] = chirp_mass_values
    # Map polarization to [0, pi]
    if "polarization" in shared_context["parameters"]:
        pol_idx = shared_context["parameters"].index("polarization")
        particles[:, pol_idx] = particles[:, pol_idx] % cp.pi

    logp = log_prior(particles, context)
    print("Log prior:", logp)

    # Check which parameters are out of bounds
    out_of_bounds = cp.any((particles < shared_context["bounds"][:, 0]) | (particles > shared_context["bounds"][:, 1]), axis=0)
    print(f"Out of bounds: {[k for k, out in zip(shared_context['parameters'], out_of_bounds) if out]}", )

    logl = log_likelihood(particles, context)

    print("Log likelihood:", logl)
    plt.figure(figsize=(8,5))
    plt.plot(cp.asnumpy(chirp_mass_values), cp.asnumpy(logl))
    plt.axvline(true_parameters["chirp_mass"], color='r', linestyle='--', label='True value')
    plt.xlabel("Chirp Mass")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood vs Chirp Mass (other params fixed)")
    plt.legend()
    plt.savefig(outdir / "likelihood_chirp_mass.png")


    if cfg.skip_sampling:
        print("Skipping sampling as per configuration.")
        return

    # Initialize particles from the prior
    # Real runs will need more
    print("Initializing particles...")
    n_particles = cfg.sampler.n_particles
    initial_particles = rng.uniform(low=shared_context["bounds"][:, 0], high=shared_context["bounds"][:, 1], size=(n_particles, dims))
    initial_log_weights = cp.full((n_particles,), -cp.log(cp.array(n_particles)))
    initial_state = ParticleState(
        particles=initial_particles,
        log_weights=initial_log_weights
    )

    print("Setting up SMC sampler...")
    # Use the prebuilt SMC with MiniPCN mutation
    # MiniPCN is the only sampler that supports the array API backends
    smc, _, base_shared_context = make_standard_bayesian_smc(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        target_ess_ratio=cfg.sampler.target_ess_ratio,
        max_temperature=1.0,
        min_delta=1e-8,
        always_resample=True,
        mutation="minipcn",
        mutation_n_steps=cfg.sampler.mutation_n_steps,    # May need more steps
        particle_transform="affine",
        auto_normalize=True,
        callbacks=[
            TracePlot(outdir=outdir, filename="trace.png", parameter_labels=shared_context["parameters"]),
            StatePlot(["temperature", "log_evidence"], outdir=outdir, filename="state.png")
        ]
    )

    shared_context.update(base_shared_context)

    print("Running SMC inference...")
    result = smc.run(initial_state, rng=rng, shared_context=shared_context)
    print("SMC inference completed.")

    if cfg.sampler.n_final_particles is not None:
        print("Drawing additional samples from prior for mutation kernel...")
        posterior_state = draw_additional_samples(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            result=result,
            rng=rng,
            n_particles=cfg.sampler.n_final_particles,
            mutation="minipcn",
            mutation_n_steps=cfg.sampler.mutation_n_steps,
        )

        posterior_samples = posterior_state.particles
    else:
        posterior_samples = result.posterior_samples()

    with h5py.File(outdir / "posterior_samples.h5", "w") as f:
        for i, param in enumerate(shared_context["parameters"]):
            f.create_dataset(param, data=cp.asnumpy(posterior_samples[:, i]))
        # Save prior bounds as attributes
        bounds_grp = f.create_group("prior_bounds")
        for i, param in enumerate(shared_context["parameters"]):
            bounds_grp.attrs[param] = cp.asnumpy(shared_context["bounds"][i])
        # Save fixed parameters
        fixed_grp = f.create_group("fixed_parameters")
        for k, v in shared_context["fixed_parameters"].items():
            fixed_grp.attrs[k] = v
        # Save true parameters
        true_grp = f.create_group("true_parameters")
        for k, v in true_parameters.items():
            true_grp.attrs[k] = cp.asnumpy(v)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    result.plot_log_evidence(ax=axs[0])
    result.plot_metadata_history("temperature", ax=axs[1])
    fig.savefig(outdir / "state.png")
    truths = [true_parameters[param] for param in shared_context["parameters"]]
    fig = result.plot_corner(
        truths=truths,
        bins=30,
        labels=shared_context["parameters"],
    )
    fig.savefig(outdir / "posterior.png")


if __name__ == "__main__":
    main(create_parser().parse_args())
