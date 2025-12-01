from config import InferenceConfig
from run_inference import get_shared_context_and_true_parameters, log_likelihood
import matplotlib.pyplot as plt
from pathlib import Path
import cupy as cp
from sequince.steps.base import StepContext


def main():

    cfg = InferenceConfig(
        outdir=Path("likelihood_plots"),
        label_suffix="",
        sampler={},
        analysis={"batch_size": 10, "cutoff_days": cp.nan},
    )
    outdir = cfg.outdir
    outdir.mkdir(exist_ok=True)

    cutoff_days = cp.arange(1.0, 10.0, dtype=cp.float64)

    logl_vals = {}

    for cutoff in cutoff_days:

        cutoff_str = f"{cutoff:.1f}"

        # Initialize shared context and true parameters
        cfg.analysis.cutoff_days = float(cutoff)
        shared_context, true_parameters = get_shared_context_and_true_parameters(cfg)
        shared_context["batch_size"] = cfg.analysis.batch_size

        # Plot the likelihood surface for chirp_mass with all other params fixed
        chirp_mass_values = cp.linspace(8e5, 9e5, 200, dtype=cp.float64)
        context = StepContext(0, shared=shared_context)
        particles = cp.full((len(chirp_mass_values), len(shared_context["parameters"])), cp.nan, dtype=cp.float64)
        # Fill other params with true values
        for i, param in enumerate(shared_context["parameters"]):
            particles[:, i] = true_parameters[param]
        # Set chirp_mass values
        particles[:, shared_context["parameters"].index("chirp_mass")] = chirp_mass_values
        # Map polarization to [0, pi]
        if "polarization" in shared_context["parameters"]:
            pol_idx = shared_context["parameters"].index("polarization")
            particles[:, pol_idx] = particles[:, pol_idx] % cp.pi

        # Compute log-likelihoods
        logl = log_likelihood(particles, context)
        # Store for later plotting
        logl_vals[cutoff_str] = logl

        plt.figure(figsize=(8,5))
        plt.plot(cp.asnumpy(chirp_mass_values), cp.asnumpy(logl))
        plt.axvline(true_parameters["chirp_mass"], color='r', linestyle='--', label='True value')
        plt.xlabel("Chirp Mass")
        plt.ylabel("Log-Likelihood")
        plt.title("Log-Likelihood vs Chirp Mass (other params fixed)")
        plt.legend()
        plt.savefig(outdir / f"likelihood_chirp_mass_{cutoff_str}.png")

    # Plot log-likelihoods for different cutoff times
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    for i, (cutoff_str, logl) in enumerate(logl_vals.items()):
        color = cmap(i / len(logl_vals))
        plt.plot(cp.asnumpy(chirp_mass_values), cp.asnumpy(logl), label=f'Cutoff: {cutoff_str} days', color=color)
    plt.axvline(true_parameters["chirp_mass"], color='k', linestyle='--', label='True Chirp Mass')
    plt.xlabel('Chirp Mass')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood vs Chirp Mass for Different Cutoff Times')
    plt.yscale("symlog")
    plt.grid()
    plt.tight_layout()
    plt.savefig(outdir / 'log_likelihood_chirp_mass_all_cutoffs.png')

if __name__ == "__main__":
    main()