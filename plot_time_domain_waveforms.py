import os
import cupy as cp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from BBHX_Phenom_GPU import _bbhx_fd


def generate_td_waveforms(params, shared):
    waveforms = _bbhx_fd(
        ifos=['LISA_A', 'LISA_E'],
        tdi=shared['tdi'],
        t_obs_start=shared['tlen'],
        delta_f=shared['delta_f'],
        f_final=shared['f_final'],
        mode_array=shared['mode_array'],
        t_offset=shared['t_offset'],
        **params,
    )
    fd_A = cp.asarray(waveforms['LISA_A'])
    fd_E = cp.asarray(waveforms['LISA_E'])
    if fd_A.ndim == 1:
        fd_A = fd_A[cp.newaxis, :]
    if fd_E.ndim == 1:
        fd_E = fd_E[cp.newaxis, :]
    fd_stacked = cp.stack([fd_A, fd_E], axis=1)  # (B, 2, Nf)

    Nf = fd_stacked.shape[-1]
    Nt = 2 * (Nf - 1)
    df = shared['delta_f']
    scale = Nt * df  # equals 1/dt
    td = cp.fft.irfft(fd_stacked * scale, n=Nt, axis=-1)  # (B, 2, Nt)
    return td


def main():
    # Shared config matching run_single_likelihood_batch
    shared = {}
    shared['tlen'] = 2592000
    shared['sample_rate'] = 0.2
    shared['delta_f'] = 1.0 / shared['tlen']
    shared['f_final'] = shared['sample_rate'] / 2
    shared['tdi'] = '1.5'
    shared['t_offset'] = 7365189.431698299
    shared['mode_array'] = [(2, 2)]

    # Baseline physical parameters (single example; batch dim optional)
    base_tc = 1931852406.9997194 - 1893024018
    params = {
        'mass1': cp.array([1_000_000.0]),
        'mass2': cp.array([1_000_000.0]),
        'spin1z': cp.array([0.0]),
        'spin2z': cp.array([0.0]),
        'distance': cp.array([27658.011507544677]),
        'eclipticlongitude': cp.array([3.448296944257913]),
        'eclipticlatitude': cp.array([0.44491231446252155]),
        'inclination': cp.array([0.9238365050097769]),
        'polarization': cp.array([3.4236020095353483]),
        'coa_phase': cp.array([2.661901610522322]),
        'tc': cp.array([base_tc]),
    }

    # Variations of end time (tc) within +/- 1 day
    day = 86400
    offsets = [-day, -day//2, 0, day//2, day]

    # Generate baseline and variants
    variants = []  # list of (label, td_waveforms_cp)
    td_base = generate_td_waveforms(params, shared)
    variants.append((f"dt={0}s", td_base))

    for off in offsets:
        if off == 0:
            continue
        p = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in params.items()}
        p['tc'] = cp.array([base_tc + off])
        td = generate_td_waveforms(p, shared)
        variants.append((f"dt={off}s", td))

    # Prepare output directory
    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)

    # Time axis
    dt = 1.0 / shared['sample_rate']  # 5 s
    Nt = td_base.shape[-1]
    t = np.arange(Nt) * dt  # seconds

    # Select batch index for plotting (first)
    bidx = 0

    # Plot full series for A and E
    for ch, ch_name in enumerate(['LISA_A', 'LISA_E']):
        plt.figure(figsize=(12, 4))
        for label, td_cp in variants:
            y = cp.asnumpy(td_cp[bidx, ch, :])
            plt.plot(t / 86400.0, y, label=label, linewidth=0.8)
        plt.xlabel('Time [days]')
        plt.ylabel('Strain')
        plt.title(f'Time-domain waveform {ch_name} (full)')
        plt.legend(loc='upper right', ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{ch_name}_full.png'), dpi=150)
        plt.close()

    # Zoom into last day
    zoom_secs = 86400
    zoom_samples = int(zoom_secs / dt)
    sl = slice(-zoom_samples, None)
    t_zoom = t[sl]

    for ch, ch_name in enumerate(['LISA_A', 'LISA_E']):
        plt.figure(figsize=(12, 4))
        for label, td_cp in variants:
            y = cp.asnumpy(td_cp[bidx, ch, sl])
            plt.plot((t_zoom - t_zoom[0]) / 3600.0, y, label=label, linewidth=0.9)
        plt.xlabel('Time in window [hours]')
        plt.ylabel('Strain')
        plt.title(f'Time-domain waveform {ch_name} (last day)')
        plt.legend(loc='upper right', ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'{ch_name}_last_day.png'), dpi=150)
        plt.close()

    # Combined A/E in one figure for last day (baseline + extremes)
    plt.figure(figsize=(12, 6))
    labels_to_plot = {"dt=-86400s", "dt=0s", "dt=86400s"}
    for label, td_cp in variants:
        if label not in labels_to_plot:
            continue
        yA = cp.asnumpy(td_cp[bidx, 0, sl])
        yE = cp.asnumpy(td_cp[bidx, 1, sl])
        plt.plot((t_zoom - t_zoom[0]) / 3600.0, yA, label=f"A {label}", linewidth=0.9)
        plt.plot((t_zoom - t_zoom[0]) / 3600.0, yE, label=f"E {label}", linewidth=0.9, linestyle='--')
    plt.xlabel('Time in window [hours]')
    plt.ylabel('Strain')
    plt.title('A and E (last day): baseline vs ±1 day tc')
    plt.legend(loc='upper right', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'AE_last_day_compare.png'), dpi=150)
    plt.close()

    print(f"Saved plots in {outdir}/: \n - LISA_A_full.png\n - LISA_E_full.png\n - LISA_A_last_day.png\n - LISA_E_last_day.png\n - AE_last_day_compare.png")


if __name__ == '__main__':
    main()
