#!/usr/bin/env python
"""
Generate a fresh LISA signal file using BBHx waveforms.

This script generates time-domain LISA strain data (LISA_A and LISA_E channels)
from BBHx waveforms and saves them to an HDF5 file compatible with PyCBC's
TimeSeries format. No filtering, whitening, or other modifications are applied.

The default parameters match those used in the existing likelihood pipelines.
"""

import argparse
import numpy as np
import h5py
from pycbc.types import TimeSeries

from BBHX_Phenom_GPU import _bbhx_fd


def generate_signal(
    output_file='signal_0_new.hdf',
    # Source parameters
    mass1=1000000.0,
    mass2=1000000.0,
    spin1z=0.0,
    spin2z=0.0,
    distance=27658.011507544677,
    eclipticlongitude=3.448296944257913,
    eclipticlatitude=0.44491231446252155,
    inclination=0.9238365050097769,
    polarization=3.4236020095353483,
    coa_phase=2.661901610522322,
    tc=30*86400,  # 30 days in seconds
    # Waveform settings
    tdi='1.5',
    t_obs_start=2592000,  # 30 days
    t_offset=7365189.431698299,
    approximant='BBHX_PhenomD',
    mode_array=None,
    # Data settings
    sample_rate=0.2,  # Hz
    tlen=2592000,  # seconds (30 days)
    delta_t=5,  # seconds
):
    """
    Generate LISA strain data and save to HDF5 file.
    
    Parameters
    ----------
    output_file : str
        Output HDF5 filename
    mass1, mass2 : float
        Component masses in solar masses
    spin1z, spin2z : float
        Dimensionless spin components aligned with orbital angular momentum
    distance : float
        Luminosity distance in Mpc
    eclipticlongitude, eclipticlatitude : float
        Sky location in ecliptic coordinates (radians)
    inclination : float
        Inclination angle (radians)
    polarization : float
        Polarization angle (radians)
    coa_phase : float
        Coalescence phase (radians)
    tc : float
        Coalescence time (seconds, GPS time)
    tdi : str
        TDI version ('1.5' or '2.0')
    t_obs_start : float
        Observation start time before merger (seconds)
    t_offset : float
        Time offset for LISA frame
    approximant : str
        Waveform approximant ('BBHX_PhenomD' or 'BBHX_PhenomHM')
    mode_array : list or None
        List of (l,m) mode tuples; defaults to [(2,2)] for PhenomD
    sample_rate : float
        Sample rate in Hz
    tlen : float
        Total length of data in seconds
    delta_t : float
        Time spacing in seconds
    """
    
    if mode_array is None:
        mode_array = [(2, 2)]
    
    # Compute frequency parameters
    delta_f = 1.0 / tlen
    f_final = sample_rate / 2.0
    
    # Build scalar parameter dict (single waveform)
    params = {
        'mass1': mass1,
        'mass2': mass2,
        'spin1z': spin1z,
        'spin2z': spin2z,
        'distance': distance,
        'eclipticlongitude': eclipticlongitude,
        'eclipticlatitude': eclipticlatitude,
        'inclination': inclination,
        'polarization': polarization,
        'coa_phase': coa_phase,
        'tc': tc,
    }
    
    print(f"Generating BBHx waveform with parameters:")
    for k, v in params.items():
        print(f"  {k:20s} = {v}")
    print(f"  approximant          = {approximant}")
    print(f"  mode_array           = {mode_array}")
    print(f"  tdi                  = {tdi}")
    print(f"  t_obs_start          = {t_obs_start} s")
    print(f"  sample_rate          = {sample_rate} Hz")
    print(f"  delta_f              = {delta_f} Hz")
    print(f"  f_final              = {f_final} Hz")
    print()
    
    # Generate frequency-domain waveforms
    waveforms = _bbhx_fd(
        ifos=['LISA_A', 'LISA_E'],
        tdi=tdi,
        t_obs_start=t_obs_start,
        delta_f=delta_f,
        f_final=f_final,
        mode_array=mode_array,
        t_offset=t_offset,
        **params,
    )
    
    # Convert to NumPy arrays (handle CuPy → NumPy transfer if needed)
    # Check if the output is a CuPy array and transfer to CPU
    try:
        import cupy as cp
        if isinstance(waveforms['LISA_A'], cp.ndarray):
            fd_A = cp.asnumpy(waveforms['LISA_A'])
        else:
            fd_A = np.asarray(waveforms['LISA_A'])
        if isinstance(waveforms['LISA_E'], cp.ndarray):
            fd_E = cp.asnumpy(waveforms['LISA_E'])
        else:
            fd_E = np.asarray(waveforms['LISA_E'])
    except ImportError:
        # CuPy not available, assume NumPy arrays
        fd_A = np.asarray(waveforms['LISA_A'])
        fd_E = np.asarray(waveforms['LISA_E'])
    
    # If batched (2D), take first element; otherwise use as-is
    if fd_A.ndim == 2:
        fd_A = fd_A[0, :]
    if fd_E.ndim == 2:
        fd_E = fd_E[0, :]
    
    print(f"Generated FD waveforms:")
    print(f"  LISA_A shape: {fd_A.shape}")
    print(f"  LISA_E shape: {fd_E.shape}")
    print(f"  Max |LISA_A|: {np.max(np.abs(fd_A)):.3e}")
    print(f"  Max |LISA_E|: {np.max(np.abs(fd_E)):.3e}")
    print()
    
    # Convert to time domain via inverse FFT
    # Determine time-domain length
    Nfreq = len(fd_A)
    n_time = 2 * (Nfreq - 1)
    
    # Scaling to match PyCBC convention: FD → TD via irfft
    # BBHx outputs are in frequency-domain strain S(f)
    # For irfft: we want time-domain strain h(t)
    # The standard DFT relationship: multiply by (n_time * delta_f) before irfft
    scale = n_time * delta_f
    
    td_A = np.fft.irfft(fd_A * scale, n=n_time)
    td_E = np.fft.irfft(fd_E * scale, n=n_time)
    
    print(f"Converted to TD:")
    print(f"  LISA_A TD shape: {td_A.shape}")
    print(f"  LISA_E TD shape: {td_E.shape}")
    print(f"  Max |LISA_A TD|: {np.max(np.abs(td_A)):.3e}")
    print(f"  Max |LISA_E TD|: {np.max(np.abs(td_E)):.3e}")
    print()
    
    # Create PyCBC TimeSeries objects
    # epoch: set to 0 for simplicity (can be adjusted if needed)
    ts_A = TimeSeries(td_A, delta_t=delta_t, epoch=0)
    ts_E = TimeSeries(td_E, delta_t=delta_t, epoch=0)
    
    # Save to HDF5 file (matching signal_0.hdf format)
    print(f"Saving to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        # Save LISA_A as a dataset (not a group) with delta_t and start_time attrs
        dset_A = f.create_dataset('LISA_A', data=ts_A.data, compression='gzip')
        dset_A.attrs['delta_t'] = ts_A.delta_t
        dset_A.attrs['start_time'] = float(ts_A.start_time)
        
        # Save LISA_E as a dataset with delta_t and start_time attrs
        dset_E = f.create_dataset('LISA_E', data=ts_E.data, compression='gzip')
        dset_E.attrs['delta_t'] = ts_E.delta_t
        dset_E.attrs['start_time'] = float(ts_E.start_time)
        
        # Optionally save metadata in a separate group for reference
        meta = f.create_group('metadata')
        meta.attrs['mass1'] = mass1
        meta.attrs['mass2'] = mass2
        meta.attrs['spin1z'] = spin1z
        meta.attrs['spin2z'] = spin2z
        meta.attrs['distance'] = distance
        meta.attrs['eclipticlongitude'] = eclipticlongitude
        meta.attrs['eclipticlatitude'] = eclipticlatitude
        meta.attrs['inclination'] = inclination
        meta.attrs['polarization'] = polarization
        meta.attrs['coa_phase'] = coa_phase
        meta.attrs['tc'] = tc
        meta.attrs['tdi'] = tdi
        meta.attrs['t_obs_start'] = t_obs_start
        meta.attrs['t_offset'] = t_offset
        meta.attrs['approximant'] = approximant
        meta.attrs['sample_rate'] = sample_rate
        meta.attrs['delta_f'] = delta_f
        meta.attrs['f_final'] = f_final
    
    print(f"Successfully saved signal to {output_file}")
    print(f"  LISA_A: {len(td_A)} samples, delta_t={delta_t} s")
    print(f"  LISA_E: {len(td_E)} samples, delta_t={delta_t} s")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate LISA signal file from BBHx waveforms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output
    parser.add_argument('--output', '-o', type=str, default='signal_0_new.hdf',
                        help='Output HDF5 filename')
    
    # Source parameters
    parser.add_argument('--mass1', type=float, default=1000000.0,
                        help='Primary mass (solar masses)')
    parser.add_argument('--mass2', type=float, default=1000000.0,
                        help='Secondary mass (solar masses)')
    parser.add_argument('--spin1z', type=float, default=0.0,
                        help='Primary spin z-component')
    parser.add_argument('--spin2z', type=float, default=0.0,
                        help='Secondary spin z-component')
    parser.add_argument('--distance', type=float, default=27658.011507544677,
                        help='Luminosity distance (Mpc)')
    parser.add_argument('--eclipticlongitude', type=float, default=3.448296944257913,
                        help='Ecliptic longitude (radians)')
    parser.add_argument('--eclipticlatitude', type=float, default=0.44491231446252155,
                        help='Ecliptic latitude (radians)')
    parser.add_argument('--inclination', type=float, default=0.9238365050097769,
                        help='Inclination angle (radians)')
    parser.add_argument('--polarization', type=float, default=3.4236020095353483,
                        help='Polarization angle (radians)')
    parser.add_argument('--coa-phase', type=float, default=2.661901610522322,
                        help='Coalescence phase (radians)')
    parser.add_argument('--tc', type=float, default=30*86400,
                        help='Coalescence time (seconds, GPS)')
    
    # Waveform settings
    parser.add_argument('--tdi', type=str, default='1.5', choices=['1.5', '2.0'],
                        help='TDI version')
    parser.add_argument('--t-obs-start', type=float, default=2592000,
                        help='Observation start time before merger (seconds)')
    parser.add_argument('--t-offset', type=float, default=7365189.431698299,
                        help='LISA time offset')
    parser.add_argument('--approximant', type=str, default='BBHX_PhenomD',
                        choices=['BBHX_PhenomD', 'BBHX_PhenomHM'],
                        help='Waveform approximant')
    
    # Data settings
    parser.add_argument('--sample-rate', type=float, default=0.2,
                        help='Sample rate (Hz)')
    parser.add_argument('--tlen', type=float, default=2592000,
                        help='Total data length (seconds)')
    parser.add_argument('--delta-t', type=float, default=5,
                        help='Time step (seconds)')
    
    args = parser.parse_args()
    
    # Convert argparse namespace to dict and call generate_signal
    generate_signal(
        output_file=args.output,
        mass1=args.mass1,
        mass2=args.mass2,
        spin1z=args.spin1z,
        spin2z=args.spin2z,
        distance=args.distance,
        eclipticlongitude=args.eclipticlongitude,
        eclipticlatitude=args.eclipticlatitude,
        inclination=args.inclination,
        polarization=args.polarization,
        coa_phase=args.coa_phase,
        tc=args.tc,
        tdi=args.tdi,
        t_obs_start=args.t_obs_start,
        t_offset=args.t_offset,
        approximant=args.approximant,
        sample_rate=args.sample_rate,
        tlen=args.tlen,
        delta_t=args.delta_t,
    )


if __name__ == '__main__':
    main()
