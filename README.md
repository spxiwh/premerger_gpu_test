# LISA Pre-merger GPU Likelihood Pipeline

This repository contains GPU-accelerated implementations of the LISA pre-merger likelihood calculation, with progressive optimizations for improved performance.

## Overview

The pipeline computes log-likelihoods for batched gravitational waveform parameters using:
- BBHx waveform generation (GPU-accelerated)
- Frequency-domain whitening
- Time-domain manipulation (zeroing, windowing)
- Overlap calculations for likelihood evaluation

## Files

### Main Pipeline Implementations

- **`run_single_likelihood_batch.py`** - Reference implementation with clear, readable code
  - Uses Python loops for time-domain zeroing
  - Per-channel overlap computations
  - ~350 ms per 100-waveform batch (after warm-up)

- **`run_single_likelihood_batch_optimized.py`** - First optimized version
  - Fused TD zeroing kernel (ElementwiseKernel)
  - Batched overlaps across channels
  - Buffer preallocation and reuse
  - ~237 ms per 100-waveform batch (1.5x speedup)

- **`run_single_likelihood_batch_optimized2.py`** - Second optimized version
  - Additional fused whitening+scaling kernel
  - Cached GPU data (PSDs, data conjugates)
  - Custom ReductionKernels for overlaps and sigmasq
  - ~246 ms per 100-waveform batch (1.5x speedup)

### Testing and Benchmarking Scripts

- **`run_compare_pipelines.py`** - Validates correctness across all implementations
  - Runs all three implementations 50 times each
  - Compares outputs with tight tolerances (rtol=1e-12, atol=1e-12)
  - Reports timing and accuracy metrics

- **`run_single_likelihood_batch_profiled.py`** - Detailed performance profiling
  - Runs warm-up calls to eliminate JIT compilation overhead
  - Breaks down timing by stage (waveform generation, whitening, zeroing, FFTs, overlaps)
  - Compares all three implementations side-by-side

- **`test_singularity.sh`** - Container environment test
  - Tests reference implementation in shared Docker environment
  - Uses Singularity to run `ghcr.io/uk-lisa-gs/shared_code_environment:latest-cuda12`
  - Verifies GPU access and dependencies

### Supporting Files

- **`BBHX_Phenom_GPU.py`** - BBHx waveform generator wrapper
- **`pre_merger_utils.py`** - Utility functions for PSD generation and data preprocessing
- **`space_coords.py`** - Coordinate transformation utilities
- **`sourcings.sh`** - Environment setup for local development

### Data Files

- **`signal_0.hdf`** - Sample LISA strain data
- **`model_AE_TDI1_SMOOTH_optimistic.txt.gz`** - PSD model for whitening

## Quick Start

### Local Development (with conda)

```bash
# Activate environment
conda activate lisa_gpu_premerger
source sourcings.sh

# Run reference implementation
python run_single_likelihood_batch.py

# Run optimized version
python run_single_likelihood_batch_optimized.py

# Run optimized2 version
python run_single_likelihood_batch_optimized2.py
```

### Testing in Shared Environment (Singularity)

```bash
# First time setup: pulls Docker image and converts to Singularity
./test_singularity.sh

# Subsequent runs use cached Singularity image
./test_singularity.sh
```

### Validation and Benchmarking

```bash
# Compare all implementations for correctness (50 runs each)
python run_compare_pipelines.py

# Detailed performance profiling with stage breakdowns
python run_single_likelihood_batch_profiled.py
```

## Performance Summary

Benchmarks for 100-waveform batches (after warm-up):

| Implementation | Time (ms) | Speedup | Notes |
|----------------|-----------|---------|-------|
| Reference      | 350-360   | 1.0x    | Clear, readable baseline |
| Optimized      | 235-240   | 1.5x    | Fused zeroing, batched overlaps |
| Optimized2     | 245-250   | 1.5x    | Custom reduction kernels |

**Key optimizations:**
- Fused TD zeroing kernel: **112x faster** than Python loop
- Batched overlap computation: reduces kernel launches
- Buffer preallocation: eliminates repeated allocations
- Cached GPU data: avoids redundant CPU→GPU transfers

**Accuracy:**
All optimized versions maintain numerical equivalence with reference:
- Max relative error: < 1e-15
- Mean relative error: < 1e-16

## Stage-by-Stage Breakdown (Profiled)

After warm-up, typical timings for 100-waveform batch:

### Reference
- Waveform generation: ~223 ms (dominates)
- Whiten + iRFFT: ~14 ms
- Zeroing (Python loop): ~100 ms
- rFFT: ~11 ms
- Overlaps: ~7 ms

### Optimized
- Whiten + iRFFT: ~15 ms
- Zeroing (fused kernel): **~1 ms**
- rFFT: ~10 ms
- Overlaps: ~8 ms

### Optimized2
- Whiten + iRFFT: ~16 ms
- Zeroing (fused kernel): **~1 ms**
- rFFT: ~11 ms
- Overlaps (custom kernels): **~2 ms**

## Output

All scripts produce:
- Log-likelihood values for each waveform in the batch
- Time-domain waveform plots (saved to `plots/`, `plots_opt/`, or `plots_opt2/`)
- Timing breakdowns (when applicable)

Example output locations:
- Reference plots: `plots/waveform_td_*.png`
- Optimized plots: `plots_opt/waveform_td_*.png`
- Optimized2 plots: `plots_opt2/waveform_td_*.png`

## Requirements

### Local Development
- CUDA 12.x
- CuPy (GPU-accelerated NumPy)
- PyCBC
- BBHx (cuda12x build)
- matplotlib (for plotting)

### Shared Environment
All dependencies pre-installed in:
```
ghcr.io/uk-lisa-gs/shared_code_environment:latest-cuda12
```

## Notes

- All implementations use `direct=True` in BBHx to avoid segmentation faults
- Batch size is fixed at 100 waveforms
- Time-domain cutoff: 7 days
- Sample rate: 0.2 Hz
- Observation duration: 30 days (2,592,000 seconds)

## Troubleshooting

**BBHx segfault:**
Ensure `direct=True` and `length=None` in BBHx configuration.

**GPU memory issues:**
Reduce batch size in the `build_params()` function.

**Singularity permissions:**
Ensure `--nv` flag is used for GPU access.

## Contact

For questions about this implementation, please refer to the LISA pre-merger analysis paper and data release.
