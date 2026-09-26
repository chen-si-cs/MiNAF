# Processed data contract

MiNAF expects one processed directory per scene:

```text
data/processed/<scene>/
  log_magnitude.h5
  phase_spectrum.h5
  features.npy
  feature_stats.npz
  split_indices.pkl
  preprocess_manifest.json
```

Use `scripts/preprocess_soundspaces.py` or `scripts/preprocess_gwa.py` to create these files. See [PREPROCESSING.md](PREPROCESSING.md) for raw-data layouts and commands.

## Spectral HDF5 files

`log_magnitude.h5` and `phase_spectrum.h5` contain matching root keys.

Metadata datasets:

- `mean`: scene-level spectrum mean with shape `(2, frequency_bins, time_frames)`.
- `std`: scene-level population standard deviation with the same shape.
- `min_len`: minimum resampled waveform length used for the scene.

Every other key is a WAV filename. Each value uses `(channels, frequency_bins, time_frames)` ordering and is stored as `float32`. With `n_fft=512`, `frequency_bins=257`.

The phase file stores standardized instantaneous frequency rather than wrapped phase. Training and evaluation integrate it back into phase during reconstruction.

WAV filenames are processed in sorted order. Split indices refer to that same HDF5 key order.

## Geometric features

Each point feature has the layout:

```text
[distances | local means | local variances | surface normals | occupancy counts]
```

For `N` rays and `K` occupancy thresholds, the total feature length is `6N + K`:

- `N` nearest-hit distances.
- `N` local distance means.
- `N` local distance variances.
- `3N` nearest-hit surface-normal components.
- `K` occupancy counts.

### SoundSpaces

`features.npy` is a `float32` matrix indexed by the integer point IDs encoded in filenames such as `13_27_90.wav`.

Default shape:

```text
(number_of_points, 6152)
```

Use `dataset_type=soundspaces`, `n_rays=1024`, and `n_occlusion=8` when training.

### GWA

`features.npy` contains a pickled dictionary mapping the source/receiver names from `sim_config.json` to `float32` vectors.

Default vector length:

```text
6158
```

Use `dataset_type=gwa`, `n_rays=1024`, and `n_occlusion=14` when training.

The loader accepts both `S<number>` and `L<number>` source aliases to match GWA metadata and RIR filename conventions.

## Feature statistics

`feature_stats.npz` stores:

- `mean` and `std` used to normalize geometric features across scene points.
- `thresholds` used for occupancy counts.
- `n_rays`.
- `neighbor_count`.

These statistics are provided for inspection and future inference tooling. The current model consumes the already-normalized `features.npy` values.

## Split indices

`split_indices.pkl` contains NumPy index arrays:

```python
{
    "train": ...,
    "val": ...,
    "test": ...,
}
```

The preprocessing scripts use a seeded NumPy generator, making the splits reproducible for the same sorted WAV list and parameters.

## Manifest

`preprocess_manifest.json` records the dataset type, scene, audio count, STFT parameters, ray settings, occupancy thresholds, split ratios, seed, and compatibility mode.

It does not contain absolute filesystem paths or machine/user identifiers.
