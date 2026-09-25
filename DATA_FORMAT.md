# Processed data contract

The public code starts after geometric and acoustic preprocessing. It expects one directory per scene under `data_root` and one geometry JSON under `geometry_root`.

## Scene files

For a scene named `<scene>`:

```text
<data_root>/<scene>/log_magnitude.h5
<data_root>/<scene>/phase_spectrum.h5
<data_root>/<scene>/features.npy
<data_root>/<scene>/split_indices.pkl
<geometry_root>/<scene>/hybrid/sim_config.json
```

### `log_magnitude.h5`

- One HDF5 dataset per RIR filename.
- Each RIR tensor must use `(channels, frequency_bins, time_frames)` ordering.
- The loader currently keeps at most the first 150 time frames.
- Root datasets `mean`, `std`, and `min_len` store normalization metadata.
- For `n_fft=512`, `frequency_bins` must be 257.

### `phase_spectrum.h5`

- Uses the same RIR keys and tensor layout as `log_magnitude.h5`.
- Root datasets `mean` and `std` store normalization metadata.
- The released training code treats the stored phase target as instantaneous frequency and integrates it during reconstruction.

### `features.npy`

- A pickled Python dictionary loaded with `numpy.load(..., allow_pickle=True).item()`.
- Keys identify transmitter/receiver point names from the RIR filenames and geometry JSON.
- Values are float arrays with length 6158.
- Layout: 1,024 distances, 1,024 local means, 1,024 local variances, 3,072 normal components, and 14 occlusion/global values.

### `split_indices.pkl`

A dictionary with integer index lists:

```python
{
    "train": [...],
    "val": [...],
    "test": [...],
}
```

Indices address the non-metadata RIR keys in the HDF5 file's iteration order.

### `sim_config.json`

The loader expects a `points` collection that maps point names to 3D coordinates. See `read_named_3d_points` in `utils.py` for the exact accepted JSON structure.

## Filename convention

The current GWA parser expects names matching:

```text
L<digits>_R<digits>.wav
```

where `L...` is the transmitter and `R...` is the receiver.
