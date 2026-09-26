# MiNAF preprocessing

The preprocessing code converts raw SoundSpaces or GWA scenes into the HDF5 spectra, normalized geometric features, and deterministic split indices consumed by `SoundDataset.py`.

The implementation is derived from the original MiNAF notebooks and the NAF data flow, but removes notebook state, hard-coded scene names, machine-specific paths, exploratory plots, and full-dataset in-memory arrays.

## Processing stages

Each dataset command performs three stages:

1. **Audio:** resample RIRs to 22.05 kHz, duplicate mono RIRs to two channels, truncate every RIR to the scene minimum length, compute a 512-point STFT, convert phase to instantaneous frequency, standardize spectra, and write HDF5 files.
2. **Geometry:** cast Fibonacci-sphere rays from every source/receiver point, retain the nearest mesh hit, compute local distance statistics and occupancy counts, normalize features across scene points, and save `features.npy`.
3. **Splits:** deterministically shuffle the sorted HDF5/audio order and write train/validation/test indices.

Audio statistics are accumulated with a streaming algorithm, so the pipeline does not keep every RIR spectrum in memory.

## SoundSpaces / Replica

Expected default layout:

```text
data/
  binaural_rirs/<scene>/*.wav
  metadata/<scene>/points.txt
  mesh/<scene>/habitat/mesh_semantic.ply
```

Run:

```bash
python scripts/preprocess_soundspaces.py --scene office_4
```

If the Replica mesh is stored elsewhere:

```bash
python scripts/preprocess_soundspaces.py \
  --scene office_4 \
  --mesh-path /path/to/office_4/mesh.ply
```

SoundSpaces defaults to 1,024 rays, eight nearest-ray neighbors, eight occupancy thresholds linearly spaced from 0.5 m to 10 m, and an 80/5/15 split. The resulting feature length is `6 * 1024 + 8 = 6152`.

Train with matching model settings:

```bash
python train_pl.py \
  dataset_type=soundspaces \
  apt=office_4 \
  n_occlusion=8
```

## GWA

Expected default layout:

```text
data/GWA/<scene>/
  <scene>.obj
  hybrid/
    sim_config.json
    *.wav
```

Run:

```bash
python scripts/preprocess_gwa.py --scene room_a
```

GWA defaults to 1,024 rays, eight nearest-ray neighbors, 14 occupancy thresholds from the original GWA notebook, and an 85/5/10 split. The resulting feature length is `6 * 1024 + 14 = 6158`.

Train with matching model settings:

```bash
python train_pl.py \
  dataset_type=gwa \
  apt=room_a \
  n_occlusion=14
```

## Custom paths

Both commands accept explicit `--audio-dir`, `--points-path`, `--mesh-path`, and `--output-dir` arguments. GWA also accepts `--scene-root`.

Use `--skip-audio`, `--skip-geometry`, or `--skip-splits` to run only selected stages. Existing outputs are protected by default; pass `--overwrite` to replace them.

## Compatibility behavior

The default occupancy calculation preserves the original NAF/MiNAF notebook behavior, where rays with no mesh hit use a distance sentinel of `-0.1` and therefore contribute to occupancy thresholds. To exclude missing rays from occupancy counts, pass:

```bash
--exclude-missing-hits-from-occupancy
```

Changing this option changes the generated features and is not compatible with checkpoints trained using the legacy behavior.

The preprocessing manifest records only dataset parameters and scene identifiers. It deliberately does not store absolute source paths, usernames, hostnames, or other machine-specific information.

## Verification

```bash
python scripts/check_preprocessing.py
```

This creates temporary synthetic WAV files, checks STFT/IF HDF5 generation and deterministic splits, and validates the geometric feature layout.
