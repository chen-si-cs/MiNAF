# MiNAF: Explicit Context-Driven Neural Acoustic Modeling for High-Fidelity RIR Generation

<div align="center">

[Chen Si](https://chen-si-cs.github.io/)<sup>1</sup> &emsp;
[Qianyi Wu](https://qianyiwu.github.io/)<sup>2</sup> &emsp;
[Chaitanya Amballa](https://achaitanya.web.illinois.edu/)<sup>3</sup> &emsp;
[Romit Roy Choudhury](https://croy.web.engr.illinois.edu/)<sup>3</sup>

<sup>1</sup>UC San Diego &emsp;
<sup>2</sup>Monash University &emsp;
<sup>3</sup>University of Illinois Urbana-Champaign

**Interspeech 2026 · Long Paper**

<a href='https://chen-si-cs.github.io/projects/MiNAF/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://github.com/chen-si-cs/MiNAF'><img src='https://img.shields.io/badge/Code-GitHub-black'></a>
<a href='https://arxiv.org/abs/2509.15210'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<!-- Replace # with the official Interspeech PDF URL when available. -->
<a href='#'><img src='https://img.shields.io/badge/Interspeech-Official%20PDF%20Coming%20Soon-lightgrey'></a>

</div>

![MiNAF teaser](https://chen-si-cs.github.io/projects/MiNAF/figs/teaser.png)

---

## Overview

**MiNAF** (Mesh-infused Neural Acoustic Field) is a neural implicit model for high-fidelity room impulse response (RIR) generation. Given transmitter and receiver locations and a rough room mesh, MiNAF probes the local geometry with uniformly distributed rays and extracts explicit physical context including distances, surface normals, proximity statistics, and global occupancy information.

These geometric features are fused with positional and temporal embeddings to predict the log-magnitude and instantaneous-frequency spectra of the RIR. MiNAF provides accurate acoustic-field predictions across different room layouts and remains robust when training data are limited or the input meshes are noisy or reconstructed.

---

## 🗺️ Release Status

| Component | Status |
|---|---|
| **MiNAF model, training, and evaluation code** | ✅ Released |
| **SoundSpaces / Replica preprocessing** | ✅ Released |
| **GWA preprocessing** | ✅ Released |
| **Processed-data format documentation** | ✅ Released — see [DATA_FORMAT.md](DATA_FORMAT.md) |
| **Official Interspeech PDF** | 🔜 Coming soon |

---

## 1. Installation

### 1.1 Clone This Repository

```bash
git clone https://github.com/chen-si-cs/MiNAF.git
cd MiNAF
```

### 1.2 Create the Environment

```bash
conda env create -f environment.yml
conda activate minaf
```

Run the release checks:

```bash
python scripts/check_release.py
python scripts/check_release.py --model
python scripts/check_preprocessing.py
```

The checks validate both SoundSpaces and GWA feature layouts and run a synthetic preprocessing pipeline. They do not reproduce the paper results without the corresponding raw datasets and training runs.

---

## 2. Data Preparation

The repository includes dataset-specific preprocessing for both benchmarks used by MiNAF.

### 2.1 SoundSpaces / Replica

Place SoundSpaces RIRs, Replica metadata, and the room mesh under `data/`, then run:

```bash
python scripts/preprocess_soundspaces.py --scene office_4
```

SoundSpaces uses eight occupancy features and produces 6,152-dimensional context vectors. Train with:

```bash
python train_pl.py dataset_type=soundspaces apt=office_4 n_occlusion=8
```

### 2.2 GWA

Place a GWA scene under `data/GWA/<scene>/`, then run:

```bash
python scripts/preprocess_gwa.py --scene room_a
```

GWA uses 14 occupancy features and produces 6,158-dimensional context vectors. Train with:

```bash
python train_pl.py dataset_type=gwa apt=room_a n_occlusion=14
```

See [PREPROCESSING.md](PREPROCESSING.md) for raw-data layouts, custom paths, stage controls, and compatibility details. See [DATA_FORMAT.md](DATA_FORMAT.md) for the generated HDF5 and feature contracts.

---

## 3. Training

Train MiNAF on a processed GWA scene:

```bash
python train_pl.py \
  dataset_type=gwa \
  apt=room_a \
  n_occlusion=14 \
  exp_name=minaf_room_a
```

Example GPU configuration:

```bash
python train_pl.py \
  apt=room_a \
  dataset_type=gwa \
  n_occlusion=14 \
  data_root=/path/to/processed \
  geometry_root=/path/to/GWA \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.precision=16-mixed
```

Training logs and checkpoints are written under `results/<scene>/<experiment>/`.

---

## 4. Evaluation

Evaluate a trained checkpoint by setting `ckpt_path` in `conf/config.yaml` or overriding it from the command line:

```bash
python test_pl.py \
  dataset_type=gwa \
  apt=room_a \
  n_occlusion=14 \
  ckpt_path=/path/to/model.ckpt
```

The evaluation script reconstructs RIR waveforms and reports spectral and acoustic metrics, including T60, C50, EDT, SNR, and PSNR.

---

## Acknowledgements

The code structure, spectral preprocessing, and neural acoustic field workflow were adapted from [Learning Neural Acoustic Fields (NAF)](https://github.com/aluo-x/Learning_Neural_Acoustic_Fields) and subsequently modified for MiNAF, including the explicit geometric context encoder, GWA support, streaming preprocessing, and a refactor to PyTorch Lightning. Please cite both MiNAF and NAF when building on this implementation.

---

## License

This project is licensed under the **Apache License 2.0**.



---

## Citation

If you find MiNAF useful, please cite our work:

```bibtex
@inproceedings{si2026minaf,
  title     = {Explicit Context-Driven Neural Acoustic Modeling for High-Fidelity RIR Generation},
  author    = {Si, Chen and Wu, Qianyi and Amballa, Chaitanya and Roy Choudhury, Romit},
  booktitle = {Interspeech 2026},
  year      = {2026}
}
```
