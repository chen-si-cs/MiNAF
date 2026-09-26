from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np

from .audio import list_wav_files, preprocess_audio
from .geometry import probe_named_points, probe_points


def read_soundspaces_points(path: Path) -> list[tuple[float, float, float]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SoundSpaces points file not found: {path}")
    points = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 4:
            raise ValueError(f"Invalid point at {path}:{line_number}: {line}")
        _, x, y, z = parts
        points.append((float(x), float(y), float(z)))
    if not points:
        raise ValueError(f"No points found in: {path}")
    return points


def read_gwa_points(path: Path) -> dict[str, tuple[float, float, float]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GWA sim_config.json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    points = {}
    for group in ("receivers", "sources"):
        for item in data.get(group, []):
            points[item["name"]] = tuple(float(value) for value in item["xyz"])
    if not points:
        raise ValueError(f"No receivers or sources found in: {path}")
    return points


def create_splits(
    n_total: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if n_total <= 0:
        raise ValueError("Cannot split an empty dataset")
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("Split ratios must be in [0, 1)")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    indices = np.arange(n_total)
    np.random.default_rng(seed).shuffle(indices)
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def _ensure_replaceable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite.")


def write_splits(
    output_dir: Path,
    n_total: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    path = Path(output_dir) / "split_indices.pkl"
    _ensure_replaceable(path, overwrite)
    splits = create_splits(n_total, val_ratio, test_ratio, seed)
    with path.open("wb") as stream:
        pickle.dump(splits, stream)
    return splits


def save_feature_outputs(
    output_dir: Path,
    features,
    mean: np.ndarray,
    std: np.ndarray,
    thresholds: np.ndarray,
    n_rays: int,
    neighbor_count: int,
    overwrite: bool,
) -> None:
    output_dir = Path(output_dir)
    feature_path = output_dir / "features.npy"
    stats_path = output_dir / "feature_stats.npz"
    _ensure_replaceable(feature_path, overwrite)
    _ensure_replaceable(stats_path, overwrite)
    np.save(feature_path, features)
    np.savez(
        stats_path,
        mean=mean,
        std=std,
        thresholds=np.asarray(thresholds, dtype=np.float32),
        n_rays=np.int64(n_rays),
        neighbor_count=np.int64(neighbor_count),
    )


def write_manifest(output_dir: Path, manifest: dict, overwrite: bool) -> None:
    path = Path(output_dir) / "preprocess_manifest.json"
    _ensure_replaceable(path, overwrite)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def preprocess_soundspaces(
    scene: str,
    audio_dir: Path,
    points_path: Path,
    mesh_path: Path,
    output_dir: Path,
    thresholds: np.ndarray,
    target_sample_rate: int,
    n_fft: int,
    n_rays: int,
    neighbor_count: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    compression: str | None,
    overwrite: bool,
    skip_audio: bool,
    skip_geometry: bool,
    skip_splits: bool,
    count_missing_as_occluded: bool,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_names = []
    if not skip_audio or not skip_splits:
        audio_names = [path.name for path in list_wav_files(audio_dir)]

    if not skip_audio:
        audio_names = preprocess_audio(
            audio_dir,
            output_dir,
            target_sample_rate=target_sample_rate,
            n_fft=n_fft,
            compression=compression,
            overwrite=overwrite,
        ).audio_names

    if not skip_geometry:
        features, mean, std = probe_points(
            read_soundspaces_points(points_path),
            mesh_path,
            thresholds,
            n_rays=n_rays,
            neighbor_count=neighbor_count,
            count_missing_as_occluded=count_missing_as_occluded,
        )
        save_feature_outputs(
            output_dir,
            features,
            mean,
            std,
            thresholds,
            n_rays,
            neighbor_count,
            overwrite,
        )

    if not skip_splits:
        write_splits(
            output_dir,
            len(audio_names),
            val_ratio,
            test_ratio,
            seed,
            overwrite,
        )

    write_manifest(
        output_dir,
        {
            "dataset_type": "soundspaces",
            "scene": scene,
            "audio_count": len(audio_names) if audio_names else None,
            "target_sample_rate": target_sample_rate,
            "n_fft": n_fft,
            "n_rays": n_rays,
            "neighbor_count": neighbor_count,
            "occlusion_thresholds": np.asarray(thresholds).tolist(),
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "legacy_missing_hit_occupancy": count_missing_as_occluded,
        },
        overwrite,
    )


def preprocess_gwa(
    scene: str,
    audio_dir: Path,
    points_path: Path,
    mesh_path: Path,
    output_dir: Path,
    thresholds: np.ndarray,
    target_sample_rate: int,
    n_fft: int,
    n_rays: int,
    neighbor_count: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    compression: str | None,
    overwrite: bool,
    skip_audio: bool,
    skip_geometry: bool,
    skip_splits: bool,
    count_missing_as_occluded: bool,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_names = []
    if not skip_audio or not skip_splits:
        audio_names = [path.name for path in list_wav_files(audio_dir)]

    if not skip_audio:
        audio_names = preprocess_audio(
            audio_dir,
            output_dir,
            target_sample_rate=target_sample_rate,
            n_fft=n_fft,
            compression=compression,
            overwrite=overwrite,
        ).audio_names

    if not skip_geometry:
        features, mean, std = probe_named_points(
            read_gwa_points(points_path),
            mesh_path,
            thresholds,
            n_rays=n_rays,
            neighbor_count=neighbor_count,
            count_missing_as_occluded=count_missing_as_occluded,
        )
        save_feature_outputs(
            output_dir,
            features,
            mean,
            std,
            thresholds,
            n_rays,
            neighbor_count,
            overwrite,
        )

    if not skip_splits:
        write_splits(
            output_dir,
            len(audio_names),
            val_ratio,
            test_ratio,
            seed,
            overwrite,
        )

    write_manifest(
        output_dir,
        {
            "dataset_type": "gwa",
            "scene": scene,
            "audio_count": len(audio_names) if audio_names else None,
            "target_sample_rate": target_sample_rate,
            "n_fft": n_fft,
            "n_rays": n_rays,
            "neighbor_count": neighbor_count,
            "occlusion_thresholds": np.asarray(thresholds).tolist(),
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "legacy_missing_hit_occupancy": count_missing_as_occluded,
        },
        overwrite,
    )
