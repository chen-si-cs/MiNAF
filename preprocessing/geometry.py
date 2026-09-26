from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh
from tqdm import tqdm


DEFAULT_SOUNDSPACES_THRESHOLDS = np.linspace(0.5, 10.0, 8, dtype=np.float32)
DEFAULT_GWA_THRESHOLDS = np.array(
    [
        0.10,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
        2.50,
        3.00,
        3.50,
        4.00,
        4.50,
        5.00,
    ],
    dtype=np.float32,
)


def fibonacci_directions(n_rays: int) -> np.ndarray:
    if n_rays <= 0:
        raise ValueError("n_rays must be positive")
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_rays, dtype=np.float64) + 0.5
    z = 1 - 2 * indices / n_rays
    radius = np.sqrt(1 - z * z)
    theta = 2 * np.pi * indices / golden_ratio
    return np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), z], axis=-1
    )


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    loaded = trimesh.load(mesh_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {mesh_path}, got {type(loaded)}")
    return loaded


def extract_features(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    thresholds: np.ndarray,
    n_rays: int = 1024,
    neighbor_count: int = 8,
    missing_distance: float = -0.1,
    count_missing_as_occluded: bool = True,
) -> np.ndarray:
    directions = fibonacci_directions(n_rays)
    ray_origins = np.repeat(
        np.asarray(origin, dtype=np.float64).reshape(1, 3), n_rays, axis=0
    )
    locations, ray_indices, triangle_indices = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=directions,
        multiple_hits=True,
    )

    distances = np.full(n_rays, missing_distance, dtype=np.float64)
    normals = np.zeros((n_rays, 3), dtype=np.float64)

    if len(locations):
        raw_distances = np.linalg.norm(
            locations - ray_origins[ray_indices], axis=1
        )
        order = np.lexsort((raw_distances, ray_indices))
        sorted_rays = ray_indices[order]
        first_mask = np.concatenate([[True], sorted_rays[1:] != sorted_rays[:-1]])
        first_indices = order[first_mask]
        first_rays = ray_indices[first_indices]
        distances[first_rays] = raw_distances[first_indices]
        normals[first_rays] = mesh.face_normals[triangle_indices[first_indices]]

    tree = cKDTree(directions)
    query_count = min(n_rays, neighbor_count + 1)
    _, neighbor_indices = tree.query(directions, k=query_count)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]

    local_distances = distances[neighbor_indices]
    valid_mask = local_distances != missing_distance
    valid_counts = valid_mask.sum(axis=1)
    sums = np.where(valid_mask, local_distances, 0.0).sum(axis=1)
    means = np.full(n_rays, missing_distance, dtype=np.float64)
    np.divide(sums, valid_counts, out=means, where=valid_counts > 0)

    centered = np.where(valid_mask, local_distances - means[:, None], 0.0)
    variance_sums = np.square(centered).sum(axis=1)
    variances = np.full(n_rays, missing_distance, dtype=np.float64)
    np.divide(
        variance_sums,
        valid_counts,
        out=variances,
        where=valid_counts > 0,
    )

    occupancy_distances = distances
    if not count_missing_as_occluded:
        occupancy_distances = distances[distances != missing_distance]
    occupancy = np.array(
        [np.sum(occupancy_distances < threshold) for threshold in thresholds],
        dtype=np.float64,
    )

    return np.concatenate(
        [distances, means, variances, normals.reshape(-1), occupancy]
    )


def normalize_features(
    raw_features: np.ndarray, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = raw_features.mean(axis=0)
    std = raw_features.std(axis=0)
    normalized = (raw_features - mean) / (std + epsilon)
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def probe_points(
    points: list[tuple[float, float, float]] | np.ndarray,
    mesh_path: Path,
    thresholds: np.ndarray,
    n_rays: int = 1024,
    neighbor_count: int = 8,
    count_missing_as_occluded: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = load_mesh(mesh_path)
    raw_features = np.stack(
        [
            extract_features(
                mesh,
                point,
                thresholds,
                n_rays=n_rays,
                neighbor_count=neighbor_count,
                count_missing_as_occluded=count_missing_as_occluded,
            )
            for point in tqdm(points, desc="Probing geometry")
        ]
    )
    return normalize_features(raw_features)


def probe_named_points(
    points: dict[str, tuple[float, float, float]],
    mesh_path: Path,
    thresholds: np.ndarray,
    n_rays: int = 1024,
    neighbor_count: int = 8,
    count_missing_as_occluded: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    names = list(points)
    normalized, mean, std = probe_points(
        [points[name] for name in names],
        mesh_path,
        thresholds,
        n_rays=n_rays,
        neighbor_count=neighbor_count,
        count_missing_as_occluded=count_missing_as_occluded,
    )
    return {name: normalized[index] for index, name in enumerate(names)}, mean, std
