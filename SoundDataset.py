import os
import pickle

import h5py
import numpy as np
from torch.utils.data import Dataset

from utils import get_loc, get_loc_GWA, read_3d_points, read_named_3d_points


class SoundDataset(Dataset):
    """Load processed MiNAF spectra and geometric context features."""

    RESERVED_H5_KEYS = {"mean", "std", "min_len"}

    def __init__(self, cfg, split: str = "train"):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.split = split
        self.dataset_type = str(getattr(cfg, "dataset_type", "gwa")).lower()
        if self.dataset_type not in {"gwa", "soundspaces"}:
            raise ValueError(
                f"dataset_type must be 'gwa' or 'soundspaces', got {self.dataset_type}"
            )

        data_dir = os.path.join(cfg.data_root, cfg.apt)
        self._log_path = os.path.join(data_dir, "log_magnitude.h5")
        self._phase_path = os.path.join(data_dir, "phase_spectrum.h5")
        feature_path = os.path.join(data_dir, "features.npy")
        split_path = os.path.join(data_dir, "split_indices.pkl")

        if self.dataset_type == "gwa":
            point_path = os.path.join(
                cfg.geometry_root, cfg.apt, "hybrid", "sim_config.json"
            )
        else:
            point_path = os.path.join(cfg.metadata_root, cfg.apt, "points.txt")

        required_paths = [
            self._log_path,
            self._phase_path,
            feature_path,
            split_path,
            point_path,
        ]
        missing_paths = [path for path in required_paths if not os.path.isfile(path)]
        if missing_paths:
            formatted = "\n".join(f"- {path}" for path in missing_paths)
            raise FileNotFoundError(f"Missing processed data files:\n{formatted}")

        feature_payload = np.load(feature_path, allow_pickle=True)
        if feature_payload.shape == ():
            self.features = feature_payload.item()
        else:
            self.features = feature_payload

        if self.dataset_type == "gwa":
            self.points = read_named_3d_points(point_path)
        else:
            self.points = read_3d_points(point_path)

        with open(split_path, "rb") as stream:
            split_indices = pickle.load(stream)
        self.indices_map = {
            name: np.asarray(split_indices[name], dtype=np.int64)
            for name in ("train", "val", "test")
        }

        with (
            h5py.File(self._log_path, "r") as log_h5,
            h5py.File(self._phase_path, "r") as phase_h5,
        ):
            self.audio_names = [
                key for key in log_h5.keys() if key not in self.RESERVED_H5_KEYS
            ]
            phase_names = [
                key for key in phase_h5.keys() if key not in self.RESERVED_H5_KEYS
            ]
            if self.audio_names != phase_names:
                raise ValueError("Magnitude and phase HDF5 files use different audio keys")

            self.avg_log_mag = log_h5["mean"][:]
            self.std_log_mag = log_h5["std"][:]
            self.min_len = log_h5["min_len"][()]
            self.avg_phase = phase_h5["mean"][:]
            self.std_phase = phase_h5["std"][:]

        for split_name, indices in self.indices_map.items():
            if len(indices) and (indices.min() < 0 or indices.max() >= len(self.audio_names)):
                raise IndexError(
                    f"{split_name} indices exceed the {len(self.audio_names)} audio samples"
                )

        self.orient_dict = {"0": 0, "90": 1, "180": 2, "270": 3}
        self.expected_feature_size = 6 * cfg.n_rays + cfg.n_occlusion
        self._validate_features()

        self._log_h5 = None
        self._phase_h5 = None

    def _validate_features(self):
        if isinstance(self.features, dict):
            invalid = {
                key: np.asarray(value).size
                for key, value in self.features.items()
                if np.asarray(value).size != self.expected_feature_size
            }
            if invalid:
                raise ValueError(
                    f"Expected context features of length {self.expected_feature_size}; "
                    f"examples with other sizes: {list(invalid.items())[:5]}"
                )
            return

        if self.features.ndim != 2 or self.features.shape[1] != self.expected_feature_size:
            raise ValueError(
                f"Expected feature matrix (*, {self.expected_feature_size}), "
                f"got {self.features.shape}"
            )

    def _ensure_open(self):
        if self._log_h5 is None:
            self._log_h5 = h5py.File(self._log_path, "r", swmr=True)
            self._phase_h5 = h5py.File(self._phase_path, "r", swmr=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_log_h5"] = None
        state["_phase_h5"] = None
        return state

    def __len__(self):
        return len(self.indices_map[self.split])

    def _gwa_feature(self, point_name: str) -> np.ndarray:
        candidates = [point_name]
        if point_name.startswith("L"):
            candidates.append("S" + point_name[1:])
        elif point_name.startswith("S"):
            candidates.append("L" + point_name[1:])
        for candidate in candidates:
            if candidate in self.features:
                return self.features[candidate]
        raise KeyError(f"No feature vector found for point {point_name}")

    def _metadata(self, name: str):
        if self.dataset_type == "gwa":
            tx_name, rx_name, orientation = get_loc_GWA(name)
            tx_loc = np.asarray(self.points[tx_name], dtype=np.float32)
            rx_loc = np.asarray(self.points[rx_name], dtype=np.float32)
            tx_feature = self._gwa_feature(tx_name)
            rx_feature = self._gwa_feature(rx_name)
        else:
            tx_index, rx_index, orientation = get_loc(name)
            tx_loc = np.asarray(self.points[tx_index], dtype=np.float32)
            rx_loc = np.asarray(self.points[rx_index], dtype=np.float32)
            tx_feature = self.features[tx_index]
            rx_feature = self.features[rx_index]

        return (
            tx_loc,
            rx_loc,
            self.orient_dict[orientation],
            np.asarray(tx_feature, dtype=np.float32),
            np.asarray(rx_feature, dtype=np.float32),
        )

    def __getitem__(self, idx):
        self._ensure_open()
        name_index = self.indices_map[self.split][idx]
        name = self.audio_names[name_index]
        log_magnitude = self._log_h5[name][:]
        phase = self._phase_h5[name][:]
        tx_loc, rx_loc, orientation, tx_feature, rx_feature = self._metadata(name)

        if self.split == "train":
            tx_loc = tx_loc + np.random.randn(3).astype(np.float32) * 5e-4
            rx_loc = rx_loc + np.random.randn(3).astype(np.float32) * 5e-4

        return (
            log_magnitude[:, :, :150].astype(np.float32),
            phase[:, :, :150].astype(np.float32),
            tx_loc,
            rx_loc,
            orientation,
            tx_feature,
            rx_feature,
        )

    def __del__(self):
        for handle in (
            getattr(self, "_log_h5", None),
            getattr(self, "_phase_h5", None),
        ):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
