import os, pickle, numpy as np, h5py, torch
from torch.utils.data import Dataset
from utils import get_loc_GWA, read_named_3d_points


class SoundDataset(Dataset):
    """
    * One set of spectrograms in log_magnitude.h5 / phase_spectrum.h5
    * split_indices.pkl holds three lists of int: train / val / test
    * HDF-5 handles opened lazily inside each worker ➜ picklable on Windows
    """

    def __init__(self, cfg, split: str = "train"):
        super().__init__()
        self.split = split
        data_dir = os.path.join(cfg.data_root, cfg.apt)

        self._log_path = os.path.join(data_dir, "log_magnitude.h5")
        self._phase_path = os.path.join(data_dir, "phase_spectrum.h5")
        feature_path = os.path.join(data_dir, "features.npy")
        split_path = os.path.join(data_dir, "split_indices.pkl")
        point_path = os.path.join(
            cfg.geometry_root, cfg.apt, "hybrid", "sim_config.json"
        )

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

        self.features = np.load(feature_path, allow_pickle=True).item()
        self.points = read_named_3d_points(point_path)

        with open(split_path, "rb") as f:
            idx = pickle.load(f)

        self.indices_map = {
            "train": idx["train"],
            "val": idx["val"],
            "test": idx["test"],
        }

        with h5py.File(self._log_path, "r") as h:
            reserved_keys = {"mean", "std", "min_len"}
            self.audio_names = [key for key in h.keys() if key not in reserved_keys]

        with (
            h5py.File(self._log_path, "r") as f_log,
            h5py.File(self._phase_path, "r") as f_phase,
        ):
            self.avg_log_mag = f_log["mean"][:]
            self.std_log_mag = f_log["std"][:]
            self.min_len = f_log["min_len"][()]
            self.avg_phase = f_phase["mean"][:]
            self.std_phase = f_phase["std"][:]

        self.orient_dict = {"0": 0, "90": 1, "180": 2, "270": 3}

        expected_feature_size = 6 * 1024 + 14
        invalid_features = {
            key: np.asarray(value).size
            for key, value in self.features.items()
            if np.asarray(value).size != expected_feature_size
        }
        if invalid_features:
            first_items = list(invalid_features.items())[:5]
            raise ValueError(
                f"Expected context features of length {expected_feature_size}; "
                f"examples with other sizes: {first_items}"
            )

        self._log_h5 = None
        self._phase_h5 = None

    def _ensure_open(self):
        if self._log_h5 is None:
            self._log_h5 = h5py.File(self._log_path, "r", swmr=True)
            self._phase_h5 = h5py.File(self._phase_path, "r", swmr=True)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_log_h5"] = st["_phase_h5"] = None
        return st

    def __len__(self):
        return len(self.indices_map[self.split])

    def __getitem__(self, idx):
        self._ensure_open()
        name_idx = self.indices_map[self.split][idx]
        name = self.audio_names[name_idx]

        log_mag = self._log_h5[name][:]  # (T,F)
        phase = self._phase_h5[name][:]  # (T,F)

        tx_idx, rx_idx, orient_str = get_loc_GWA(name)
        tx_loc, rx_loc = map(np.array, (self.points[tx_idx], self.points[rx_idx]))
        orientation = self.orient_dict[orient_str]

        try:
            tx_feat, rx_feat = self.features[tx_idx], self.features[rx_idx]
        except KeyError as e:
            tx_feat, rx_feat = self.features["S" + tx_idx[1:]], self.features[rx_idx]

        if self.split == "train":
            tx_loc += np.random.randn(3) * 5e-4
            rx_loc += np.random.randn(3) * 5e-4

        return (
            log_mag[:, :, :150].astype(np.float32),
            phase[:, :, :150].astype(np.float32),
            tx_loc.astype(np.float32),
            rx_loc.astype(np.float32),
            orientation,
            tx_feat.astype(np.float32),
            rx_feat.astype(np.float32),
        )

    def __del__(self):
        for h in [getattr(self, "_log_h5", None), getattr(self, "_phase_h5", None)]:
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass
