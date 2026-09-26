import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.geometry import DEFAULT_GWA_THRESHOLDS
from preprocessing.pipelines import preprocess_gwa


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess GWA data for MiNAF.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-root")
    parser.add_argument("--audio-dir")
    parser.add_argument("--points-path")
    parser.add_argument("--mesh-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--target-sample-rate", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--n-rays", type=int, default=1024)
    parser.add_argument("--neighbor-count", type=int, default=8)
    parser.add_argument(
        "--occlusion-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_GWA_THRESHOLDS.tolist(),
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compression", choices=["none", "gzip", "lzf"], default="none")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument(
        "--exclude-missing-hits-from-occupancy",
        action="store_true",
        help="Use physically strict occupancy counts instead of legacy NAF behavior.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene = args.scene
    scene_root = Path(args.scene_root or Path("data") / "GWA" / scene)
    preprocess_gwa(
        scene=scene,
        audio_dir=Path(args.audio_dir or scene_root / "hybrid"),
        points_path=Path(args.points_path or scene_root / "hybrid" / "sim_config.json"),
        mesh_path=Path(args.mesh_path or scene_root / f"{scene}.obj"),
        output_dir=Path(args.output_dir or Path("data") / "processed" / scene),
        thresholds=np.asarray(args.occlusion_thresholds, dtype=np.float32),
        target_sample_rate=args.target_sample_rate,
        n_fft=args.n_fft,
        n_rays=args.n_rays,
        neighbor_count=args.neighbor_count,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        compression=None if args.compression == "none" else args.compression,
        overwrite=args.overwrite,
        skip_audio=args.skip_audio,
        skip_geometry=args.skip_geometry,
        skip_splits=args.skip_splits,
        count_missing_as_occluded=not args.exclude_missing_hits_from_occupancy,
    )


if __name__ == "__main__":
    main()
