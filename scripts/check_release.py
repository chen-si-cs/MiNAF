import argparse
import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_FILES = sorted(ROOT.rglob("*.py"))
REQUIRED_FILES = [
    ROOT / "README.md",
    ROOT / "DATA_FORMAT.md",
    ROOT / "environment.yml",
    ROOT / "conf" / "config.yaml",
]


def check_static() -> None:
    missing = [path for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {missing}")

    for path in PYTHON_FILES:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    print(f"Static check passed: {len(PYTHON_FILES)} Python files parsed.")


def check_model() -> None:
    sys.path.insert(0, str(ROOT))
    import torch

    from network import RIRNetwork

    batch_size = 2
    model = RIRNetwork(L=15, h=256, F=257, device="cpu").eval()
    with torch.no_grad():
        output = model(
            torch.randn(batch_size, 3),
            torch.randn(batch_size, 3),
            torch.randn(batch_size, 6158),
            torch.randn(batch_size, 6158),
            torch.tensor(0.25),
        )

    expected_shape = (batch_size, 257)
    if tuple(output.shape) != expected_shape:
        raise AssertionError(f"Expected {expected_shape}, got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError("Model output contains non-finite values")

    print(f"Model check passed: output shape {tuple(output.shape)}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="store_true",
        help="Import PyTorch and run a synthetic MiNAF forward pass.",
    )
    args = parser.parse_args()

    check_static()
    if args.model:
        check_model()


if __name__ == "__main__":
    main()
