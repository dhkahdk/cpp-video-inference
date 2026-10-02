"""Compare the C++ input tensor with the existing Python validation preprocessing."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def python_reference(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    height, width = image.shape[:2]
    ratio = min(640 / height, 640 / width, 1.0)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    pad_w = (640 - resized_width) / 2
    pad_h = (640 - resized_height) / 2
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(
        image,
        round(pad_h - 0.1),
        round(pad_h + 0.1),
        round(pad_w - 0.1),
        round(pad_w + 0.1),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "input.bin"
        for image in args.images:
            subprocess.run(
                [str(args.exe), "--input", str(image), "--dump-tensor", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            actual = np.fromfile(output, dtype=np.float32).reshape(3, 640, 640)
            expected = python_reference(image)
            delta = np.abs(actual - expected)
            print(f"{image.name}: max_abs={delta.max():.9g}, mismatched={np.count_nonzero(delta > 1e-6)}")
            if np.any(delta > 1e-6):
                raise SystemExit(1)


if __name__ == "__main__":
    main()
