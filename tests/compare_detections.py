"""Compare C++ boxes with the validation postprocess using stable score ties."""

import argparse
import csv
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def load_reference(path: Path, trt_lib: Path):
    os.environ["PATH"] = str(trt_lib) + os.pathsep + os.environ["PATH"]
    dll_directory = os.add_dll_directory(str(trt_lib)) if hasattr(os, "add_dll_directory") else None
    spec = importlib.util.spec_from_file_location("validation_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dll_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trt-lib", type=Path, required=True)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    reference, dll_directory = load_reference(args.reference, args.trt_lib)
    failures = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.bin"
            csv_path = Path(directory) / "detections.csv"
            for image_path in args.images:
                subprocess.run(
                    [str(args.exe), "--input", str(image_path), "--engine", str(args.engine),
                     "--dump-output", str(raw_path), "--dump-detections", str(csv_path)],
                    check=True, capture_output=True, text=True,
                )
                raw = np.fromfile(raw_path, dtype=np.float32).reshape(1, 9, 33600)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Could not decode {image_path}")
                _, ratio, pad = reference.letterbox(image)
                # The existing script uses NumPy's default argsort, whose tie
                # ordering differs from C++ for equal FP16 scores. Keep its
                # thresholds and NMS logic, but make equal-score order explicit.
                original_argsort = np.argsort
                np.argsort = lambda values, *a, **kw: original_argsort(
                    values, *a, kind="stable", **kw
                )
                try:
                    boxes, scores, classes = reference.postprocess_prediction(
                        raw, image.shape, ratio, pad
                    )
                finally:
                    np.argsort = original_argsort
                with csv_path.open(newline="", encoding="utf-8") as file:
                    cpp = list(csv.DictReader(file))
                if len(cpp) != len(boxes):
                    failures.append(f"{image_path.name}: count C++={len(cpp)}, Python={len(boxes)}")
                    continue
                unmatched = set(range(len(cpp)))
                max_box_difference = 0.0
                max_score_difference = 0.0
                for box, score, class_id in zip(boxes, scores, classes):
                    eligible = [index for index in unmatched
                                if int(cpp[index]["class_id"]) == int(class_id)]
                    if not eligible:
                        failures.append(f"{image_path.name}: missing class {class_id}")
                        break
                    def distance(index):
                        coordinates = np.array([float(cpp[index][key])
                                                for key in ("x1", "y1", "x2", "y2")])
                        return np.max(np.abs(coordinates - box))
                    best = min(eligible, key=distance)
                    max_box_difference = max(max_box_difference, distance(best))
                    max_score_difference = max(
                        max_score_difference, abs(float(cpp[best]["score"]) - float(score))
                    )
                    unmatched.remove(best)
                print(f"{image_path.name}: boxes={len(boxes)}, "
                      f"max_box_abs={max_box_difference:.6g}px, "
                      f"max_score_abs={max_score_difference:.6g}")
                if max_box_difference > 0.01 or max_score_difference > 1e-5:
                    failures.append(f"{image_path.name}: box difference {max_box_difference:.6g}px")
    finally:
        if dll_directory is not None:
            dll_directory.close()
    if failures:
        print("Failed comparisons:")
        for failure in failures:
            print("  " + failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
