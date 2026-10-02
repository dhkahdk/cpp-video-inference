"""Check that a processed MP4 and its CSV contain one entry per source frame."""

import argparse
import csv
from pathlib import Path

import cv2


def decode_count(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            raise RuntimeError(f"Empty frame in {path}")
        count += 1
    capture.release()
    return count, fps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    source_frames, source_fps = decode_count(args.input)
    output_frames, output_fps = decode_count(args.output)
    with args.csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    ids = [int(row["frame_id"]) for row in rows]
    timestamps = [float(row["media_pts_ms"]) for row in rows]
    if not (source_frames == output_frames == len(rows)):
        raise AssertionError(
            f"Frame counts differ: source={source_frames}, output={output_frames}, CSV={len(rows)}"
        )
    if ids != list(range(source_frames)):
        raise AssertionError("CSV frame IDs are not sequential from zero")
    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        raise AssertionError("Media timestamps are not monotonic")
    if abs(source_fps - output_fps) > 0.01:
        raise AssertionError(f"FPS differs: source={source_fps}, output={output_fps}")
    print(f"PASS: {source_frames} source frames = {output_frames} output frames = "
          f"{len(rows)} CSV rows; FPS {source_fps:g} -> {output_fps:g}")


if __name__ == "__main__":
    main()
