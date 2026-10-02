"""Repeat a local MP4 to create a longer synthetic pipeline-test input."""

import argparse
import math
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=float, help="Output FPS; defaults to source FPS")
    args = parser.parse_args()
    if args.frames <= 0 or args.input.resolve() == args.output.resolve():
        parser.error("Use a positive frame count and distinct input/output paths")
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    fps = args.fps if args.fps is not None else capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Input lacks valid FPS or dimensions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {args.output}")
    try:
        for frame_id in range(args.frames):
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Input video has no decodable frames")
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
    print(f"Created {args.output}: {args.frames} frames, {fps:g} FPS")


if __name__ == "__main__":
    main()
