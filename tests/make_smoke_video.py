"""Create a short local MP4 from validation images for pipeline checks only."""

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    images = sorted(args.images_dir.glob("*.jpg"))[:20]
    if not images or args.frames <= 0 or args.fps <= 0:
        raise SystemExit("Need JPEG images, positive frame count, and positive FPS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 writer")
    try:
        for frame_id in range(args.frames):
            source = cv2.imread(str(images[frame_id % len(images)]), cv2.IMREAD_COLOR)
            if source is None:
                raise RuntimeError(f"Could not read {images[frame_id % len(images)]}")
            source_height, source_width = source.shape[:2]
            scale = min(width / source_width, height / source_height)
            target_width = round(source_width * scale)
            target_height = round(source_height * scale)
            resized = cv2.resize(source, (target_width, target_height))
            frame = cv2.copyMakeBorder(
                resized,
                (height - target_height) // 2,
                height - target_height - (height - target_height) // 2,
                (width - target_width) // 2,
                width - target_width - (width - target_width) // 2,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            cv2.putText(frame, f"source frame {frame_id}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()
    print(f"Created {args.output}: {args.frames} frames at {args.fps:g} FPS")


if __name__ == "__main__":
    main()
