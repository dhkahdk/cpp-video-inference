"""Validate two-stream frame accounting and bounded-queue CSV records."""

import argparse
import csv
from pathlib import Path

import cv2


def video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            raise RuntimeError(f"Empty frame in {path}")
        count += 1
    capture.release()
    return count


def check_stream(source: Path, output: Path, csv_path: Path,
                 queue_capacity: int) -> tuple[int, int, int, int]:
    source_count = video_frames(source)
    output_count = video_frames(output)
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    ids = [int(row["frame_id"]) for row in rows]
    statuses = [row["status"] for row in rows]
    processed = statuses.count("processed")
    dropped = statuses.count("dropped_oldest")
    depths = [int(row["queue_depth_after_enqueue"]) for row in rows]
    if len(rows) != source_count or sorted(ids) != list(range(source_count)):
        raise AssertionError(
            f"Source/CSV mismatch: source={source_count}, rows={len(rows)}, complete_ids={sorted(ids) == list(range(source_count))}"
        )
    if any(status not in {"processed", "dropped_oldest"} for status in statuses):
        raise AssertionError(f"Unexpected status in {csv_path}")
    if source_count != processed + dropped or output_count != processed:
        raise AssertionError(
            f"Accounting mismatch: source={source_count}, processed={processed}, "
            f"dropped={dropped}, output={output_count}"
        )
    if any(depth < 1 or depth > queue_capacity for depth in depths):
        raise AssertionError(f"Queue depth exceeded capacity {queue_capacity}")
    for row in rows:
        output_fields = ("output_frame_id", "output_unix_ms", "queue_wait_ms",
                         "capture_to_output_ms", "detections", "drawn_detections",
                         "preprocess_ms", "h2d_gpu_ms", "inference_gpu_ms",
                         "d2h_gpu_ms", "infer_wall_ms", "postprocess_ms",
                         "draw_ms", "write_ms")
        if row["status"] == "dropped_oldest" and any(row[field] for field in output_fields):
            raise AssertionError("Dropped row unexpectedly contains output fields")
    processed_rows = sorted((row for row in rows if row["status"] == "processed"),
                            key=lambda row: int(row["output_frame_id"]))
    output_ids = [int(row["output_frame_id"]) for row in processed_rows]
    source_ids_by_output = [int(row["frame_id"]) for row in processed_rows]
    if output_ids != list(range(processed)) or source_ids_by_output != sorted(source_ids_by_output):
        raise AssertionError("Processed output frame IDs are not sequential")
    return source_count, processed, dropped, output_count


def main() -> None:
    parser = argparse.ArgumentParser()
    for stream in (1, 2):
        parser.add_argument(f"--input{stream}", type=Path, required=True)
        parser.add_argument(f"--output{stream}", type=Path, required=True)
        parser.add_argument(f"--csv{stream}", type=Path, required=True)
    parser.add_argument("--require-drops", action="store_true")
    parser.add_argument("--queue-capacity", type=int, required=True)
    args = parser.parse_args()
    total_drops = 0
    for stream in (1, 2):
        result = check_stream(getattr(args, f"input{stream}"),
                              getattr(args, f"output{stream}"),
                              getattr(args, f"csv{stream}"), args.queue_capacity)
        total_drops += result[2]
        print(f"stream {stream}: source={result[0]}, processed/output={result[1]}, dropped={result[2]}")
    if args.require_drops and total_drops == 0:
        raise AssertionError("Stress run did not trigger the drop-oldest policy")
    print(f"PASS: two-stream accounting is complete; total dropped={total_drops}")


if __name__ == "__main__":
    main()
