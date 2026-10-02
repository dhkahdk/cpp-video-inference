"""Run video_infer, sample resource use, and summarize its per-frame CSV files."""

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

import psutil


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return round(values[lower] + (values[upper] - values[lower]) * (position - lower), 3)


def gpu_memory_mib():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        return sum(int(line.strip()) for line in result.stdout.splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def summarize_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    processed = [row for row in rows if row.get("status", "processed") == "processed"]
    if not processed:
        raise ValueError(f"No processed frames: {path}")
    fields = [
        "decode_ms", "frame_handoff_ms", "queue_wait_ms", "preprocess_ms", "h2d_gpu_ms",
        "inference_gpu_ms", "d2h_gpu_ms", "infer_wall_ms", "postprocess_ms",
        "draw_ms", "write_ms", "capture_to_output_ms", "file_read_to_output_ms",
    ]
    timings = {}
    for field in fields:
        values = [float(row[field]) for row in processed if row.get(field)]
        if values:
            timings[field] = {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}
    first_column = "capture_unix_ms" if "capture_unix_ms" in rows[0] else "read_wall_unix_ms"
    last_column = "output_unix_ms" if "output_unix_ms" in rows[0] else "write_wall_unix_ms"
    duration_s = (max(int(row[last_column]) for row in processed) -
                  min(int(row[first_column]) for row in rows)) / 1000
    return {
        "source_frames": len(rows), "processed_frames": len(processed),
        "dropped_frames": sum(row.get("status") == "dropped_oldest" for row in rows),
        "active_file_interval_s": round(duration_s, 3),
        "effective_fps_active_interval": round(len(processed) / duration_s, 3) if duration_s > 0 else None,
        "timings_ms": timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples-csv", type=Path,
                        help="Save sampled process RSS and total GPU memory over time")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("Supply the video_infer command after --")
    baseline_gpu = gpu_memory_mib()
    start = time.perf_counter()
    process = subprocess.Popen(command)
    peak_rss = 0
    peak_private = 0
    peak_gpu = baseline_gpu
    next_gpu_sample = start
    samples = []
    while process.poll() is None:
        rss = None
        private = None
        try:
            memory = psutil.Process(process.pid).memory_info()
            rss = memory.rss
            private = getattr(memory, "private", None)
            peak_rss = max(peak_rss, rss)
            if private is not None:
                peak_private = max(peak_private, private)
        except psutil.Error:
            pass
        if time.perf_counter() >= next_gpu_sample:
            used = gpu_memory_mib()
            if used is not None:
                peak_gpu = max(peak_gpu or 0, used)
            samples.append({"elapsed_s": round(time.perf_counter() - start, 3),
                            "process_rss_mib": round(rss / 1048576, 2) if rss is not None else None,
                            "process_private_mib": round(private / 1048576, 2) if private is not None else None,
                            "device_memory_used_mib": used})
            next_gpu_sample = time.perf_counter() + 0.5
        time.sleep(0.05)
    elapsed = time.perf_counter() - start
    if process.returncode:
        raise SystemExit(f"video_infer failed with exit code {process.returncode}")
    results = [summarize_csv(path) for path in args.csv]
    early = [sample["process_rss_mib"] for sample in samples
             if 10 <= sample["elapsed_s"] < 20 and sample["process_rss_mib"] is not None]
    late = [sample["process_rss_mib"] for sample in samples
            if elapsed - 15 <= sample["elapsed_s"] < elapsed - 5 and
            sample["process_rss_mib"] is not None]
    early_private = [sample["process_private_mib"] for sample in samples
                     if 10 <= sample["elapsed_s"] < 20 and sample["process_private_mib"] is not None]
    late_private = [sample["process_private_mib"] for sample in samples
                    if elapsed - 15 <= sample["elapsed_s"] < elapsed - 5 and
                    sample["process_private_mib"] is not None]
    report = {
        "command": command, "process_wall_s_including_startup": round(elapsed, 3),
        "total_effective_fps_including_startup": round(
            sum(result["processed_frames"] for result in results) / elapsed, 3),
        "peak_process_rss_mib_sampled": round(peak_rss / 1048576, 2),
        "peak_process_private_mib_sampled": round(peak_private / 1048576, 2) if peak_private else None,
        "process_rss_10_to_20s_median_mib": round(statistics.median(early), 2) if early else None,
        "process_rss_late_window_median_mib": round(statistics.median(late), 2) if late else None,
        "process_rss_late_minus_early_mib":
            round(statistics.median(late) - statistics.median(early), 2) if early and late else None,
        "process_private_10_to_20s_median_mib":
            round(statistics.median(early_private), 2) if early_private else None,
        "process_private_late_window_median_mib":
            round(statistics.median(late_private), 2) if late_private else None,
        "process_private_late_minus_early_mib":
            round(statistics.median(late_private) - statistics.median(early_private), 2)
            if early_private and late_private else None,
        "resource_sample_count": len(samples),
        "gpu_memory_used_mib_system_baseline": baseline_gpu,
        "gpu_memory_used_mib_system_peak": peak_gpu,
        "gpu_memory_peak_delta_mib_approx":
            peak_gpu - baseline_gpu if peak_gpu is not None and baseline_gpu is not None else None,
        "streams": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.samples_csv:
        args.samples_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=[
                "elapsed_s", "process_rss_mib", "process_private_mib",
                "device_memory_used_mib"])
            writer.writeheader()
            writer.writerows(samples)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
