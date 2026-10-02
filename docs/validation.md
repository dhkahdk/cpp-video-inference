# Local validation record

## Public model-free check

`python tests/make_smoke_image.py --output outputs/smoke.ppm` generated a local 971×543 synthetic image. On 2026-10-02, `python tests/compare_preprocess.py --exe build/Release/video_infer.exe outputs/smoke.ppm` reported `max_abs=0, mismatched=0` on the RTX 4060 Laptop GPU Windows build. This check requires no research model and can be repeated with the published code. It tests image decoding and preprocessing, not TensorRT inference or video throughput.

Date: 2026-10-02. Hardware: RTX 4060 Laptop GPU. CUDA 12.4, TensorRT 10.11.0.33, C++ OpenCV 4.13.0. This record is for the initial single-image milestone, not a video performance result.

Reference engine: private local FP16 TensorRT engine. Its file and hash are withheld while the related paper is unpublished.

| Private validation image | C++ vs Python preprocessing max absolute difference | C++ vs Python TensorRT raw output max absolute difference |
| --- | ---: | ---: |
| A | 0 | 0 |
| B | 0 | 0 |
| C | 0 | 0 |

Preprocessing comparison used `tests/compare_preprocess.py` with Python OpenCV 4.13.0. Raw inference comparison used `tests/compare_raw_output.py` with Python TensorRT 10.11.0.33. Both tests read images from the existing local validation set and kept tensor files temporary or in ignored `outputs/`. The raw inference comparison fed the same C++-generated input tensor to both TensorRT runtimes, isolating the inference result from image decoding differences.

## Detection-box comparison

Twenty private validation JPEGs were processed with the same FP16 engine. C++ box decoding, coordinate restoration, and class-aware NMS were compared with the local Python reference using stable sorting for equal FP16 scores. All 20 images matched in detection count and class; maximum matched box-coordinate difference was at most `0.000005` pixel, and maximum score difference was below `0.000000001`.

The unmodified Python script uses NumPy's default `argsort`, which has no guaranteed tie order. Two private images selected neighboring overlapping boxes with identical scores, yielding up to about 4.53 pixels of coordinate difference; a third returned 88 boxes versus 89 with stable ordering. Re-running that same Python postprocess with stable `argsort` matched the C++ output. This is a tie-policy difference, not a model-output difference. The original research script and its results were not changed.

## Single-video file smoke test

Generated a **synthetic** 50-frame, 1280×720, 25 FPS MP4 from the first 20 local validation JPEGs with `tests/make_smoke_video.py`; each frame carries its source frame ID. The generated video is ignored by Git and is only a pipeline check, not a natural-motion benchmark clip.

The C++ program processed the complete file with the local FP16 engine and produced an annotated MP4 plus per-frame CSV. `tests/check_video_output.py` decoded both videos and verified: **50 input frames = 50 output frames = 50 CSV rows**, sequential frame IDs `0..49`, monotonic media timestamps, and 25 FPS input/output metadata. The run reported 3,368 post-NMS detections, of which 435 met the display threshold of 0.25 and were drawn. A decoded output frame was visually inspected for box rendering.

These counts show frame correspondence for this short file. They do not establish real-time throughput, live capture-to-output latency, long-run stability, or detection accuracy on moving footage.

## Two-stream bounded-queue checks

Two synthetic local streams were generated from the same validation JPEG set. They are reproducible pipeline inputs rather than natural-motion benchmark footage. Producers were paced using video FPS metadata and shared one TensorRT inference consumer.

- Normal arrival check: two 50-frame, 1280×720, 25 FPS streams, queue capacity 4. Both streams decoded and processed all 50 frames and dropped 0. Maximum observed queue depth was 1 for stream 0 and 2 for stream 1.
- Deliberate congestion check: two 100-frame, 1280×720, 200 FPS streams, queue capacity 3. In the recorded run, stream 0 processed 51 and dropped 49; stream 1 processed 56 and dropped 44. Both maximum queue depths were exactly 3, so the queues stayed within the configured bound. Exact processed/drop counts can vary with host scheduling; the invariant under test is complete accounting and the configured queue bound.

`tests/check_two_stream_output.py` decoded each source and output MP4 and checked every CSV row. For both runs, every source frame was classified as `processed` or `dropped_oldest`, processed frame count equaled output video frame count, output IDs were sequential, dropped rows contained no output fields, and recorded queue depth never exceeded capacity.

The 200 FPS case exists only to force overload and verify policy. It is not a performance claim.

## Initial segmented timing and resource sampling

Date: 2026-10-02. Same RTX 4060 Laptop GPU, FP16 engine, OpenCV 4.13.0, CUDA 12.4, TensorRT 10.11.0.33, 1280×720 synthetic videos. These are short smoke benchmarks, each run once, with no warmup exclusion or confidence interval. Keep the ignored `outputs/benchmark_*.csv` and `.json` locally for the full per-frame data and exact command. The executable was compiled from the D-drive source using MSVC Release in a separate writable build directory because CMake could not update a timestamp in the existing D-drive `build/` directory.

| Mode | Processed / decoded | Drops | Active-file FPS | FPS including startup | Latency P50 / P95 | Pure GPU inference P50 / P95 | Peak process RSS | Approx. GPU memory increase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single sequential, 25 FPS metadata, unpaced | 50 / 50 | 0 | 49.85 | 20.19 | file read→write 18.24 / 25.07 ms | 3.78 / 4.40 ms | 504.08 MiB | 196 MiB |
| Two paced streams, each 25 FPS, stream 0 | 50 / 50 | 0 | 25.77 | 18.18 combined | queue admission→write 10.74 / 89.43 ms | 4.57 / 5.83 ms | 583.16 MiB combined | 239 MiB combined |
| Two paced streams, each 25 FPS, stream 1 | 50 / 50 | 0 | 25.67 | 18.18 combined | queue admission→write 12.53 / 24.50 ms | 4.16 / 5.37 ms | same process | same process |

The single-video run is intentionally unpaced, so its 49.85 active-file FPS is file-processing throughput and cannot be compared directly with 25 FPS real-time arrivals. The two-stream active intervals begin at each stream's first queue admission and end at its last output, while the combined startup FPS includes process launch, engine loading, and completion. Stream 0's P95 includes a transient queue delay: queue wait P95 was 52.55 ms (stream 1: 14.37 ms). Decode P50 was 9.41 ms in the single run and about 10 ms per stream in the dual run. These local files are generated from static images, so the test does not establish natural-video accuracy or live camera latency.

GPU timings use CUDA events on the inference stream. The first frame can contain CUDA/TensorRT warmup; it remains in the percentiles. CPU process memory was polled about every 50 ms. GPU memory is `nvidia-smi` **total device memory used** sampled about every 500 ms; the baseline-to-peak increase can include other processes and can miss a transient peak. The frame-count and CSV correspondence tests passed after adding timing columns: single 50=50=50 and dual 50 processed per stream with no drops. The 200 FPS overload test also passed after instrumentation: stream 0 processed 54/dropped 46, stream 1 processed 47/dropped 53, and both queue maxima were 3. These counts depend on scheduling and are not throughput claims. At this milestone, a before/after optimization comparison was still pending; it is recorded below.

## 60-second two-stream stability check and moving-video input

Date: 2026-10-02. `tests/loop_video.py` repeated the prior 50-frame 1280×720 synthetic clip to create two distinct 1,500-frame MP4 files at 25 FPS. With queue capacity 4, the instrumented executable completed two runs. The second run used the now-copied OpenCV FFmpeg plugin and sampled both process working set and Windows private bytes; its ignored files are `outputs/sustain_repeat*.csv`, `outputs/sustain_repeat_report.json`, and `outputs/sustain_repeat_resources.csv`.

| Second run | Decoded / processed / dropped | Active FPS | Queue-admission→file-write P50 / P95 | Pure GPU inference P50 / P95 | Max queue depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stream 0 | 1500 / 1500 / 0 | 25.01 | 28.43 / 35.33 ms | 5.00 / 5.84 ms | 3 |
| Stream 1 | 1500 / 1500 / 0 | 25.02 | 17.68 / 34.68 ms | 5.02 / 5.83 ms | 3 |

The second process ran for 60.43 seconds including startup, with 49.64 combined processed FPS over that full interval. The frame checker decoded both outputs and confirmed all 3,000 source frames were represented in the videos and CSVs. Queue-wait P95 in the first versus last 10 seconds was 20.58→19.58 ms for stream 0 and 19.57→17.98 ms for stream 1, so no increasing wait was observed in this interval. Private bytes median was 783.22 MiB at 10–20 seconds and 784.31 MiB in a 10-second window ending 5 seconds before process exit, a +1.09 MiB difference. Sampled peak private bytes was 794.24 MiB; sampled peak working set was 568.73 MiB. System-wide GPU memory rose by approximately 167 MiB from the baseline; this is not process-specific VRAM. The first 60-second run also completed 1500/1500 frames per stream with no drops, but it used a different OpenCV video backend before the FFmpeg plugin was copied, so its timing numbers are not directly comparable.

**Limit found at this milestone:** the then-current `StreamRuntime.records` stored a record for every decoded frame and wrote the CSV only when each stream finished. This was O(total frames) memory even though the frame queues were bounded. The following milestone removes that accumulation.

## Incremental per-frame CSV writing

Date: 2026-10-02. The prior `StreamRuntime.records` vector was removed. Each producer writes a row when it drops an old queued frame; the single consumer writes a row after processing and video output. A per-stream mutex serializes CSV writes. The in-memory frame records are now limited to queued frames, the frame being processed, and producer-local references. CSV rows are in **completion order** and can be out of source-frame order. The checker was updated to validate the complete set of source frame IDs and the output frame sequence independently.

The D-drive Release build succeeded. The normal two-stream 50-frame check passed with 0 drops. At 200 FPS and queue capacity 3, stream 0 processed 14/dropped 86 and stream 1 processed 13/dropped 87; both maximum queue depths were 3. All 200 input frames were accounted for, and the checker verified output counts and empty output fields on dropped rows. Nine source-order inversions occurred in stream 0's CSV, confirming the checker exercised completion-order rows. Drop counts vary with scheduling and video backend.

A new 60-second run with two 1,500-frame, 25 FPS inputs and queue capacity 4 completed 1,500 processed / 0 dropped per stream. Active processed rate was 25.01 and 25.02 FPS; queue-admission-to-file-write latency P50/P95 was 16.68/33.11 ms for stream 0 and 27.45/34.16 ms for stream 1. The output checker decoded both videos and verified all 3,000 frames and CSV rows. The CSV files had already reached about 124 KB each while the process was still running, confirming records were emitted before completion. `outputs/streaming_sustain_report.json`, `outputs/streaming_sustain_resources.csv`, per-frame CSVs, and videos are retained locally under the ignored `outputs/` directory.

For this run, process private bytes had a median of 785.35 MiB at 10–20 seconds and 785.54 MiB in the late 10-second window, a +0.19 MiB difference; sampled peak was 795.92 MiB. This is a one-run observation, not proof against every memory leak. The earlier 60-second run used the same FFmpeg-equipped build configuration but the old deferred CSV writer; its private-byte median changed by +1.09 MiB over the corresponding windows. The runs show no material latency or memory regression from incremental CSV writing, but one run per version is insufficient for a precise speed comparison.

For real motion decoding, downloaded OpenCV's [`vtest.avi`](https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi) and kept it outside Git. File size: 8,131,690 bytes; SHA256: `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`. OpenCV reports 795 frames at 10 FPS, 768×576. After CMake copied `opencv_videoio_ffmpeg4130_64.dll` next to the executable, the C++ single-video run without special environment variables produced 795 output frames and 795 CSV rows; `tests/check_video_output.py` passed. This tests moving-video I/O only. The source is pedestrian footage, outside the maritime infrared model domain, so its detection counts are not used as an accuracy claim.

## Frame handoff: copy versus move

Date: 2026-10-02. The two 1,500-frame, 1280×720, 25 FPS inputs above were run once each with the same FP16 engine, Release build, queue capacity 4, output codec, and host. The baseline copied every decoded `cv::Mat` with `frame.clone()` before enqueueing. The changed version transfers ownership with `std::move(frame)`. Both runs emitted the same `frame_handoff_ms` timing field. Full commands and per-frame data are in ignored `outputs/experiments/frame_handoff/` as `clone_*` and `move_*` files. No warmup frames were excluded.

| Two-stream aggregate | `clone()` | `std::move` |
| --- | ---: | ---: |
| Decode P50 / P95 | 2.356 / 3.280 ms | 3.119 / 4.246 ms |
| Frame handoff P50 / P95 | 1.313 / 1.908 ms | 0.000 / 0.001 ms (CSV precision) |
| Decode + handoff P50 / P95, computed per frame | 3.683 / 4.898 ms | 3.119 / 4.247 ms |
| Queue admission to file write P50 / P95 | 23.225 / 38.952 ms | 20.099 / 36.268 ms |
| Combined FPS including startup | 49.641 | 49.691 |
| Sampled peak process private bytes | 796.91 MiB | 792.57 MiB |

Moving removes the measured 1.3 ms median pixel copy, while median decode time rises by about 0.8 ms because OpenCV refills a moved-from Mat. The net decode-plus-handoff change in this run is about 0.56 ms per frame. Combined throughput is effectively unchanged because input is paced to 50 FPS total. The observed latency and memory differences come from one run per version and can reflect scheduling or background load; they are not a reliable speedup claim. TensorRT GPU inference remains about 5–6 ms per frame and video writing about 5 ms per frame in these runs.

Both 60-second runs processed 1,500/1,500 frames per stream with no drops, and `tests/check_two_stream_output.py` passed. For every frame ID, processed status and detection/drawn counts matched between versions. Decoded annotated output frames 0, 100, 750, and 1499 were pixel-identical in both streams. A 200 FPS overload check of the move version also passed: each stream decoded 100 frames, processed 14, dropped 86, and stayed within queue capacity 3. These checks support retaining the move change as a small, correctness-preserving reduction in producer work, without claiming a system-level throughput gain.

## Sustained two-stream overload

Date: 2026-10-02. To test the queue policy beyond a short burst, `tests/loop_video.py` was given an optional `--fps` argument. It repeated the existing synthetic 1280×720 clip into two distinct 1,200-frame files at 40 FPS (30 seconds each). The inputs contain repeated validation images, not natural motion; this experiment evaluates pipeline behavior, not detector accuracy. The optimized Release executable used the same FP16 engine, one consumer, and queue capacity 3. Exact command, per-frame CSVs, resource samples, and outputs are in ignored `outputs/experiments/sustained_overload/`.

| Stream | Decoded | Processed | Dropped oldest | Effective processed FPS, active interval | Queue admission→file write P50 / P95 | Queue wait P50 / P95 | Max queue depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1200 | 1056 | 144 | 35.153 | 74.736 / 88.798 ms | 61.646 / 74.935 ms | 3 |
| 1 | 1200 | 1057 | 143 | 35.167 | 73.613 / 88.868 ms | 60.303 / 74.762 ms | 3 |

Combined process time was 30.455 seconds including startup; 2,113 frames were processed, 287 dropped, and combined processed rate including startup was 69.382 FPS. `tests/check_two_stream_output.py --require-drops` decoded both inputs and outputs and verified all 2,400 source frames have exactly one final CSV status, output video frame counts equal processed counts, output IDs remain sequential, dropped rows have no output fields, and no recorded queue depth exceeds 3.

To check for increasing delay, rows were grouped by each stream's `capture_unix_ms` into the first and last five seconds. Stream 0 queue-wait P95 was 74.080→75.385 ms and queue-admission-to-write P95 was 87.851→88.782 ms; stream 1 was 74.276→73.828 ms and 87.244→87.877 ms. No steadily growing queue delay was observed during these 30 seconds; drop counts varied by window. Process private-byte median was 796.65 MiB at 10–20 seconds and 795.95 MiB in the late window (−0.70 MiB). Peak private bytes sampled was 806.01 MiB. Total GPU memory used rose by about 250 MiB from the pre-run baseline, but this figure is device-wide, not process-specific.

The reported latency starts **after file decode and FPS pacing**, at queue admission, and ends after writing the output video frame. It is not camera capture-to-display latency. This single run demonstrates bounded queue behavior at 80 FPS aggregate arrivals under these conditions; it does not establish a universal latency bound or natural-video accuracy.

## Public moving-video two-stream check

Date: 2026-10-02. Both producers read OpenCV's public `vtest.avi` pedestrian clip from byte-identical local copies kept outside Git. The source is 768×576, 795 frames at 10 FPS (79.5 seconds of media); its SHA256 and original source link are recorded above. The two inputs contain the **same scene**, so this checks two-reader pipeline behavior with real motion rather than independence across scenes or maritime detection accuracy. The optimized Release executable used the same local FP16 engine and queue capacity 4. Full commands, per-frame CSVs, resource samples, and output MP4s are in ignored `outputs/experiments/public_motion/`.

| Stream | Decoded / processed / dropped | Active processed FPS | Queue admission→file write P50 / P95 | Pure GPU inference P50 / P95 | Max queue depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 795 / 795 / 0 | 10.010 | 16.174 / 32.542 ms | 8.545 / 16.607 ms | 1 |
| 1 | 795 / 795 / 0 | 10.008 | 25.083 / 40.860 ms | 8.548 / 15.093 ms | 1 |

Process wall time including startup was 79.808 seconds, with 1,590 processed frames and no drops. `tests/check_two_stream_output.py` decoded both outputs and verified that all source frames have a corresponding output frame and CSV record. Sampled peak process private bytes was 666.19 MiB; the median changed from 652.56 MiB at 10–20 seconds to 655.27 MiB in the late window (+2.71 MiB). This single observation does not prove indefinite memory stability. The clip's 10 FPS arrival rate is below the tested 25 FPS target, and its pedestrian content is outside the maritime infrared model's domain. Detection counts from this run are not used as an accuracy result.

## Explicit-dependency clean build check

Date: 2026-10-02. Removed the machine-specific TensorRT fallback from CMake. The Windows build script now accepts `-TensorRTRoot`, optional `-OpenCVDir`, and `-BuildDir`; CMake copies the OpenCV runtime DLL matching the detected OpenCV version. A clean MSVC Release build was configured with TensorRT 10.11.0.33, OpenCV 4.13.0, and CUDA 12.4 in ignored `outputs/build-portability-clean/`. The new executable, `nvinfer_10.dll`, `opencv_world4130.dll`, and `opencv_videoio_ffmpeg4130_64.dll` were present together in its Release directory. The executable processed all 795 frames of public `vtest.avi`; `tests/check_video_output.py` confirmed 795 input frames = 795 output frames = 795 CSV rows at 10 FPS.

The Codex shell in this session had duplicate `PATH`/`Path` environment entries that disrupted a direct clean CMake configure. The successful build invoked the same `scripts/build-windows.ps1` with a cleaned subprocess environment. This is a local shell limitation; the repository still needs a model distribution or retrieval path before an outside reader can reproduce inference results.
