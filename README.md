# C++ Video Inference System

An original C++ video detection pipeline under development. The model comes from the existing UAV algorithm and PC TensorRT validation work; this repository focuses on single-frame parity, continuous video processing, bounded queues, and latency measurement.

**Model availability:** The research ONNX model and TensorRT engine are not public while the related paper is unpublished. Neither is included in this repository. The recorded inference and performance results used that private model and cannot currently be reproduced by an outside reader. Running inference requires a locally available, compatible model and engine; the current executable expects the five-class input/output contract in [docs/model-contract.md](docs/model-contract.md). The build and source can be inspected without model files. A public-model demonstration would require a separately documented model and any necessary interface changes.

## Current status

**Milestone 8: sustained overload check.** The executable handles still images, sequential single-video files, and two local videos paced from their FPS metadata. Two producer threads feed one TensorRT consumer through an independent bounded queue per stream. A full queue drops its oldest waiting frame and records that decision in CSV. Per-frame CSV separates file decoding, frame handoff, queue wait, preprocessing, GPU transfers, pure GPU inference, postprocessing, drawing, and video writing. Each processed or dropped frame is written to CSV when its status is final; the program no longer retains every frame's record until exit. The producer transfers each decoded `cv::Mat` to the queue instead of cloning its pixels. A same-input before/after measurement and a 30-second, two-stream 40 FPS overload run are in [docs/validation.md](docs/validation.md).

## Model interface

See [docs/model-contract.md](docs/model-contract.md). The research ONNX and FP16 engine are local inputs excluded from Git. TensorRT engines depend on hardware and TensorRT version; authorized users should build one for each target environment from their locally available ONNX model.

## Build

On Windows with Visual Studio Build Tools 2022, CUDA 12.4, TensorRT 10, and OpenCV 4, run from the repository root. Pass the TensorRT installation directory explicitly; `-OpenCVDir` is optional when the OpenCV package is in the sibling `toolchains/` folder:

```powershell
.\scripts\build-windows.ps1 -TensorRTRoot "C:\path\to\TensorRT-10" -OpenCVDir "C:\path\to\opencv\build"
```

The script detects CMake in the sibling `toolchains/` folder or on `PATH`. It also accepts `TENSORRT_ROOT` and `OpenCV_DIR` environment variables, plus `-BuildDir` for a separate build folder. CMake finds the local OpenCV package in sibling `toolchains/` when `OpenCV_DIR` is omitted. Equivalent manual commands are:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DTENSORRT_ROOT="C:/path/to/TensorRT-10" -DOpenCV_DIR="C:/path/to/opencv/build"
cmake --build build --config Release
```

The executable uses C++17, OpenCV, TensorRT 10, and the CUDA runtime. The build copies the matching OpenCV runtime DLL and FFmpeg videoio plugin beside the executable when present in the official Windows OpenCV package. Without that plugin, some AVI files may open but report unusable FPS metadata. TensorRT engines must be rebuilt for the target TensorRT version and GPU.

## Run the current preprocessing check

```powershell
New-Item -ItemType Directory outputs -Force | Out-Null
.\build\Release\video_infer.exe --input "C:\path\to\image.jpg" --dump-tensor "outputs\input.bin"
```

To compare its tensor with the Python validation preprocessing, run `python tests/compare_preprocess.py --exe build/Release/video_infer.exe path/to/image.jpg`. This requires Python, NumPy and OpenCV 4.13.

To run one raw inference with the local FP16 engine:

```powershell
.\build\Release\video_infer.exe --input "C:\path\to\image.jpg" --engine "C:\path\to\compatible.engine" --dump-output "outputs\output.bin"
```

Add `--dump-detections "outputs\detections.csv"` to save final `class_id,score,x1,y1,x2,y2` rows in original-image pixels. The raw file is contiguous float32 in `[1,9,33600]` layout and remains available for parity checks.

To compare the C++ output with Python TensorRT using the **same C++ input tensor**, run:

```powershell
python tests/compare_raw_output.py --exe build/Release/video_infer.exe --engine "C:\path\to\compatible.engine" --trt-lib "C:\path\to\TensorRT\lib" "C:\path\to\image.jpg"
```

The three-image tensor checks and the twenty-image detection-box check are recorded in [docs/validation.md](docs/validation.md). Equal FP16 scores are handled with stable order in C++ and in the Python comparison test; the older Python validation script uses NumPy's default tie order, which can select a neighboring box.

For detection-box comparison, run `tests/compare_detections.py` in the existing deployment Python environment and pass `--reference` pointing to its `validate_trt_strict.py`, plus the same `--exe`, `--engine`, `--trt-lib`, and image paths used by the raw-output comparison.

## Single local video

```powershell
.\build\Release\video_infer.exe --video "C:\path\to\input.mp4" --engine "C:\path\to\compatible.engine" --output-video "outputs\annotated.mp4" --frames-csv "outputs\frames.csv" --display-threshold 0.25
```

The CSV records `frame_id`, source media timestamp in milliseconds, wall-clock times after reading and writing, detection counts, and stage timings. Every decoded input frame writes one output frame. The `0.25` threshold controls drawing only; the validation NMS threshold remains `0.001`. Output MP4 uses `mp4v` at the source's reported FPS, so variable-rate input is written as constant-rate output. `file_read_to_output_ms` begins just before local file decoding and ends after video writing; the first frame also includes writer setup. This is local file-processing latency, not camera capture-to-display latency.

For a local smoke test, `tests/make_smoke_video.py` can create a short MP4 from JPEG images; `tests/check_video_output.py` checks decoded source/output/CSV frame counts and IDs. Keep any source images and generated videos in ignored `data/` or `outputs/`. See [docs/validation.md](docs/validation.md) for the 50-frame smoke result.

## Two local video inputs and bounded queues

```powershell
.\build\Release\video_infer.exe `
  --video "C:\path\to\stream1.mp4" --video2 "C:\path\to\stream2.mp4" `
  --engine "C:\path\to\compatible.engine" `
  --output-video "outputs\stream1.mp4" --output-video2 "outputs\stream2.mp4" `
  --frames-csv "outputs\stream1.csv" --frames-csv2 "outputs\stream2.csv" `
  --queue-capacity 4 --display-threshold 0.25
```

Each producer paces delivery according to its input FPS, which approximates two live sources while using reproducible local files. Each stream has its own queue with the requested capacity. When a queue is full, the oldest waiting frame receives status `dropped_oldest`; the new frame is queued. One consumer alternates between nonempty queues and owns the TensorRT context.

The per-stream CSV contains every decoded source frame. Processed rows contain a sequential output-frame ID, queue wait, `capture_to_output_ms`, detection counts, and stage timings; dropped rows keep output fields empty but retain decoding time. Rows are written as frames finish, so **CSV row order can differ from source frame order** under congestion; use `frame_id` and `output_frame_id` for correspondence. `capture_to_output_ms` starts **after local file decode and FPS pacing**, at queue admission, and ends after video writing. It measures simulated live-pipeline latency, not real camera capture latency. The annotated MP4 contains processed frames only. It retains source FPS metadata, so its playback duration becomes shorter when frames are dropped. `tests/check_two_stream_output.py` verifies source = processed + dropped, output frames = processed, complete IDs, empty output fields for dropped rows, and queue depth within capacity.

`frame_handoff_ms` measures the producer operation that gives a decoded `cv::Mat` to the queue. Compare it together with `decode_ms`: moving the frame removes a pixel copy but can make the next decode allocate a fresh buffer. The saved source and annotated video still use the same pixels; the move does not change image preprocessing or inference.

## Short-run benchmark

Run `python scripts/benchmark.py --csv outputs/run.csv --report outputs/run.json --samples-csv outputs/resources.csv -- build/Release/video_infer.exe --video ... --engine ... --output-video ... --frames-csv outputs/run.csv` and supply a second `--csv` for two-stream mode. The script needs `psutil` and, for GPU memory sampling, `nvidia-smi`. It reports P50/P95 from processed frames, decoded/processed/dropped counts, effective FPS during the active file interval, total FPS including model loading and process startup, sampled peak process RSS, Windows private bytes when available, and system-wide GPU memory baseline/peak. The optional resource CSV saves samples over time. The GPU memory delta is only an approximation of this process's allocation because other applications can change VRAM use. GPU transfer and inference times use CUDA events; `infer_wall_ms` includes host enqueue and synchronization. These quantities are not additive because GPU event intervals and CPU wall intervals have different clocks and boundaries.

For a reproducible 60-second synthetic input, use `python tests/loop_video.py --input outputs/normal_stream1.mp4 --output outputs/sustain_input1.mp4 --frames 1500` and make a second copy with a distinct path for `--video2`. The local sample `vtest.avi` was downloaded from [OpenCV's sample repository](https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi) and kept outside this Git repository. It is a 10 FPS pedestrian video used only to test moving-video decoding and frame correspondence; the maritime infrared model's detections on it are not an accuracy evaluation.

For a sustained overload input, `tests/loop_video.py` also accepts `--fps`. The recorded test repeated a local synthetic video to 1,200 frames at 40 FPS per stream (`--frames 1200 --fps 40`) and ran two distinct paths with `--queue-capacity 3`. The data and complete commands are in ignored `outputs/experiments/sustained_overload/`; results are in [docs/validation.md](docs/validation.md). In a congested run, the output MP4 still uses source FPS metadata but contains only processed frames, so use CSV source and output IDs to interpret dropped frames.

A separate two-stream check used the public moving-video `vtest.avi` at its original 10 FPS for roughly 80 seconds. Both streams processed all 795 frames with no drops; see [docs/validation.md](docs/validation.md). The two input paths point to byte-identical copies, and this clip is outside the model's training domain. It checks sustained moving-video I/O and frame correspondence, not detection accuracy or 25 FPS capacity.

Output and test media belong in ignored `outputs/` and `data/` folders. Publish only media and model files for which redistribution is allowed, or document how to obtain them.

## Reproducibility checklist

Before publishing performance results, record the GPU, driver, CUDA, TensorRT, OpenCV, compiler, model hash, video properties, run duration, and measurement method. A working `trtexec` from the matching TensorRT package can rebuild a local FP16 engine from ONNX:

```powershell
trtexec --onnx=C:\path\to\authorized-model.onnx --saveEngine=C:\path\to\compatible.engine --fp16
```

Use the actual `trtexec.exe` path if it is not on `PATH`. This command is for an authorized, compatible ONNX model; it does not provide the private research model. Validate the rebuilt engine against the matching Python reference before using it in the video pipeline.
