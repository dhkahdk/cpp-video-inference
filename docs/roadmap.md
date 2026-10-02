# Milestones

1. **Local environment — complete**: installed the C++ toolchain and local dependencies and verified the Release build on the RTX 4060 Laptop GPU. Public-machine setup instructions still need cleanup.
2. **Single-frame parity — complete**: implemented engine loading, matching preprocessing, inference, decoded output parsing, and class-aware NMS; compared fixed images with Python.
3. **Single video — complete**: local MP4 input writes an annotated MP4 and one CSV record per decoded frame.
4. **Two streams and congestion — complete**: two FPS-paced producers use independent bounded queues and drop oldest on overflow. A 30-second two-stream 40 FPS overload run accounted for every frame and kept queue depth within capacity 3.
5. **Measurement and improvement — complete for local experiments**: recorded stage timings, effective FPS, P50/P95 latency, drops, and sampled process/device memory. Completed a 60-second two-stream run and compared decoded-frame copy with ownership transfer on the same inputs. See [validation.md](validation.md) for numbers and measurement limits.
6. **GitHub reproducibility — pending**: keep the unpublished research ONNX model, engine, and validation images private; review tracked content before the first commit. Public readers can inspect and build the pipeline, but cannot reproduce the recorded inference runs until an authorized compatible model and test inputs are available. A separate public-model demonstration remains possible after interface adaptation and validation.

The local tests met the original 25 FPS single-stream target and demonstrated bounded queues under two-stream overload. These results apply to the stated PC GPU, model, and synthetic video conditions; they are not an edge-device or live-camera result.
