# Model contract

This contract records the interface used in local parity tests against the private Python reference. The reference script and research model are not distributed with this repository. The algorithm and deployment work use one model lineage.

| Item | Current reference |
| --- | --- |
| Model | Private five-class detection model |
| ONNX input | `images`, float32, `[1, 3, 640, 640]` |
| ONNX output | `output0`, float32, `[1, 9, 33600]` |
| Preferred local engine | FP16 on RTX 4060 Laptop GPU |
| Class order | Five class IDs, in the same order as the private Python reference |
| Input image | OpenCV BGR |
| Resize | Letterbox to 640×640, keep aspect ratio, `scaleup=False`, pad BGR `(114,114,114)` |
| Tensor input | Convert BGR to RGB, HWC to CHW, float32 divided by 255, batch dimension 1 |
| Output | 4 decoded `xywh` coordinates and 5 class scores per candidate; do not repeat DFL decode |
| Postprocess | Multi-label score threshold 0.001, class-aware NMS IoU 0.7, maximum 300 outputs; undo letterbox and clip to original image |

For equal FP16 confidence scores, the C++ implementation preserves original candidate order during sorting. The older Python validation script uses NumPy's default `argsort`, whose equal-score order can differ. `tests/compare_detections.py` uses stable `argsort` for a deterministic cross-language comparison without changing the original research script.

The exact Python reference uses `round()` for resized dimensions and split padding with `round(dh ± 0.1)` / `round(dw ± 0.1)`. C++ parity must check these details with images of several aspect ratios. The Python validation threshold is tuned for mAP evaluation; a separate display threshold may be chosen later, but it must be recorded and kept identical in comparisons.

The existing Python benchmark includes modes with incomplete postprocessing. Use `validate_trt_strict.py` as the output comparison reference, and label pure GPU and full pipeline timings separately.
