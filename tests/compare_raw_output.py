"""Compare one C++ TensorRT call with the existing Python TensorRT runtime."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def python_infer(engine_path: Path, input_tensor: np.ndarray, trt_lib: Path) -> np.ndarray:
    # Python TensorRT 10 on Windows searches PATH at import time.
    os.environ["PATH"] = str(trt_lib) + os.pathsep + os.environ["PATH"]
    dll_directory = os.add_dll_directory(str(trt_lib)) if hasattr(os, "add_dll_directory") else None
    import tensorrt as trt
    from cuda import cudart

    def checked(result, operation):
        if result[0] != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation}: {result[0]}")
        return result[1] if len(result) > 1 else None

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Python could not deserialize engine")
    context = engine.create_execution_context()
    output = np.empty((1, 9, 33600), dtype=np.float32)
    device_input = checked(cudart.cudaMalloc(input_tensor.nbytes), "cudaMalloc input")
    device_output = checked(cudart.cudaMalloc(output.nbytes), "cudaMalloc output")
    try:
        input_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                          if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT)
        output_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                           if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT)
        if -1 in tuple(engine.get_tensor_shape(input_name)):
            assert context.set_input_shape(input_name, (1, 3, 640, 640))
        assert context.set_tensor_address(input_name, int(device_input))
        assert context.set_tensor_address(output_name, int(device_output))
        checked(cudart.cudaMemcpy(device_input, input_tensor.ctypes.data, input_tensor.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice), "copy input")
        if not context.execute_async_v3(0):
            raise RuntimeError("Python execute_async_v3 failed")
        checked(cudart.cudaMemcpy(output.ctypes.data, device_output, output.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost), "copy output")
        return output
    finally:
        checked(cudart.cudaFree(device_input), "cudaFree input")
        checked(cudart.cudaFree(device_output), "cudaFree output")
        if dll_directory is not None:
            dll_directory.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--trt-lib", type=Path, required=True)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "input.bin"
        output_path = Path(directory) / "output.bin"
        for image in args.images:
            subprocess.run(
                [str(args.exe), "--input", str(image), "--engine", str(args.engine),
                 "--dump-tensor", str(input_path), "--dump-output", str(output_path)],
                check=True, capture_output=True, text=True,
            )
            input_tensor = np.fromfile(input_path, dtype=np.float32).reshape(1, 3, 640, 640)
            actual = np.fromfile(output_path, dtype=np.float32).reshape(1, 9, 33600)
            expected = python_infer(args.engine, input_tensor, args.trt_lib)
            difference = np.abs(actual - expected)
            print(f"{image.name}: max_abs={difference.max():.9g}, mismatched>1e-4={np.count_nonzero(difference > 1e-4)}")
            if not np.allclose(actual, expected, atol=1e-4, rtol=1e-4):
                raise SystemExit(1)


if __name__ == "__main__":
    main()
