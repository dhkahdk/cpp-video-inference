#include "trt_engine.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <chrono>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr size_t kInputCount = 1ULL * 3 * 640 * 640;
constexpr size_t kOutputCount = 1ULL * 9 * 33600;

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

class DeviceBuffer {
public:
    ~DeviceBuffer() { if (pointer_) cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer() = default;
    void allocate(size_t bytes) { check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc"); }
    void* get() const { return pointer_; }

private:
    void* pointer_ = nullptr;
};

class CudaStream {
public:
    CudaStream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    ~CudaStream() { if (stream_) cudaStreamDestroy(stream_); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~CudaEvent() { if (event_) cudaEventDestroy(event_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};

bool is_shape(nvinfer1::Dims shape, int d0, int d1, int d2) {
    return shape.nbDims == 3 && shape.d[0] == d0 && shape.d[1] == d1 && shape.d[2] == d2;
}
}  // namespace

struct TrtEngine::Impl {
    explicit Impl(const std::filesystem::path& path) {
        std::ifstream engine_file(path, std::ios::binary);
        if (!engine_file) throw std::runtime_error("Cannot open TensorRT engine");
        const std::vector<char> bytes((std::istreambuf_iterator<char>(engine_file)),
                                      std::istreambuf_iterator<char>());
        if (bytes.empty()) throw std::runtime_error("TensorRT engine is empty");

        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("createInferRuntime failed");
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("createExecutionContext failed");

        if (engine->getNbIOTensors() != 2) throw std::runtime_error("Expected exactly two I/O tensors");
        for (int index = 0; index < engine->getNbIOTensors(); ++index) {
            const char* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name = name;
            else if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) output_name = name;
        }
        if (input_name.empty() || output_name.empty()) throw std::runtime_error("Missing input or output tensor");
        if (engine->getTensorDataType(input_name.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType(output_name.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Expected float32 input and output tensors");
        }
        const auto input_shape = engine->getTensorShape(input_name.c_str());
        if (input_shape.nbDims == 4 && input_shape.d[0] == -1) {
            if (!context->setInputShape(input_name.c_str(), nvinfer1::Dims4{1, 3, 640, 640})) {
                throw std::runtime_error("setInputShape failed");
            }
        }
        const auto actual_input = context->getTensorShape(input_name.c_str());
        const auto actual_output = context->getTensorShape(output_name.c_str());
        if (actual_input.nbDims != 4 || actual_input.d[0] != 1 || actual_input.d[1] != 3 ||
            actual_input.d[2] != 640 || actual_input.d[3] != 640 ||
            !is_shape(actual_output, 1, 9, 33600)) {
            throw std::runtime_error("Engine shape differs from expected [1,3,640,640] -> [1,9,33600]");
        }

        device_input.allocate(kInputCount * sizeof(float));
        device_output.allocate(kOutputCount * sizeof(float));
        if (!context->setTensorAddress(input_name.c_str(), device_input.get()) ||
            !context->setTensorAddress(output_name.c_str(), device_output.get())) {
            throw std::runtime_error("setTensorAddress failed");
        }
    }

    std::vector<float> infer(const std::vector<float>& input, Timings* timings) {
        if (input.size() != kInputCount) throw std::runtime_error("Input tensor has wrong size");
        const auto wall_start = std::chrono::steady_clock::now();
        std::vector<float> output(kOutputCount);
        if (timings) check_cuda(cudaEventRecord(start.get(), stream.get()), "cudaEventRecord start");
        check_cuda(cudaMemcpyAsync(device_input.get(), input.data(), kInputCount * sizeof(float),
                                   cudaMemcpyHostToDevice, stream.get()), "cudaMemcpyAsync H2D");
        if (timings) check_cuda(cudaEventRecord(after_h2d.get(), stream.get()), "cudaEventRecord H2D");
        if (!context->enqueueV3(stream.get())) throw std::runtime_error("enqueueV3 failed");
        if (timings) check_cuda(cudaEventRecord(after_inference.get(), stream.get()), "cudaEventRecord inference");
        check_cuda(cudaMemcpyAsync(output.data(), device_output.get(), kOutputCount * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream.get()), "cudaMemcpyAsync D2H");
        if (timings) check_cuda(cudaEventRecord(after_d2h.get(), stream.get()), "cudaEventRecord D2H");
        check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize");
        if (timings) {
            float milliseconds = 0.0f;
            check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), after_h2d.get()), "H2D event time");
            timings->h2d_gpu_ms = milliseconds;
            check_cuda(cudaEventElapsedTime(&milliseconds, after_h2d.get(), after_inference.get()), "inference event time");
            timings->inference_gpu_ms = milliseconds;
            check_cuda(cudaEventElapsedTime(&milliseconds, after_inference.get(), after_d2h.get()), "D2H event time");
            timings->d2h_gpu_ms = milliseconds;
            timings->infer_wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wall_start).count();
        }
        return output;
    }

    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::string input_name;
    std::string output_name;
    DeviceBuffer device_input;
    DeviceBuffer device_output;
    CudaStream stream;
    CudaEvent start;
    CudaEvent after_h2d;
    CudaEvent after_inference;
    CudaEvent after_d2h;
};

TrtEngine::TrtEngine(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}
TrtEngine::~TrtEngine() = default;
std::vector<float> TrtEngine::infer(const std::vector<float>& input, Timings* timings) {
    return impl_->infer(input, timings);
}
