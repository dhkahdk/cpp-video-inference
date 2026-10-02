#pragma once

#include <filesystem>
#include <memory>
#include <vector>

class TrtEngine {
public:
    struct Timings {
        double h2d_gpu_ms = 0.0;
        double inference_gpu_ms = 0.0;
        double d2h_gpu_ms = 0.0;
        double infer_wall_ms = 0.0;
    };
    explicit TrtEngine(const std::filesystem::path& path);
    ~TrtEngine();
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Output is contiguous float32 [1,9,33600] in the engine's native layout.
    std::vector<float> infer(const std::vector<float>& input, Timings* timings = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
