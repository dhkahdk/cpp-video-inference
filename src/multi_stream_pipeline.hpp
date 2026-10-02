#pragma once

#include "video_pipeline.hpp"

#include <array>
#include <cstddef>
#include <filesystem>

struct MultiStreamOptions {
    std::array<VideoOptions, 2> streams;
    size_t queue_capacity = 4;
};

struct StreamSummary {
    size_t decoded = 0;
    size_t processed = 0;
    size_t dropped = 0;
    size_t max_queue_depth = 0;
    double source_fps = 0.0;
};

// Two paced file readers feed one TensorRT consumer. Each stream has an
// independent bounded queue and drops its oldest queued frame when full.
std::array<StreamSummary, 2> process_two_streams(const MultiStreamOptions& options,
                                                 TrtEngine& engine);
