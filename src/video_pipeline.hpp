#pragma once

#include "trt_engine.hpp"

#include <filesystem>
#include <cstddef>

struct VideoOptions {
    std::filesystem::path input;
    std::filesystem::path output_video;
    std::filesystem::path frames_csv;
    float display_threshold = 0.25f;
};

struct VideoSummary {
    size_t frames = 0;
    size_t total_detections = 0;
    size_t drawn_detections = 0;
    double source_fps = 0.0;
};

// Sequential file processing: every decoded input frame writes one output frame.
VideoSummary process_video(const VideoOptions& options, TrtEngine& engine);
