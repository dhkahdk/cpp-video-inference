#pragma once

#include <vector>

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

// Matches the validation settings in validate_trt_strict.py.
std::vector<Detection> decode_detections(const std::vector<float>& output,
                                         int original_width, int original_height,
                                         double ratio, double pad_w, double pad_h);
