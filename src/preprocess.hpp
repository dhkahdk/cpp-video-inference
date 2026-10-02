#pragma once

#include <opencv2/core.hpp>

#include <vector>

struct PreprocessedFrame {
    std::vector<float> tensor;
    double ratio;
    double pad_w;
    double pad_h;
};

// BGR image -> RGB float32 NCHW input matching validate_trt_strict.py.
PreprocessedFrame preprocess_image(const cv::Mat& bgr);
