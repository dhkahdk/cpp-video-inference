#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {
constexpr int kImageSize = 640;
}

PreprocessedFrame preprocess_image(const cv::Mat& source) {
    if (source.empty() || source.type() != CV_8UC3) {
        throw std::invalid_argument("Expected a nonempty 8-bit three-channel BGR image");
    }
    const double ratio = std::min({static_cast<double>(kImageSize) / source.rows,
                                   static_cast<double>(kImageSize) / source.cols, 1.0});
    const int resized_width = static_cast<int>(std::nearbyint(source.cols * ratio));
    const int resized_height = static_cast<int>(std::nearbyint(source.rows * ratio));
    const double pad_w = (kImageSize - resized_width) / 2.0;
    const double pad_h = (kImageSize - resized_height) / 2.0;

    cv::Mat resized;
    if (resized_width != source.cols || resized_height != source.rows) {
        cv::resize(source, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = source;
    }
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       static_cast<int>(std::nearbyint(pad_h - 0.1)),
                       static_cast<int>(std::nearbyint(pad_h + 0.1)),
                       static_cast<int>(std::nearbyint(pad_w - 0.1)),
                       static_cast<int>(std::nearbyint(pad_w + 0.1)),
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    if (padded.rows != kImageSize || padded.cols != kImageSize) {
        throw std::runtime_error("Letterbox did not produce a 640x640 image");
    }

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    std::vector<float> tensor(3 * kImageSize * kImageSize);
    const size_t plane_size = static_cast<size_t>(kImageSize) * kImageSize;
    for (int y = 0; y < kImageSize; ++y) {
        const auto* row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kImageSize; ++x) {
            const size_t offset = static_cast<size_t>(y) * kImageSize + x;
            for (int channel = 0; channel < 3; ++channel) {
                tensor[channel * plane_size + offset] =
                    static_cast<float>(row[x][channel]) / 255.0f;
            }
        }
    }
    return {std::move(tensor), ratio, pad_w, pad_h};
}
