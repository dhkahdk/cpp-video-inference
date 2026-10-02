#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
constexpr int kCandidates = 33600;
constexpr int kClasses = 5;
constexpr float kConfidenceThreshold = 0.001f;
constexpr float kIouThreshold = 0.7f;
constexpr size_t kMaxNms = 30000;
constexpr size_t kMaxDetections = 300;

float overlap(const Detection& a, const Detection& b) {
    const float width = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float height = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float intersection = width * height;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    return intersection / (area_a + area_b - intersection + 1e-7f);
}

bool higher_score(const Detection& a, const Detection& b) {
    return a.score > b.score;
}
}  // namespace

std::vector<Detection> decode_detections(const std::vector<float>& output,
                                         int original_width, int original_height,
                                         double ratio, double pad_w, double pad_h) {
    if (output.size() != static_cast<size_t>(9 * kCandidates) ||
        original_width <= 0 || original_height <= 0 || ratio <= 0.0) {
        throw std::invalid_argument("Invalid detection output or image geometry");
    }
    std::vector<Detection> selected;
    const float scale = static_cast<float>(ratio);
    const float offset_x = static_cast<float>(pad_w);
    const float offset_y = static_cast<float>(pad_h);
    for (int index = 0; index < kCandidates; ++index) {
        const float x = output[index];
        const float y = output[kCandidates + index];
        const float width = output[2 * kCandidates + index];
        const float height = output[3 * kCandidates + index];
        for (int class_id = 0; class_id < kClasses; ++class_id) {
            const float score = output[(4 + class_id) * kCandidates + index];
            if (!(score > kConfidenceThreshold)) continue;
            Detection detection{
                (x - width / 2.0f - offset_x) / scale,
                (y - height / 2.0f - offset_y) / scale,
                (x + width / 2.0f - offset_x) / scale,
                (y + height / 2.0f - offset_y) / scale,
                score,
                class_id,
            };
            detection.x1 = std::clamp(detection.x1, 0.0f, static_cast<float>(original_width));
            detection.x2 = std::clamp(detection.x2, 0.0f, static_cast<float>(original_width));
            detection.y1 = std::clamp(detection.y1, 0.0f, static_cast<float>(original_height));
            detection.y2 = std::clamp(detection.y2, 0.0f, static_cast<float>(original_height));
            selected.push_back(detection);
        }
    }
    if (selected.size() > kMaxNms) {
        std::partial_sort(selected.begin(), selected.begin() + kMaxNms, selected.end(), higher_score);
        selected.resize(kMaxNms);
    }

    std::vector<Detection> kept;
    for (int class_id = 0; class_id < kClasses; ++class_id) {
        std::vector<Detection> candidates;
        for (const auto& detection : selected) {
            if (detection.class_id == class_id) candidates.push_back(detection);
        }
        // Preserve candidate order for equal FP16 scores. Otherwise a tied,
        // heavily overlapping box can replace its neighbor after NMS.
        std::stable_sort(candidates.begin(), candidates.end(), higher_score);
        for (const auto& candidate : candidates) {
            bool suppressed = false;
            for (const auto& previous : kept) {
                if (previous.class_id == class_id && overlap(candidate, previous) > kIouThreshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) kept.push_back(candidate);
        }
    }
    std::stable_sort(kept.begin(), kept.end(), higher_score);
    if (kept.size() > kMaxDetections) kept.resize(kMaxDetections);
    return kept;
}
