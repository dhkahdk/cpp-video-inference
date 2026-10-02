#include "video_pipeline.hpp"

#include "postprocess.hpp"
#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using SteadyClock = std::chrono::steady_clock;
double elapsed_ms(SteadyClock::time_point start, SteadyClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
int64_t wall_time_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

size_t draw_detections(cv::Mat& frame, const std::vector<Detection>& detections,
                       float threshold) {
    static const std::array<const char*, 5> names = {
        "class_0", "class_1", "class_2", "class_3", "class_4"};
    size_t drawn = 0;
    std::vector<cv::Rect> used_labels;
    for (const auto& detection : detections) {
        if (detection.score < threshold) continue;
        const cv::Point top_left(static_cast<int>(std::round(detection.x1)),
                                 static_cast<int>(std::round(detection.y1)));
        const cv::Point bottom_right(static_cast<int>(std::round(detection.x2)),
                                     static_cast<int>(std::round(detection.y2)));
        const cv::Scalar color(0, 255, 255);
        cv::rectangle(frame, top_left, bottom_right, color, 2);
        const std::string label = std::string(names.at(detection.class_id)) + " " +
                                  cv::format("%.2f", detection.score);
        const double font_scale = 0.5;
        const int text_thickness = 1;
        int baseline = 0;
        const cv::Size text = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                               font_scale, text_thickness, &baseline);
        const int label_x = std::clamp(top_left.x, 0, std::max(0, frame.cols - text.width - 6));
        const int label_height = text.height + baseline + 7;
        const int label_width = text.width + 6;
        cv::Rect label_rect;
        bool placed = false;
        for (int layer = 0; layer < 12 && !placed; ++layer) {
            for (int direction = 0; direction < 2 && !placed; ++direction) {
                const int y = direction == 0
                    ? top_left.y - (layer + 1) * label_height
                    : bottom_right.y + layer * label_height + 2;
                cv::Rect candidate(label_x, y, label_width, label_height);
                if (candidate.y < 0 || candidate.br().y > frame.rows) continue;
                const bool overlaps = std::any_of(used_labels.begin(), used_labels.end(),
                    [&](const cv::Rect& used) { return (candidate & used).area() > 0; });
                if (!overlaps) {
                    label_rect = candidate;
                    placed = true;
                }
            }
        }
        if (!placed) {
            label_rect = cv::Rect(label_x,
                std::clamp(top_left.y - label_height, 0, std::max(0, frame.rows - label_height)),
                label_width, label_height);
        }
        used_labels.push_back(label_rect);
        cv::rectangle(frame, label_rect, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, label,
                    cv::Point(label_rect.x + 3, label_rect.y + text.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness);
        ++drawn;
    }
    return drawn;
}
}

VideoSummary process_video(const VideoOptions& options, TrtEngine& engine) {
    if (options.display_threshold < 0.0f || options.display_threshold > 1.0f ||
        options.output_video.empty() || options.frames_csv.empty()) {
        throw std::invalid_argument("Invalid video output paths or display threshold");
    }
    cv::VideoCapture capture(options.input.string());
    if (!capture.isOpened()) throw std::runtime_error("Could not open input video");
    const double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        throw std::runtime_error("Input video has no usable FPS metadata");
    }
    cv::Mat frame;
    auto read_start = SteadyClock::now();
    if (!capture.read(frame) || frame.empty()) throw std::runtime_error("Input video has no frames");
    auto read_end = SteadyClock::now();

    const auto video_parent = options.output_video.parent_path();
    const auto csv_parent = options.frames_csv.parent_path();
    if (!video_parent.empty()) std::filesystem::create_directories(video_parent);
    if (!csv_parent.empty()) std::filesystem::create_directories(csv_parent);
    cv::VideoWriter writer(options.output_video.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, frame.size(), true);
    if (!writer.isOpened()) throw std::runtime_error("Could not open MP4 output writer");
    std::ofstream csv(options.frames_csv);
    if (!csv) throw std::runtime_error("Could not open per-frame CSV");
    csv << "frame_id,media_pts_ms,read_wall_unix_ms,write_wall_unix_ms,"
           "detections,drawn_detections,decode_ms,preprocess_ms,h2d_gpu_ms,"
           "inference_gpu_ms,d2h_gpu_ms,infer_wall_ms,postprocess_ms,"
           "draw_ms,write_ms,file_read_to_output_ms\n";

    VideoSummary summary;
    summary.source_fps = fps;
    do {
        const int64_t read_wall = wall_time_ms();
        const double decode_ms = elapsed_ms(read_start, read_end);
        double media_pts = capture.get(cv::CAP_PROP_POS_MSEC);
        if (!std::isfinite(media_pts) || media_pts < 0.0) {
            media_pts = static_cast<double>(summary.frames) * 1000.0 / fps;
        }
        const auto preprocess_start = SteadyClock::now();
        const auto preprocessed = preprocess_image(frame);
        const auto preprocess_end = SteadyClock::now();
        TrtEngine::Timings timings;
        const auto raw = engine.infer(preprocessed.tensor, &timings);
        const auto postprocess_start = SteadyClock::now();
        const auto detections = decode_detections(raw, frame.cols, frame.rows,
                                                  preprocessed.ratio,
                                                  preprocessed.pad_w, preprocessed.pad_h);
        const auto postprocess_end = SteadyClock::now();
        const size_t drawn = draw_detections(frame, detections, options.display_threshold);
        const auto draw_end = SteadyClock::now();
        writer.write(frame);
        const auto write_end = SteadyClock::now();
        const int64_t write_wall = wall_time_ms();
        csv << summary.frames << ',' << std::fixed << std::setprecision(3) << media_pts << ','
            << read_wall << ',' << write_wall << ',' << detections.size() << ',' << drawn << ','
            << decode_ms << ',' << elapsed_ms(preprocess_start, preprocess_end) << ','
            << timings.h2d_gpu_ms << ',' << timings.inference_gpu_ms << ','
            << timings.d2h_gpu_ms << ',' << timings.infer_wall_ms << ','
            << elapsed_ms(postprocess_start, postprocess_end) << ','
            << elapsed_ms(postprocess_end, draw_end) << ','
            << elapsed_ms(draw_end, write_end) << ','
            << elapsed_ms(read_start, write_end) << '\n';
        if (!csv) throw std::runtime_error("Could not write per-frame CSV");
        ++summary.frames;
        summary.total_detections += detections.size();
        summary.drawn_detections += drawn;
        read_start = SteadyClock::now();
        if (!capture.read(frame) || frame.empty()) break;
        read_end = SteadyClock::now();
    } while (true);
    writer.release();
    capture.release();
    csv.close();
    return summary;
}
