#include "postprocess.hpp"
#include "preprocess.hpp"
#include "trt_engine.hpp"
#include "video_pipeline.hpp"
#include "multi_stream_pipeline.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {
void print_usage(const char* program) {
    std::cout << "Image: " << program
              << " --input <image> [--engine <file>] [--dump-tensor <bin>]"
                 " [--dump-output <bin>] [--dump-detections <csv>]\n"
              << "Video: " << program
              << " --video <mp4> --engine <file> --output-video <mp4>"
                 " --frames-csv <csv> [--display-threshold 0.25]\n"
              << "Two streams: add --video2 <mp4> --output-video2 <mp4>"
                 " --frames-csv2 <csv> [--queue-capacity 4]\n";
}

void write_floats(const fs::path& path, const std::vector<float>& data) {
    if (path.empty()) return;
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Could not open binary tensor output");
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!out) throw std::runtime_error("Could not write binary tensor output");
}

void run_image(const fs::path& image_path, const fs::path& engine_path,
               const fs::path& dump_tensor, const fs::path& dump_output,
               const fs::path& dump_detections) {
    const cv::Mat source = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (source.empty()) throw std::runtime_error("OpenCV could not decode the image");
    const auto preprocessed = preprocess_image(source);
    write_floats(dump_tensor, preprocessed.tensor);
    std::cout << "Image: " << source.cols << 'x' << source.rows << '\n'
              << "Letterbox ratio: " << preprocessed.ratio << ", pad: "
              << preprocessed.pad_w << ',' << preprocessed.pad_h << '\n'
              << "Tensor: float32 [1,3,640,640], values=" << preprocessed.tensor.size() << '\n';
    if (engine_path.empty()) {
        std::cout << "TensorRT inference: skipped (no --engine)\n";
        return;
    }
    TrtEngine engine(engine_path);
    const auto output = engine.infer(preprocessed.tensor);
    write_floats(dump_output, output);
    const auto [minimum, maximum] = std::minmax_element(output.begin(), output.end());
    const auto nonfinite = std::count_if(output.begin(), output.end(),
                                          [](float value) { return !std::isfinite(value); });
    std::cout << "Output: float32 [1,9,33600], values=" << output.size()
              << ", min=" << *minimum << ", max=" << *maximum
              << ", nonfinite=" << nonfinite << '\n';
    const auto detections = decode_detections(output, source.cols, source.rows,
                                              preprocessed.ratio, preprocessed.pad_w,
                                              preprocessed.pad_h);
    std::cout << "Detections after class-aware NMS: " << detections.size() << '\n';
    if (!dump_detections.empty()) {
        std::ofstream out(dump_detections);
        if (!out) throw std::runtime_error("Could not open detections CSV");
        out << "class_id,score,x1,y1,x2,y2\n" << std::setprecision(9);
        for (const auto& detection : detections) {
            out << detection.class_id << ',' << detection.score << ','
                << detection.x1 << ',' << detection.y1 << ','
                << detection.x2 << ',' << detection.y2 << '\n';
        }
        if (!out) throw std::runtime_error("Could not write detections CSV");
    }
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        fs::path image_path, video_path, video_path2, engine_path, dump_tensor, dump_output;
        fs::path dump_detections, output_video, output_video2, frames_csv, frames_csv2;
        float display_threshold = 0.25f;
        size_t queue_capacity = 4;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            }
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << arg << '\n';
                return 2;
            }
            const std::string value = argv[++i];
            if (arg == "--input") image_path = value;
            else if (arg == "--video") video_path = value;
            else if (arg == "--video2") video_path2 = value;
            else if (arg == "--engine") engine_path = value;
            else if (arg == "--dump-tensor") dump_tensor = value;
            else if (arg == "--dump-output") dump_output = value;
            else if (arg == "--dump-detections") dump_detections = value;
            else if (arg == "--output-video") output_video = value;
            else if (arg == "--output-video2") output_video2 = value;
            else if (arg == "--frames-csv") frames_csv = value;
            else if (arg == "--frames-csv2") frames_csv2 = value;
            else if (arg == "--display-threshold") display_threshold = std::stof(value);
            else if (arg == "--queue-capacity") queue_capacity = std::stoull(value);
            else {
                std::cerr << "Unknown option: " << arg << '\n';
                print_usage(argv[0]);
                return 2;
            }
        }
        if (image_path.empty() == video_path.empty()) {
            throw std::invalid_argument("Specify exactly one of --input and --video");
        }
        const fs::path source = video_path.empty() ? image_path : video_path;
        if (!fs::is_regular_file(source)) throw std::runtime_error("Input file not found");
        if (!video_path2.empty() && !fs::is_regular_file(video_path2)) {
            throw std::runtime_error("Second video file not found");
        }
        if (!engine_path.empty() && !fs::is_regular_file(engine_path)) {
            throw std::runtime_error("Engine file not found");
        }
        if (!video_path.empty()) {
            if (engine_path.empty()) throw std::invalid_argument("--video requires --engine");
            if (output_video.empty() || frames_csv.empty()) {
                throw std::invalid_argument("--video requires --output-video and --frames-csv");
            }
            if (!dump_tensor.empty() || !dump_output.empty() || !dump_detections.empty()) {
                throw std::invalid_argument("Image-only dump options cannot be used with --video");
            }
            const auto input_full = fs::absolute(video_path).lexically_normal();
            const auto engine_full = fs::absolute(engine_path).lexically_normal();
            const auto output_full = fs::absolute(output_video).lexically_normal();
            const auto csv_full = fs::absolute(frames_csv).lexically_normal();
            if (input_full == output_full || input_full == csv_full || output_full == csv_full ||
                engine_full == output_full || engine_full == csv_full) {
                throw std::invalid_argument("Input, engine, and output paths must differ");
            }
            TrtEngine engine(engine_path);
            if (video_path2.empty()) {
                if (!output_video2.empty() || !frames_csv2.empty()) {
                    throw std::invalid_argument("Second-stream outputs require --video2");
                }
                const auto result = process_video(
                    {video_path, output_video, frames_csv, display_threshold}, engine);
                std::cout << "Processed frames: " << result.frames
                          << ", source FPS: " << result.source_fps
                          << ", detections: " << result.total_detections
                          << ", drawn: " << result.drawn_detections << '\n';
            } else {
                if (output_video2.empty() || frames_csv2.empty()) {
                    throw std::invalid_argument("--video2 requires --output-video2 and --frames-csv2");
                }
                const std::array<fs::path, 7> paths = {
                    video_path, video_path2, engine_path, output_video, output_video2,
                    frames_csv, frames_csv2};
                for (size_t left = 0; left < paths.size(); ++left) {
                    for (size_t right = left + 1; right < paths.size(); ++right) {
                        if (fs::absolute(paths[left]).lexically_normal() ==
                            fs::absolute(paths[right]).lexically_normal()) {
                            throw std::invalid_argument("All two-stream input and output paths must differ");
                        }
                    }
                }
                const auto results = process_two_streams({{{
                    {video_path, output_video, frames_csv, display_threshold},
                    {video_path2, output_video2, frames_csv2, display_threshold}}},
                    queue_capacity}, engine);
                for (int stream = 0; stream < 2; ++stream) {
                    std::cout << "Stream " << stream << ": decoded=" << results[stream].decoded
                              << ", processed=" << results[stream].processed
                              << ", dropped=" << results[stream].dropped
                              << ", max_queue=" << results[stream].max_queue_depth
                              << ", source_fps=" << results[stream].source_fps << '\n';
                }
            }
        } else {
            if (!video_path2.empty() || !output_video.empty() || !output_video2.empty() ||
                !frames_csv.empty() || !frames_csv2.empty()) {
                throw std::invalid_argument("Video output options require --video");
            }
            if (engine_path.empty() && (!dump_output.empty() || !dump_detections.empty())) {
                throw std::invalid_argument("Inference output export requires --engine");
            }
            run_image(image_path, engine_path, dump_tensor, dump_output, dump_detections);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
