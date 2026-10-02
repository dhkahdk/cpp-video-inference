#include "multi_stream_pipeline.hpp"

#include "postprocess.hpp"
#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using SteadyClock = std::chrono::steady_clock;

int64_t unix_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

double elapsed_ms(SteadyClock::time_point start, SteadyClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct FrameRecord {
    size_t frame_id = 0;
    double media_pts_ms = 0.0;
    int64_t capture_unix_ms = 0;
    std::string status = "queued";
    size_t queue_depth_after_enqueue = 0;
    std::optional<size_t> output_frame_id;
    std::optional<int64_t> output_unix_ms;
    std::optional<double> queue_wait_ms;
    std::optional<double> capture_to_output_ms;
    std::optional<double> decode_ms;
    std::optional<double> frame_handoff_ms;
    std::optional<double> preprocess_ms;
    std::optional<double> h2d_gpu_ms;
    std::optional<double> inference_gpu_ms;
    std::optional<double> d2h_gpu_ms;
    std::optional<double> infer_wall_ms;
    std::optional<double> postprocess_ms;
    std::optional<double> draw_ms;
    std::optional<double> write_ms;
    std::optional<size_t> detections;
    std::optional<size_t> drawn_detections;
};

struct FramePacket {
    int stream_id = 0;
    cv::Mat frame;
    SteadyClock::time_point captured_at;
    std::shared_ptr<FrameRecord> record;
};

class BoundedQueues {
public:
    explicit BoundedQueues(size_t capacity) : capacity_(capacity) {
        if (capacity == 0) throw std::invalid_argument("Queue capacity must be positive");
    }

    std::shared_ptr<FrameRecord> push(FramePacket packet, StreamSummary& summary) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& queue = queues_.at(packet.stream_id);
        if (aborted_) return nullptr;
        std::shared_ptr<FrameRecord> dropped;
        if (queue.size() == capacity_) {
            dropped = std::move(queue.front().record);
            queue.pop_front();
            ++summary.dropped;
        }
        queue.push_back(std::move(packet));
        queue.back().record->queue_depth_after_enqueue = queue.size();
        summary.max_queue_depth = std::max(summary.max_queue_depth, queue.size());
        ready_.notify_one();
        return dropped;
    }

    bool pop(FramePacket& packet) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] {
            return aborted_ || !queues_[0].empty() || !queues_[1].empty() ||
                   (closed_[0] && closed_[1]);
        });
        if (aborted_) return false;
        for (int offset = 0; offset < 2; ++offset) {
            const int stream = (next_stream_ + offset) % 2;
            if (!queues_[stream].empty()) {
                packet = std::move(queues_[stream].front());
                queues_[stream].pop_front();
                next_stream_ = (stream + 1) % 2;
                return true;
            }
        }
        return false;
    }

    void close(int stream_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_.at(stream_id) = true;
        ready_.notify_all();
    }

    void abort() {
        std::lock_guard<std::mutex> lock(mutex_);
        aborted_ = true;
        for (auto& queue : queues_) {
            for (auto& packet : queue) packet.record->status = "aborted";
            queue.clear();
        }
        ready_.notify_all();
    }

private:
    size_t capacity_;
    std::array<std::deque<FramePacket>, 2> queues_;
    std::array<bool, 2> closed_{false, false};
    int next_stream_ = 0;
    bool aborted_ = false;
    std::mutex mutex_;
    std::condition_variable ready_;
};

struct StreamRuntime {
    double fps = 0.0;
    cv::Size frame_size;
    cv::VideoWriter writer;
    std::ofstream csv;
    std::mutex csv_mutex;
    size_t csv_rows = 0;
    size_t output_frame_id = 0;
};

void validate_video(const VideoOptions& options, StreamRuntime& runtime) {
    cv::VideoCapture capture(options.input.string());
    if (!capture.isOpened()) throw std::runtime_error("Could not open input video: " + options.input.string());
    runtime.fps = capture.get(cv::CAP_PROP_FPS);
    cv::Mat first;
    if (!std::isfinite(runtime.fps) || runtime.fps <= 0.0 ||
        !capture.read(first) || first.empty()) {
        throw std::runtime_error("Input video lacks valid FPS or frames: " + options.input.string());
    }
    runtime.frame_size = first.size();
    const auto parent = options.output_video.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    runtime.writer.open(options.output_video.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        runtime.fps, runtime.frame_size, true);
    if (!runtime.writer.isOpened()) {
        throw std::runtime_error("Could not open output video: " + options.output_video.string());
    }
    const auto csv_parent = options.frames_csv.parent_path();
    if (!csv_parent.empty()) std::filesystem::create_directories(csv_parent);
    runtime.csv.open(options.frames_csv);
    if (!runtime.csv) throw std::runtime_error("Could not open stream CSV: " + options.frames_csv.string());
    runtime.csv << "frame_id,media_pts_ms,capture_unix_ms,status,queue_depth_after_enqueue,"
                   "output_frame_id,output_unix_ms,queue_wait_ms,capture_to_output_ms,"
                   "detections,drawn_detections,decode_ms,frame_handoff_ms,preprocess_ms,h2d_gpu_ms,"
                   "inference_gpu_ms,d2h_gpu_ms,infer_wall_ms,postprocess_ms,draw_ms,write_ms\n"
                << std::fixed << std::setprecision(3);
    if (!runtime.csv) throw std::runtime_error("Could not write stream CSV header");
}

size_t draw_boxes(cv::Mat& frame, const std::vector<Detection>& detections, float threshold) {
    static const std::array<const char*, 5> names = {
        "class_0", "class_1", "class_2", "class_3", "class_4"};
    size_t count = 0;
    for (const auto& detection : detections) {
        if (detection.score < threshold) continue;
        const cv::Point a(static_cast<int>(std::round(detection.x1)),
                          static_cast<int>(std::round(detection.y1)));
        const cv::Point b(static_cast<int>(std::round(detection.x2)),
                          static_cast<int>(std::round(detection.y2)));
        cv::rectangle(frame, a, b, cv::Scalar(0, 255, 255), 2);
        const std::string label = std::string(names.at(detection.class_id)) + " " +
                                  cv::format("%.2f", detection.score);
        cv::putText(frame, label, cv::Point(a.x, std::max(16, a.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
        ++count;
    }
    return count;
}

void write_record(StreamRuntime& runtime, const FrameRecord& record) {
    // Producers finish dropped frames; the consumer finishes processed frames.
    // Completion order can differ from source frame order during congestion.
    std::lock_guard<std::mutex> lock(runtime.csv_mutex);
    auto& csv = runtime.csv;
    csv << record.frame_id << ',' << record.media_pts_ms << ','
        << record.capture_unix_ms << ',' << record.status << ','
        << record.queue_depth_after_enqueue << ',';
    if (record.output_frame_id) csv << *record.output_frame_id;
    csv << ',';
    if (record.output_unix_ms) csv << *record.output_unix_ms;
    csv << ',';
    if (record.queue_wait_ms) csv << *record.queue_wait_ms;
    csv << ',';
    if (record.capture_to_output_ms) csv << *record.capture_to_output_ms;
    csv << ',';
    if (record.detections) csv << *record.detections;
    csv << ',';
    if (record.drawn_detections) csv << *record.drawn_detections;
    const std::array<std::optional<double>, 10> times = {
        record.decode_ms, record.frame_handoff_ms, record.preprocess_ms, record.h2d_gpu_ms,
        record.inference_gpu_ms, record.d2h_gpu_ms, record.infer_wall_ms,
        record.postprocess_ms, record.draw_ms, record.write_ms};
    for (const auto& value : times) {
        csv << ',';
        if (value) csv << *value;
    }
    csv << '\n';
    if (!csv) throw std::runtime_error("Could not write stream CSV");
    ++runtime.csv_rows;
}
}  // namespace

std::array<StreamSummary, 2> process_two_streams(const MultiStreamOptions& options,
                                                 TrtEngine& engine) {
    if (options.queue_capacity == 0) throw std::invalid_argument("Queue capacity must be positive");
    std::array<StreamRuntime, 2> runtime;
    std::array<StreamSummary, 2> summary;
    for (int stream = 0; stream < 2; ++stream) {
        const auto& config = options.streams[stream];
        if (config.display_threshold < 0.0f || config.display_threshold > 1.0f ||
            config.output_video.empty() || config.frames_csv.empty()) {
            throw std::invalid_argument("Invalid two-stream output paths or display threshold");
        }
        validate_video(config, runtime[stream]);
        summary[stream].source_fps = runtime[stream].fps;
    }

    BoundedQueues queues(options.queue_capacity);
    std::mutex error_mutex;
    std::exception_ptr thread_error;
    auto save_error = [&](std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!thread_error) thread_error = error;
        queues.abort();
    };

    std::array<std::thread, 2> producers;
    for (int stream = 0; stream < 2; ++stream) {
        producers[stream] = std::thread([&, stream] {
            try {
                cv::VideoCapture capture(options.streams[stream].input.string());
                if (!capture.isOpened()) throw std::runtime_error("Producer could not open video");
                const auto start = SteadyClock::now();
                cv::Mat frame;
                size_t frame_id = 0;
                while (true) {
                    const auto decode_start = SteadyClock::now();
                    if (!capture.read(frame) || frame.empty()) break;
                    const auto decode_end = SteadyClock::now();
                    const auto target = start + std::chrono::duration_cast<SteadyClock::duration>(
                        std::chrono::duration<double>(frame_id / runtime[stream].fps));
                    std::this_thread::sleep_until(target);
                    auto record = std::make_shared<FrameRecord>();
                    record->frame_id = frame_id;
                    record->media_pts_ms = static_cast<double>(frame_id) * 1000.0 / runtime[stream].fps;
                    record->capture_unix_ms = unix_time_ms();
                    record->decode_ms = elapsed_ms(decode_start, decode_end);
                    const auto handoff_start = SteadyClock::now();
                    // The queue owns these pixels; the next read refills the moved-from Mat.
                    cv::Mat queued_frame = std::move(frame);
                    const auto handoff_end = SteadyClock::now();
                    record->frame_handoff_ms = elapsed_ms(handoff_start, handoff_end);
                    ++summary[stream].decoded;
                    auto dropped = queues.push(
                        {stream, std::move(queued_frame), SteadyClock::now(), record}, summary[stream]);
                    if (dropped) {
                        dropped->status = "dropped_oldest";
                        write_record(runtime[stream], *dropped);
                    }
                    ++frame_id;
                }
                queues.close(stream);
            } catch (...) {
                queues.close(stream);
                save_error(std::current_exception());
            }
        });
    }

    try {
        FramePacket packet;
        while (queues.pop(packet)) {
            const auto inference_start = SteadyClock::now();
            const auto preprocessed = preprocess_image(packet.frame);
            const auto preprocess_end = SteadyClock::now();
            TrtEngine::Timings timings;
            const auto raw = engine.infer(preprocessed.tensor, &timings);
            const auto postprocess_start = SteadyClock::now();
            const auto detections = decode_detections(raw, packet.frame.cols, packet.frame.rows,
                                                      preprocessed.ratio, preprocessed.pad_w,
                                                      preprocessed.pad_h);
            const auto postprocess_end = SteadyClock::now();
            const size_t drawn = draw_boxes(packet.frame, detections,
                options.streams[packet.stream_id].display_threshold);
            const auto draw_end = SteadyClock::now();
            auto& state = runtime[packet.stream_id];
            state.writer.write(packet.frame);
            const auto completed = SteadyClock::now();
            packet.record->status = "processed";
            packet.record->output_frame_id = state.output_frame_id++;
            packet.record->output_unix_ms = unix_time_ms();
            packet.record->queue_wait_ms = elapsed_ms(packet.captured_at, inference_start);
            packet.record->capture_to_output_ms = elapsed_ms(packet.captured_at, completed);
            packet.record->preprocess_ms = elapsed_ms(inference_start, preprocess_end);
            packet.record->h2d_gpu_ms = timings.h2d_gpu_ms;
            packet.record->inference_gpu_ms = timings.inference_gpu_ms;
            packet.record->d2h_gpu_ms = timings.d2h_gpu_ms;
            packet.record->infer_wall_ms = timings.infer_wall_ms;
            packet.record->postprocess_ms = elapsed_ms(postprocess_start, postprocess_end);
            packet.record->draw_ms = elapsed_ms(postprocess_end, draw_end);
            packet.record->write_ms = elapsed_ms(draw_end, completed);
            packet.record->detections = detections.size();
            packet.record->drawn_detections = drawn;
            write_record(state, *packet.record);
            ++summary[packet.stream_id].processed;
        }
    } catch (...) {
        save_error(std::current_exception());
    }

    for (auto& producer : producers) producer.join();
    for (auto& state : runtime) state.writer.release();
    if (thread_error) std::rethrow_exception(thread_error);
    for (int stream = 0; stream < 2; ++stream) {
        if (summary[stream].decoded != summary[stream].processed + summary[stream].dropped) {
            throw std::runtime_error("Decoded frame accounting mismatch");
        }
        if (runtime[stream].csv_rows != summary[stream].decoded) {
            throw std::runtime_error("CSV frame accounting mismatch");
        }
        runtime[stream].csv.close();
        if (!runtime[stream].csv) throw std::runtime_error("Could not close stream CSV");
    }
    return summary;
}
