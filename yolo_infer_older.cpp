#include "yolo_infer.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Logger
// ─────────────────────────────────────────────────────────────────────────────

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
        std::cout << "[TRT] " << msg << std::endl;
}

// ─────────────────────────────────────────────────────────────────────────────
// getTensorMemorySize
// ─────────────────────────────────────────────────────────────────────────────

size_t getTensorMemorySize(const nvinfer1::Dims& dims, int elem_size)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims,
                           int64_t{1}, std::multiplies<int64_t>())
           * elem_size;
}

// ─────────────────────────────────────────────────────────────────────────────
// loadEngine
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<nvinfer1::ICudaEngine>
loadEngine(const std::string& engine_path, nvinfer1::IRuntime& runtime)
{
    std::ifstream file(engine_path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open engine file: " + engine_path);

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime.deserializeCudaEngine(data.data(), size));

    if (!engine)
        throw std::runtime_error("Failed to deserialise engine: " + engine_path);

    return engine;
}

// ─────────────────────────────────────────────────────────────────────────────
// letterbox
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat letterbox(const cv::Mat& img,
                  int new_w, int new_h,
                  float& scale, int& x_off, int& y_off)
{
    const int orig_w = img.cols;
    const int orig_h = img.rows;

    scale = std::min(static_cast<float>(new_w) / orig_w,
                     static_cast<float>(new_h) / orig_h);

    const int resized_w = static_cast<int>(orig_w * scale);
    const int resized_h = static_cast<int>(orig_h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, resized_h));

    cv::Mat output(new_h, new_w, CV_8UC3, cv::Scalar(114, 114, 114));

    x_off = (new_w - resized_w) / 2;
    y_off = (new_h - resized_h) / 2;

    resized.copyTo(output(cv::Rect(x_off, y_off, resized_w, resized_h)));

    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// imageToCHW
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> imageToCHW(const cv::Mat& img, int width, int height)
{
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    cv::Mat norm;
    rgb.convertTo(norm, CV_32F, 1.0 / 255.0);

    std::vector<float> buffer(3 * width * height);
    const int HW = width * height;

    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < height; ++h)
            for (int w = 0; w < width; ++w)
                buffer[c * HW + h * width + w] = norm.at<cv::Vec3f>(h, w)[c];

    return buffer;
}

// ─────────────────────────────────────────────────────────────────────────────
// getBestDetection
// ─────────────────────────────────────────────────────────────────────────────

Detection getBestDetection(const float* data,
                           int num_preds, int num_classes,
                           int orig_cols, int orig_rows,
                           float scale, int x_off, int y_off,
                           float conf_thresh)
{
    Detection best{};  // confidence == 0 by default

    for (int i = 0; i < num_preds; ++i)
    {
        // ── find best class for this prediction ──────────────────────────
        float best_score = 0.0f;
        int   cls        = -1;

        for (int c = 0; c < num_classes; ++c)
        {
            const float score = data[(4 + c) * num_preds + i];
            if (score > best_score)
            {
                best_score = score;
                cls        = c;
            }
        }

        if (best_score < conf_thresh)
            continue;

        // ── decode centre-format box and unletterbox ──────────────────────
        const float cx = data[0 * num_preds + i];
        const float cy = data[1 * num_preds + i];
        const float w  = data[2 * num_preds + i];
        const float h  = data[3 * num_preds + i];

        float x1 = (cx - w * 0.5f - x_off) / scale;
        float y1 = (cy - h * 0.5f - y_off) / scale;
        float x2 = (cx + w * 0.5f - x_off) / scale;
        float y2 = (cy + h * 0.5f - y_off) / scale;

        // clamp to original image bounds
        x1 = std::max(0.f, std::min(static_cast<float>(orig_cols), x1));
        y1 = std::max(0.f, std::min(static_cast<float>(orig_rows), y1));
        x2 = std::max(0.f, std::min(static_cast<float>(orig_cols), x2));
        y2 = std::max(0.f, std::min(static_cast<float>(orig_rows), y2));

        const int bw = static_cast<int>(x2 - x1);
        const int bh = static_cast<int>(y2 - y1);

        if (bw <= 0 || bh <= 0)
            continue;

        // ── keep global best ─────────────────────────────────────────────
        if (best_score > best.confidence)
        {
            best.confidence = best_score;
            best.class_id   = cls;
            best.box        = cv::Rect(static_cast<int>(x1),
                                       static_cast<int>(y1), bw, bh);
        }
    }

    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// drawDetection
// ─────────────────────────────────────────────────────────────────────────────

void drawDetection(cv::Mat& image,
                   const Detection& det,
                   const cv::Scalar& color)
{
    if (det.confidence <= 0.f)
        return;

    cv::rectangle(image, det.box, color, 2);

    const std::string label =
        cv::format("C%d %.2f", det.class_id, det.confidence);

    cv::putText(image, label,
                {det.box.x, det.box.y - 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// YoloInferSession – constructor
// ─────────────────────────────────────────────────────────────────────────────

YoloInferSession::YoloInferSession(const std::string& engine_path,
                                   int input_w, int input_h,
                                   const std::string& input_name,
                                   const std::string& output_name)
    : input_w_(input_w)
    , input_h_(input_h)
    , input_name_(input_name)
    , output_name_(output_name)
{
    // Initialise CUDA
    cudaFree(0);

    // Build runtime + engine + context
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
        throw std::runtime_error("Failed to create TensorRT runtime.");

    engine_ = loadEngine(engine_path, *runtime_);

    context_.reset(engine_->createExecutionContext());
    if (!context_)
        throw std::runtime_error("Failed to create execution context.");

    // Bind dynamic input shape
    context_->setInputShape(input_name_.c_str(),
                            nvinfer1::Dims4{1, 3, input_h_, input_w_});

    // Query output dimensions
    const auto output_dims = context_->getTensorShape(output_name_.c_str());

    std::cout << "Output tensor dims:";
    for (int i = 0; i < output_dims.nbDims; ++i)
        std::cout << " " << output_dims.d[i];
    std::cout << std::endl;

    num_preds_   = output_dims.d[2];
    num_classes_ = output_dims.d[1] - 4;

    // Allocate GPU memory
    input_byte_size_  = 3 * input_h_ * input_w_ * sizeof(float);
    output_byte_size_ = getTensorMemorySize(output_dims, sizeof(float));

    cudaMalloc(&input_mem_,  input_byte_size_);
    cudaMalloc(&output_mem_, output_byte_size_);

    output_buf_.resize(output_byte_size_ / sizeof(float));

    // Create CUDA stream
    cudaStreamCreate(&stream_);
}

// ─────────────────────────────────────────────────────────────────────────────
// YoloInferSession – destructor
// ─────────────────────────────────────────────────────────────────────────────

YoloInferSession::~YoloInferSession()
{
    if (input_mem_)  cudaFree(input_mem_);
    if (output_mem_) cudaFree(output_mem_);
    if (stream_)     cudaStreamDestroy(stream_);
}

// ─────────────────────────────────────────────────────────────────────────────
// YoloInferSession – infer
// ─────────────────────────────────────────────────────────────────────────────

Detection YoloInferSession::infer(const cv::Mat& image, float conf_thresh)
{
    // ── Preprocess ───────────────────────────────────────────────────────
    float scale;
    int   x_off, y_off;

    const cv::Mat lb_img = letterbox(image, input_w_, input_h_,
                                     scale, x_off, y_off);
    const auto input_data = imageToCHW(lb_img, input_w_, input_h_);

    // ── Upload & infer ───────────────────────────────────────────────────
    cudaMemcpyAsync(input_mem_, input_data.data(),
                    input_byte_size_, cudaMemcpyHostToDevice, stream_);

    context_->setTensorAddress(input_name_.c_str(),  input_mem_);
    context_->setTensorAddress(output_name_.c_str(), output_mem_);

    context_->enqueueV3(stream_);

    cudaMemcpyAsync(output_buf_.data(), output_mem_,
                    output_byte_size_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // ── Postprocess ──────────────────────────────────────────────────────
    return getBestDetection(output_buf_.data(),
                            num_preds_, num_classes_,
                            image.cols, image.rows,
                            scale, x_off, y_off,
                            conf_thresh);
}
