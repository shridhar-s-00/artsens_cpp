#pragma once

#include <string>
#include <vector>
#include <memory>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// TensorRT Logger
// ─────────────────────────────────────────────────────────────────────────────

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Detection result
// ─────────────────────────────────────────────────────────────────────────────

struct Detection {
    cv::Rect box;
    float    confidence;
    int      class_id;
};

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Resize image with letterboxing (grey padding to preserve aspect ratio).
 *
 * @param img     Source BGR image.
 * @param new_w   Target width.
 * @param new_h   Target height.
 * @param scale   [out] Scale factor applied to the original image.
 * @param x_off   [out] Horizontal padding offset (pixels).
 * @param y_off   [out] Vertical padding offset (pixels).
 * @return        Letterboxed BGR image of size (new_w x new_h).
 */
cv::Mat letterbox(const cv::Mat& img,
                  int new_w, int new_h,
                  float& scale, int& x_off, int& y_off);

/**
 * Convert a BGR image to a flat CHW float buffer normalised to [0, 1].
 * Channel order is converted to RGB before packing.
 *
 * @param img    Source BGR image (already resized to width x height).
 * @param width  Expected image width.
 * @param height Expected image height.
 * @return       Float buffer of length 3 * width * height.
 */
std::vector<float> imageToCHW(const cv::Mat& img, int width, int height);

// ─────────────────────────────────────────────────────────────────────────────
// Engine helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute the total byte size of a TensorRT tensor.
 *
 * @param dims      Tensor dimensions.
 * @param elem_size Size of one element in bytes (e.g. sizeof(float)).
 * @return          Total size in bytes.
 */
size_t getTensorMemorySize(const nvinfer1::Dims& dims, int elem_size);

/**
 * Load a serialised TensorRT engine from disk and deserialise it.
 *
 * @param engine_path Path to the .engine file.
 * @param logger      TensorRT logger instance.
 * @return            Owning pointer to the deserialised ICudaEngine,
 *                    or nullptr on failure.
 */
std::unique_ptr<nvinfer1::ICudaEngine>
loadEngine(const std::string& engine_path,
           nvinfer1::IRuntime& runtime);

// ─────────────────────────────────────────────────────────────────────────────
// Postprocessing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Parse raw YOLO output tensor and return the single highest-confidence
 * detection above the confidence threshold (no NMS needed for single-pick).
 *
 * The expected output layout is [1, 4 + num_classes, num_preds] in
 * column-major prediction order (each attribute is a contiguous row).
 *
 * @param data         Pointer to the flat host-side output buffer.
 * @param num_preds    Number of candidate predictions.
 * @param num_classes  Number of object classes.
 * @param orig_cols    Width  of the original (pre-letterbox) image.
 * @param orig_rows    Height of the original (pre-letterbox) image.
 * @param scale        Letterbox scale factor (returned by letterbox()).
 * @param x_off        Letterbox horizontal offset (returned by letterbox()).
 * @param y_off        Letterbox vertical offset  (returned by letterbox()).
 * @param conf_thresh  Minimum confidence to accept a detection.
 * @return             Best Detection, or a Detection with confidence == 0
 *                     if nothing passes the threshold.
 */
Detection getBestDetection(const float* data,
                           int num_preds, int num_classes,
                           int orig_cols, int orig_rows,
                           float scale, int x_off, int y_off,
                           float conf_thresh = 0.25f);

// ─────────────────────────────────────────────────────────────────────────────
// Visualisation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Draw a bounding box and label onto a BGR image (in-place).
 *
 * @param image     Image to annotate.
 * @param det       Detection to draw; nothing is drawn if confidence == 0.
 * @param color     Box and text colour (BGR).
 */
void drawDetection(cv::Mat& image,
                   const Detection& det,
                   const cv::Scalar& color = {0, 255, 0});

// ─────────────────────────────────────────────────────────────────────────────
// High-level inference pipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Self-contained YOLO TensorRT inference session.
 *
 * Owns GPU memory, CUDA stream, engine, and context for the lifetime of
 * the object.  Call infer() for each image.
 */
class YoloInferSession {
public:
    /**
     * Construct and initialise the session.
     *
     * @param engine_path  Path to the serialised .engine file.
     * @param input_w      Model input width.
     * @param input_h      Model input height.
     * @param input_name   Name of the input tensor.
     * @param output_name  Name of the output tensor.
     */
    YoloInferSession(const std::string& engine_path,
                     int input_w = 800, int input_h = 800,
                     const std::string& input_name  = "images",
                     const std::string& output_name = "output0");

    ~YoloInferSession();

    // Non-copyable, movable
    YoloInferSession(const YoloInferSession&)            = delete;
    YoloInferSession& operator=(const YoloInferSession&) = delete;
    YoloInferSession(YoloInferSession&&)                 = default;
    YoloInferSession& operator=(YoloInferSession&&)      = default;

    /**
     * Run inference on a single BGR image.
     *
     * @param image        Input BGR image (any size).
     * @param conf_thresh  Confidence threshold.
     * @return             Best detection found, or {confidence == 0} if none.
     */
    Detection infer(const cv::Mat& image, float conf_thresh = 0.25f);

    int numClasses() const { return num_classes_; }
    int inputWidth()  const { return input_w_; }
    int inputHeight() const { return input_h_; }

private:
    Logger logger_;

    std::unique_ptr<nvinfer1::IRuntime>        runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>     engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    int input_w_    = 0;
    int input_h_    = 0;
    int num_preds_  = 0;
    int num_classes_ = 0;

    std::string input_name_;
    std::string output_name_;

    void* input_mem_  = nullptr;
    void* output_mem_ = nullptr;
    size_t input_byte_size_  = 0;
    size_t output_byte_size_ = 0;

    cudaStream_t stream_ = nullptr;
    std::vector<float> channel_buf_;
    std::vector<float> output_buf_;
    cv::Mat lb_img_; 
};
