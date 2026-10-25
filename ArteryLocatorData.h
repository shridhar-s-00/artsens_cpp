#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "yolo_infer_older.h"

struct Artery_locator_data
{
    cv::Mat output_image;
    Detection detection;
    std::vector<float> latest_RF_frame;
    std::vector<int> dist_history;
};
