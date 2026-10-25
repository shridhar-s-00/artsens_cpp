#pragma once


#include <vector>
#include <cmath>
#include <algorithm>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "threadlib/processor.hpp"
#include "threadlib/threadqueue.hpp"

#include "ArteryLocatorData.h"

#define SCALING_CONST 4000 / 1024
#define PEAK_REGION_HALF_WIDTH_DEFAULT 50


#define FS 80e6
#define FILT_ORDER 4
#define LOW_CUTOFF 800000

namespace py = pybind11;

class MotionTracking
    : public ConnectProcessor<Artery_locator_data, Artery_locator_data>
{
private:

    Artery_locator_data output_data_packet;

    std::vector<int> nw_shifted;
    std::vector<int> fw_shifted;
    std::vector<int> wall_peaks;
    std::vector<int> distentionBuffer;
    std::vector<std::vector<float>> RFBuffer;
    
    
    int first_nw_fw_done = 0;
    int NW_prev = 0;
    int FW_prev = 0;
    const float CONF_THRESH = 0.25f;


    py::object wallDetector;


    void motion_artefact_tracking(
        const std::vector<std::vector<float>>& frameBuffer,
        const int coordinates[]
    );


public:

    MotionTracking(
        CircularBuffer<Artery_locator_data>* in,
        CircularBuffer<Artery_locator_data>* out
    );

    Artery_locator_data process(
        const Artery_locator_data& InDataPacket
    ) override;
};


void Cross_Correlation(const std::vector<float>& input_x, const std::vector<float>& input_y, std::vector<float>& output);

void Correlation_Lags(int N, int M, std::vector<int>& outputLags);

int find_lag(const std::vector<float>& frame1,const std::vector<float>& frame2);

