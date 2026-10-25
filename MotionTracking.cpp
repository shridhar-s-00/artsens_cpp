#include "MotionTracking.h"

MotionTracking::MotionTracking(
    CircularBuffer<Artery_locator_data>* in,
    CircularBuffer<Artery_locator_data>* out
)
    : ConnectProcessor(in, out)
{   


    py::gil_scoped_acquire acquire;
    
    py::module wall_module =
        py::module::import("wall_detector");

    wallDetector =
        wall_module.attr("WallDetector")();
}


void Cross_Correlation(const std::vector<float>& input_x, const std::vector<float>& input_y, std::vector<float>& output){

	
	
	
	
	
	output.clear();
	
	if (input_x.empty() || input_y.empty()) // Incase some dumb ass gives empty data
	{
		return;
	}


	
	
	int N = input_x.size();
	int M = input_y.size();
	
	
	output.reserve(N + M - 1);
	
		
	float temp_sum=0;
	
	
	// Cross Correlation Logic ( O(N*2M) ig)
	
	
	
	for(int lag = -(M - 1); lag <= N - 1; lag++){    // Iterate over  range 
					
		temp_sum=0;
		
		for(int i=0; i < N; i++){
		
			
			if ((i + lag) >= 0 && (i + lag) < M){   // Boundary Condition
				
				
				
				temp_sum+= input_x[i]*input_y[i+lag];
			}
		}
		
		output.push_back(temp_sum);
		
		
	}
	
}


void Correlation_Lags(int N, int M, std::vector<int>& outputLags){

	
	outputLags.clear();
    	outputLags.reserve(N + M - 1);
	
	
	
	for(int i= -(M-1); i <= (N-1) ; i++){
		
		
		outputLags.push_back(i);
	
	}
	
}
	
	
	
int find_lag(
    const std::vector<float>& frame1,
    const std::vector<float>& frame2)
{
    std::vector<float> correlation;
    std::vector<int> lags;

    Cross_Correlation(
        frame1,
        frame2,
        correlation
    );

    Correlation_Lags(
        frame1.size(),
        frame2.size(),
        lags
    );

    if (correlation.empty())
        return 0;

    auto maxElement =
        std::max_element(
            correlation.begin(),
            correlation.end()
        );

    size_t maxIndex =
        std::distance(
            correlation.begin(),
            maxElement
        );

    return lags[maxIndex];
}




void MotionTracking::motion_artefact_tracking(
    const std::vector<std::vector<float>>& frameBuffer,
    const int coordinates[]
)
{
        
  // std::cout<<"tryint to go to framebuffer 0"<<std::endl;
    nw_shifted.clear();
    fw_shifted.clear();
    wall_peaks.clear();
   // std::cout<<"tryint to go to framebuffer 1"<<std::endl;
    int roi_x1 = coordinates[0];                      // roi_x1 and roi_x2 are your bounding box returned coordinates along x axis plane
    int roi_x2 = coordinates[2];

    roi_x1 = static_cast<int>(                        // static cast for ensuring int type is returned
        std::floor(roi_x1 * SCALING_CONST));

    roi_x2 = static_cast<int>(
        std::ceil(roi_x2 * SCALING_CONST));

    
    
    int roi_center= (roi_x1 + roi_x2)/2;
    
    std::vector<std::vector<float>> tempDistentionBuffer;
    
    
    
 //   std::cout<<"tryint to go to framebuffer 2"<<std::endl;
    for (size_t i = 0; i < frameBuffer.size(); ++i)
    {
        const auto& frame = frameBuffer[i];
        
        
     
        
        py::list py_frame;

        for (float value : frame)
            py_frame.append(value);

        py::list py_coordinates;

        for (int j = 0; j < 4; ++j)
            py_coordinates.append(coordinates[j]);

        py::tuple result =
            wallDetector.attr("wall_detection")(
                py_frame,
                py_coordinates
            );

        py::list peaks =
            result[0].cast<py::list>();

        wall_peaks.clear();

        for (auto peak : peaks)
        {
            int peak_value = peak.cast<int>();

            wall_peaks.push_back(peak_value);

            
        }
       
        
        int center = (wall_peaks[0] + wall_peaks[1])/2;
        
        int shift = center - roi_center;
        shift = 0;
        //shift_center_buffer.push_back(shift);
        
        
        nw_shifted.push_back((wall_peaks[0] - roi_x1) - shift);
        fw_shifted.push_back((wall_peaks[1] - roi_x1) - shift);
        
        if (first_nw_fw_done ==0){
                NW_prev= (wall_peaks[0] - roi_x1) ;
                FW_prev= (wall_peaks[1] - roi_x1) ;
                first_nw_fw_done=1;
        }
        
        int start = roi_x1 + shift; //shift_center_buffer[i];
        int end   = roi_x2 + shift; //shift_center_buffer[i];

        std::vector<float> shiftedFrame(
                frame.begin() + start,
                frame.begin() + end
         );

         tempDistentionBuffer.push_back(
                std::move(shiftedFrame)
         );
           
    }
    
     //std::cout<<" out of framebuffer"<<std::endl;
    
    // setting the boundaries for the correlation windows------------------
    
    
      int min_nw = *std::min_element(
            nw_shifted.begin(),
            nw_shifted.end()
        );

      int max_nw = *std::max_element(
            nw_shifted.begin(),
            nw_shifted.end()
        );

      int nw_window_min = std::max(
            min_nw - PEAK_REGION_HALF_WIDTH_DEFAULT,
            0
        );

      int nw_window_max =
            max_nw + PEAK_REGION_HALF_WIDTH_DEFAULT;


      int min_fw = *std::min_element(
            fw_shifted.begin(),
            fw_shifted.end()
        );

      int max_fw = *std::max_element(
            fw_shifted.begin(),
            fw_shifted.end()
        );

      int fw_window_min =
            min_fw - PEAK_REGION_HALF_WIDTH_DEFAULT;

      int fw_window_max =
            max_fw + PEAK_REGION_HALF_WIDTH_DEFAULT;
            
     //--------------------------------------------------------------------------
     
     
     
        
     
     
     for (size_t i = 1; i < tempDistentionBuffer.size(); ++i)
        {
            const auto& currentFrame = tempDistentionBuffer[i];
            const auto& previousFrame = tempDistentionBuffer[i - 1];

            std::vector<float> currentNW(
                currentFrame.begin() + nw_window_min,
                currentFrame.begin() + nw_window_max
            );

            std::vector<float> previousNW(
                previousFrame.begin() + nw_window_min,
                previousFrame.begin() + nw_window_max
            );

            int t0 = find_lag(currentNW, previousNW);


            std::vector<float> currentFW(
                currentFrame.begin() + fw_window_min,
                currentFrame.begin() + fw_window_max
            );

            std::vector<float> previousFW(
                previousFrame.begin() + fw_window_min,
                previousFrame.begin() + fw_window_max
            );

            int t1 = find_lag(currentFW, previousFW);


            NW_prev += t0;
            FW_prev += t1;

            distentionBuffer.push_back(
                std::abs(FW_prev - NW_prev)
            );
        }
        
        
      // add the remove outlier function here
      
 
       
    
}







Artery_locator_data MotionTracking::process(
    const Artery_locator_data& InDataPacket
)
{
    
    RFBuffer.push_back(InDataPacket.latest_RF_frame);
    if(!InDataPacket.output_image.empty()){
    
    output_data_packet.output_image= InDataPacket.output_image;
    
    }
    
    if (RFBuffer.size() < 20)
    {
        return output_data_packet;
    }

    // Run motion tracking on 20-frame buffer
    
    const cv::Rect& box = InDataPacket.detection.box;
    int coordinates[4] = {
        box.x,
        box.y,
        box.x + box.width,
        box.y + box.height
    };
    
    
    if (InDataPacket.detection.confidence > CONF_THRESH){
    py::gil_scoped_acquire acquire;
    //std::cout<<"GIL ACQUIRED"<<std::endl;
    motion_artefact_tracking(
        RFBuffer,
        coordinates
    );

    }
    

    output_data_packet.dist_history = distentionBuffer;
    

    RFBuffer.clear();

    return output_data_packet;
}
