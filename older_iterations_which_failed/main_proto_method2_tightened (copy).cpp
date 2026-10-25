





#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/signal_processing/signal_processing.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/threadlib/processor.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/threadlib/threadqueue.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/utils/utils.hpp"

#include "yolo_infer_older.h"
#include "MotionTracking.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


#include <numeric>
	
using namespace std;


class DataSource : public SourceProcessor<std::vector<uint8_t>>
{


	public:
	
		 
		 
		 
		 ImageInfo info ;
		 std::ifstream file;
		 int frame_size;
		 uint8_t frame_counter;
	         
	         
	         DataSource(CircularBuffer<std::vector<uint8_t>>* out):SourceProcessor(out) {
	         
	         	info =  parse_image_name("110824_1_295_685_PLRAW_1");
	         	
	         	file.open("/home/act/Project Senna/Signal files/Copy of 110109_F_D_D_Left Carotid_Sit_(80MSps_4000_50Hz)_T1_LRAW.sgl",std::ios::binary);
	         	
	         	frame_size= 4*4000;
	         	
	         	
	         	
	         }
	         
	         std::vector<uint8_t> process() override
	         {
	         	 
	         	 std::this_thread::sleep_for(std::chrono::milliseconds(20));
	         	 
	    
	         	 std::vector<uint8_t> data_vector= read_one_frame(file, frame_size, DEAD_BAND_WIDTH);
	         	 
	         	 
	         	 return data_vector;
	         }

};

struct Artery_locator_data{

	cv::Mat output_image;
	Detection detection;
	std::vector<float> latest_RF_frame;
	std::vector<float> dist_history;
};


class FrameProcessor: public ConnectProcessor<std::vector<uint8_t>,Artery_locator_data>
{
	private:
	       std::vector<float> hilbert_filtered_data;
	       SignalPreprocessing preprocessor;
	       std::vector<std::vector<float>> frame_buffer;
	       cv::Mat grayscale_image;
	       Artery_locator_data data_packet;
	
	public:
		
		FrameProcessor(CircularBuffer<std::vector<uint8_t>>* in,CircularBuffer<Artery_locator_data>* out):ConnectProcessor(in,out) {
		    
		}
		Artery_locator_data process(const std::vector<uint8_t>& raw_data) override
		{
			
			
			
			preprocessor.filter_RF_frame(raw_data,hilbert_filtered_data);
		
			if(frame_buffer.size()<NO_OF_FRAMES){
			    frame_buffer.push_back(hilbert_filtered_data);
			    //return cv::Mat();
			    return data_packet;
			 }
			 else{
			    frame_buffer.erase(frame_buffer.begin());
			    frame_buffer.push_back(hilbert_filtered_data);
			 }
			         	
			
			 
	         	preprocessor.frame_buffer_to_image(frame_buffer,grayscale_image);
	         	data_packet.output_image= grayscale_image;
	         	data_packet.latest_RF_frame= hilbert_filtered_data;
	         	
	         	//return grayscale_image;
	
			return data_packet;
		}
		
};





class InferenceEngine: public ConnectProcessor<Artery_locator_data, Artery_locator_data>
{
	private: 
	
	YoloInferSession session;
    	
    	const float CONF_THRESH = 0.25f;
    	
    	Artery_locator_data output_data_packet;
    	
    	
    	public:
    	
    	InferenceEngine(CircularBuffer<Artery_locator_data>* in,CircularBuffer<Artery_locator_data>* out):ConnectProcessor(in,out), session(
        "/home/act/Project Senna/models/yolo_b1.engine",
        /*input_w=*/800, /*input_h=*/800,
        /*input_name=*/"images",
        /*output_name=*/"output0"
    	) {
		    
	}
    	
    	//"/home/act/Project Senna/models/yolo_b1.engine   yolo10n_fp16_b1",
       
    	
    	Artery_locator_data process(const Artery_locator_data& m_mode_image_data) override
    	{	
    		
    		
    		cv::Mat m_mode_image= m_mode_image_data.output_image;
    		
    		cv::Mat out = m_mode_image.clone();
    		
    		if(!m_mode_image.empty()){
    			static int count = 0;
    			static auto start = std::chrono::steady_clock::now();

			count++;

	    		Detection det = session.infer(m_mode_image, CONF_THRESH);

			output_data_packet.detection= det;

			
			auto now = std::chrono::steady_clock::now();
			double elapsed = std::chrono::duration<double>(now - start).count();

			if (elapsed >= 1.0) {
			    //std::cout << "Inference FPS: " << count / elapsed << std::endl;
			    count = 0;
			    start = now;
			}
						
			
                        
			if (det.confidence > CONF_THRESH)
			    {
				drawDetection(out, det);
				
			    }
			else
			    {
				//std::cout << "No detection above threshold." << std::endl;
				
			    }
			
		}
		
		output_data_packet.output_image= out;
		output_data_packet.latest_RF_frame= m_mode_image_data.latest_RF_frame;
		
		
		
		return output_data_packet;
	}
    	
    	
};   	



    		
    	     


class DisplayProcessor : public SinkProcessor<Artery_locator_data>
{
private:
    const int width_distention = 800;
    const int height_distention = 400;

    size_t frame_count = 0;
    std::chrono::steady_clock::time_point start_time;

    cv::Mat dist_canvas;   
    std::vector<float> dist_history;

public:
    DisplayProcessor(CircularBuffer<Artery_locator_data>* in)
        : SinkProcessor(in),
          dist_canvas(height_distention, width_distention, CV_8UC3) 
    {
        start_time = std::chrono::steady_clock::now();
        dist_canvas.setTo(cv::Scalar(0, 0, 0));
    }

    void process(const Artery_locator_data& display_data) override
    {
        if (!display_data.output_image.empty())
        {
           
            dist_canvas.setTo(cv::Scalar(0, 0, 0));

            dist_history = display_data.dist_history;

            if (dist_history.empty()) return;

            float min_d = *std::min_element(dist_history.begin(), dist_history.end());
            float max_d = *std::max_element(dist_history.begin(), dist_history.end());
            float range_d = std::max(max_d - min_d, 1e-6f);

            int N_dist = dist_history.size();
            float x_scale_dist = (float)(width_distention - 1) / std::max(N_dist - 1, 1);

            for (int i = 1; i < N_dist; i++)
            {
                int x1_d = (int)((i - 1) * x_scale_dist);
                int x2_d = (int)(i * x_scale_dist);

                float norm1 = (dist_history[i - 1] - min_d) / range_d;
                float norm2 = (dist_history[i]     - min_d) / range_d;

                int y1_d = height_distention - (int)(norm1 * height_distention);
                int y2_d = height_distention - (int)(norm2 * height_distention);

                cv::line(dist_canvas,
                         cv::Point(x1_d, y1_d),
                         cv::Point(x2_d, y2_d),
                         cv::Scalar(255, 255, 0), 2);
            }

            cv::imshow("Live Signal", dist_canvas);
            cv::imshow("M-Mode", display_data.output_image);
            cv::waitKey(1);
        }
    }
};


	

int main(){

       CircularBuffer<std::vector<uint8_t>> data_to_sp(1024);
       CircularBuffer<Artery_locator_data> sp_to_inference(1024);
       CircularBuffer<Artery_locator_data> inference_to_locator(1024);
       CircularBuffer<Artery_locator_data> locator_to_display(1024);
       
       DataSource source(&data_to_sp);
       FrameProcessor frameprocessor(&data_to_sp,&sp_to_inference);
       InferenceEngine inference(&sp_to_inference,&inference_to_locator);
       MotionTracking artery_locator(&inference_to_locator,&locator_to_display);
       
       DisplayProcessor display(&locator_to_display);
       
       
       source.start();
       frameprocessor.start();
       inference.start();
       artery_locator.start();
       display.start();
       
       
       while(true){
       
       		std::this_thread::sleep_for(std::chrono::seconds(1));
       	}
       
       
       return 0;
}
