





#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/signal_processing/signal_processing.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/threadlib/processor.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/threadlib/threadqueue.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/utils/utils.hpp"

#include "yolo_infer.h"




	



class DataSource : public SourceProcessor<std::vector<uint8_t>>
{


	public:
	
		 ImageInfo info ;
		 std::ifstream file;
		 int frame_size;
		 uint8_t frame_counter;
	         
	         
	         DataSource(CircularBuffer<std::vector<uint8_t>>* out):SourceProcessor(out) {
	         
	         	info =  parse_image_name("110057_1_ 405_1820_PLRAW_1");
	         	
	         	file.open("/home/act/Project Senna/Signal files/110057_1_ 405_1820_PLRAW.sgl",std::ios::binary);
	         	
	         	frame_size= 4*4000;
	         	
	         	
	         	
	         }
	         
	         std::vector<uint8_t> process() override
	         {
	         	 
	         	 std::this_thread::sleep_for(std::chrono::milliseconds(8));
	         	 
	    
	         	 std::vector<uint8_t> data_vector= read_one_frame(file, frame_size, DEAD_BAND_WIDTH);
	         	 
	         	 
	         	 return data_vector;
	         }

};

struct Artery_locator_data{

	cv::Mat output_image;
	Detection detection;
	std::vector<float> latest_RF_frame;
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
        "/home/act/Project Senna/models/yolo10n_fp16_b1.engine",
        /*input_w=*/512, /*input_h=*/512,
        /*input_name=*/"images",
        /*output_name=*/"output0"
    	) {
		    
	}
    	
    	//"/home/act/Project Senna/models/yolo_b1.engine",
       
    	
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
			    std::cout << "Inference FPS: " << count / elapsed << std::endl;
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









class ArteryLocator : public  ConnectProcessor<Artery_locator_data, Artery_locator_data>
{

private:

	 float  x_rescaler= 3700/512; // change this depending on model input size
	 SignalPreprocessing SP_handle;
	 std::vector<float> dist_history;
	 
	 

public: 
	ArteryLocator(CircularBuffer<Artery_locator_data>* in,CircularBuffer<Artery_locator_data>* out):ConnectProcessor(in,out){
	
	}
	
	
	Artery_locator_data process(const Artery_locator_data& inference_data) override
	{
	    //float x_rescaler = 3700.0f / 512.0f;
	   
	   
	   
	    
	    int x1_nbb = static_cast<int>(x_rescaler * inference_data.detection.box.x);
	    int x2_nbb = static_cast<int>(x_rescaler * (inference_data.detection.box.x + inference_data.detection.box.width));

	    if (x2_nbb <= x1_nbb)
		return inference_data;

	    int start = std::max(0, x1_nbb - 150);
	    int end   = std::min((int)inference_data.latest_RF_frame.size(), x2_nbb + 100);

	    if (start >= end)
		return inference_data;

	    std::vector<float> ROI_envelope(
		inference_data.latest_RF_frame.begin() + start,
		inference_data.latest_RF_frame.begin() + end
	    );

	    if (ROI_envelope.empty())
		return inference_data;

	    float* channels_lp[1];
	    channels_lp[0] = ROI_envelope.data();

	    SP_handle.lowpass_filter_cut.process(ROI_envelope.size(), channels_lp);
	    auto [p1, p2] = SP_handle.findTwoProminentPeaks(ROI_envelope,
                                          /*min_distance=*/ 300,
                                          /*height_threshold=*/ 0.0f);
                                          
           
                                          
                                          
            
	    
	    
	    
	    
	    return inference_data;
	}
	  
	  
};
		
		











class DisplayProcessor : public SinkProcessor<Artery_locator_data>
{
private:
    size_t frame_count = 0;
    std::chrono::steady_clock::time_point start_time;

public:
    DisplayProcessor(CircularBuffer<Artery_locator_data>* in)
        : SinkProcessor(in)
    {
        start_time = std::chrono::steady_clock::now();
    }

    void process(const Artery_locator_data& display_data) override
    {	
    
  
        if (!display_data.output_image.empty())
        {
            

          
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
       ArteryLocator artery_locator(&inference_to_locator,&locator_to_display);
       
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
