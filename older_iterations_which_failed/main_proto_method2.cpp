





#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/signal_processing/signal_processing.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/threadlib/processor.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/threadlib/threadqueue.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/utils/utils.hpp"

#include "yolo_infer_older.h"



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
	         	
	         	file.open("/home/act/Project Senna/Signal files/motion_artifact/motion artifact from amit/1112_1_16_7_2718_PLRAW.sgl",std::ios::binary);
	         	
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


class ArteryLocator : public ConnectProcessor<Artery_locator_data, Artery_locator_data>
{
private:
    // ---------------- CONFIG ----------------
    const int width_distention = 800;
    float x_rescaler = 3700.0f / 800.0f;

    SignalPreprocessing SP_handle;

    std::vector<float> distention_buffer;
    int max_history = width_distention / 4;

    int reference_center = -1;

    // -------- Detection Params --------
    const int MAX_PEAKS = 8;
    const int PEAK_MIN_DISTANCE = 300;
    const float PEAK_MIN_HEIGHT = 0.1f;

    const int WINDOW_HALF_WIDTH = 32;

    const int MIN_DISTENTION = 200;
    const int MAX_DISTENTION = 1500;

    const int MIN_NW_POSITION = 200;
    const int MAX_NW_POSITION = 4000;

    // -------- Tracking State --------
    enum class Mode { DETECT, TRACK };
    Mode current_mode = Mode::DETECT;

    int tracked_nw = -1;
    int tracked_fw = -1;

    std::vector<int> nw_history;
    std::vector<int> fw_history;

    int tracking_fail_count = 0;
    const int MAX_TRACK_FAIL = 10;

    std::vector<float> previous_roi;

    Artery_locator_data display_data_packet;

    // ---------------- PEAK DETECTION ----------------
    std::vector<std::pair<int, float>> find_peaks(const std::vector<float>& signal)
    {
        std::vector<std::pair<int, float>> peaks;

        for (int i = 1; i < (int)signal.size() - 1; i++)
        {
            if (signal[i] > signal[i - 1] &&
                signal[i] > signal[i + 1] &&
                signal[i] > PEAK_MIN_HEIGHT)
            {
                if (!peaks.empty() && (i - peaks.back().first) < PEAK_MIN_DISTANCE)
                    continue;

                peaks.emplace_back(i, signal[i]);
            }
        }

        std::sort(peaks.begin(), peaks.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });

        if ((int)peaks.size() > MAX_PEAKS)
            peaks.resize(MAX_PEAKS);

        std::sort(peaks.begin(), peaks.end(),
                  [](auto& a, auto& b) { return a.first < b.first; });

        return peaks;
    }

    // ---------------- LAG ----------------
    int compute_lag(const std::vector<float>& prev,
                    const std::vector<float>& curr,
                    int center)
    {
        int best_lag = 0;
        float best_score = -1e9;

        for (int lag = -WINDOW_HALF_WIDTH; lag <= WINDOW_HALF_WIDTH; lag++)
        {
            float score = 0.0f;

            for (int i = -WINDOW_HALF_WIDTH; i <= WINDOW_HALF_WIDTH; i++)
            {
                int idx1 = center + i;
                int idx2 = center + i + lag;

                if (idx1 < 0 || idx1 >= prev.size() ||
                    idx2 < 0 || idx2 >= curr.size())
                    continue;

                score += prev[idx1] * curr[idx2];
            }

            if (score > best_score)
            {
                best_score = score;
                best_lag = lag;
            }
        }

        return best_lag;
    }

    // ---------------- DETECT ----------------
 
   
	std::pair<int, int> detect_walls(const std::vector<float>& current,
		                         const std::vector<float>& previous)
	{
	    int N = current.size();
	    if (N < 50) return {-1, -1};

	    int mid = N / 2;

	    // -------- Step 1: split into sectors --------
	    int left_start = 0;
	    int left_end   = mid;

	    int right_start = mid;
	    int right_end   = N;

	    // -------- Step 2: find strongest peak in each sector --------
	    int nw = -1;
	    int fw = -1;

	    float max_left = -1e9;
	    for (int i = left_start + 1; i < left_end - 1; i++)
	    {
		if (current[i] > current[i-1] &&
		    current[i] > current[i+1] &&
		    current[i] > PEAK_MIN_HEIGHT)
		{
		    if (current[i] > max_left)
		    {
		        max_left = current[i];
		        nw = i;
		    }
		}
	    }

	    float max_right = -1e9;
	    for (int i = right_start + 1; i < right_end - 1; i++)
	    {
		if (current[i] > current[i-1] &&
		    current[i] > current[i+1] &&
		    current[i] > PEAK_MIN_HEIGHT)
		{
		    if (current[i] > max_right)
		    {
		        max_right = current[i];
		        fw = i;
		    }
		}
	    }
	   
	    if (nw == -1 || fw == -1)
		return {-1, -1};

	    // -------- Step 3: distance check --------
	    int dist = fw - nw;

	    if (!(dist > MIN_DISTENTION && dist < MAX_DISTENTION))
		return {-1, -1};

	    // -------- Step 4: local lag computation --------
	    int half_region = dist / 6;
	    half_region = std::max(30, std::min(half_region, 100));

	    auto get_window = [&](const std::vector<float>& sig, int center)
	    {
		int s = std::max(0, center - half_region);
		int e = std::min((int)sig.size(), center + half_region);

		return std::vector<float>(sig.begin() + s, sig.begin() + e);
	    };

	    auto prev_nw = get_window(previous, nw);
	    auto curr_nw = get_window(current, nw);

	    auto prev_fw = get_window(previous, fw);
	    auto curr_fw = get_window(current, fw);

	    int lag_nw = compute_lag(prev_nw, curr_nw, prev_nw.size()/2);
	    int lag_fw = compute_lag(prev_fw, curr_fw, prev_fw.size()/2);

	    if (lag_nw == 0 || lag_fw == 0)
		return {-1, -1};

	    // -------- Step 5: opposite motion check --------
	    if (!((lag_nw > 0) ^ (lag_fw > 0)))
	    	//check with shifted
	        
		return {-1, -1};

	    return {nw, fw};
	}
    // ---------------- TRACK ----------------
    bool track_walls(const std::vector<float>& current,
                     const std::vector<float>& previous,
                     int& nw, int& fw)
    {
        int half_region = (fw - nw) / 6;
        half_region = std::max(30, std::min(half_region, 100));

        int t0 = compute_lag(previous, current, nw);
        int t1 = compute_lag(previous, current, fw);

        nw += t0;
        fw += t1;

        if (fw <= nw)
            return false;

        nw_history.push_back(nw);
        fw_history.push_back(fw);

        if (nw_history.size() > 25)
        {
            nw_history.erase(nw_history.begin());
            fw_history.erase(fw_history.begin());
        }

        if (nw_history.size() >= 10)
        {
            float mean_nw = 0, mean_fw = 0;

            for (int i = 0; i < nw_history.size(); i++)
            {
                mean_nw += nw_history[i];
                mean_fw += fw_history[i];
            }

            mean_nw /= nw_history.size();
            mean_fw /= fw_history.size();

            float num = 0, d1 = 0, d2 = 0;

            for (int i = 0; i < nw_history.size(); i++)
            {
                float a = nw_history[i] - mean_nw;
                float b = fw_history[i] - mean_fw;

                num += a * b;
                d1 += a * a;
                d2 += b * b;
            }

            float R = num / (std::sqrt(d1 * d2) + 1e-6);

            if (R > -0.1)
                return false;
        }

        return true;
    }

    // ---------------- PLOT ----------------
    cv::Mat plot_rf_with_walls(const std::vector<float>& signal, int p1, int p2)
    {
        int width = 800, height = 400;
        cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        float min_val = *std::min_element(signal.begin(), signal.end());
        float max_val = *std::max_element(signal.begin(), signal.end());
        float range = std::max(1.0f, max_val - min_val);

        for (int i = 1; i < signal.size(); i++)
        {
            int x1 = (i - 1) * width / signal.size();
            int x2 = i * width / signal.size();

            int y1 = height - (signal[i - 1] - min_val) / range * height;
            int y2 = height - (signal[i] - min_val) / range * height;

            cv::line(img, {x1, y1}, {x2, y2}, {0, 255, 0}, 1);
        }

        int x1 = p1 * width / signal.size();
        int x2 = p2 * width / signal.size();

        cv::line(img, {x1, 0}, {x1, height}, {0, 0, 255}, 2);
        cv::line(img, {x2, 0}, {x2, height}, {255, 0, 0}, 2);

        return img;
    }
    int x1_nbb;
    int x2_nbb;
    int one_time_use=0;
    int center_prev=-1;
    int center_curr;
    std::vector<float> RF_data;
      int track_loss_count=0;

public:
    ArteryLocator(CircularBuffer<Artery_locator_data>* in,
                  CircularBuffer<Artery_locator_data>* out)
        : ConnectProcessor(in, out) {}

    Artery_locator_data process(const Artery_locator_data& inference_data) override
    {
        	
           if(current_mode == Mode::DETECT){
           	x1_nbb = static_cast<int>(x_rescaler * inference_data.detection.box.x) -200 ;
	   	x2_nbb = static_cast<int>(x_rescaler * (inference_data.detection.box.x + inference_data.detection.box.width)) ;
	   }
	   //x1_nbb=0;
	   //x2_nbb= 3700;
           //std::cout<<x1_nbb<<"  "<<x2_nbb<<std::endl;
           

	    if (x2_nbb <= x1_nbb)
		return inference_data;
            //x1_nbb= x1_nbb-50;
            //x2_nbb= x2_nbb+100;
           // std::cout<<x1_nbb<<"  "<<x2_nbb<<std::endl;
	    int start = max(0, x1_nbb );
	    int end = min((int)inference_data.latest_RF_frame.size(), x2_nbb );
	    //start=start-50;

	    if (start >= end)
		return inference_data;
	   RF_data= inference_data.latest_RF_frame;
	   float* channels[1] = {RF_data.data()};
           SP_handle.lowpass_filter_cut.process(RF_data.size(), channels);
	    
	    std::vector<float> ROI(
		RF_data.begin() + start ,
		RF_data.begin() + end  
	    );
	    
	    int bb1= static_cast<int>(x_rescaler * inference_data.detection.box.x)  ;
	    int bb2= static_cast<int>(x_rescaler * (inference_data.detection.box.x + inference_data.detection.box.width)) ;
            
            //std::vector<float> ROI= inference_data.latest_RF_frame;
	    if (ROI.empty())
		return inference_data;

		
        
        
        
        
        
 
        

        if (previous_roi.empty())
        {
            previous_roi = ROI;
            return inference_data;
        }

        // -------- STATE MACHINE --------
        if (current_mode == Mode::DETECT)
        {
            auto [nw, fw] = detect_walls(ROI, previous_roi);
            
           
            
	    
            if (nw != -1 && fw != -1)
            {
                tracked_nw =  nw ;//+ start; 
		tracked_fw = fw ;//+ start ;

                nw_history.clear();
                fw_history.clear();

                current_mode = Mode::TRACK;
                //std::cout << ">>> DETECT → TRACK\n";
                center_prev= (tracked_nw+ tracked_fw)/2;
                //std::cout<<"center-"<<center_prev<<std::endl;
                
            }
           // else{
            
                //retry with center approach
               // __NOP();
           // }    
                
             std::cout << ">>> DETECT Failed\n";
        }
        else
        {
            bool ok = track_walls(ROI, previous_roi, tracked_nw, tracked_fw);
            center_curr= (tracked_nw+ tracked_fw)/2;
            //std::cout<<"center tracking-"<<center_curr<<std::endl;
            int shift= center_curr-center_prev;
            
	    center_curr= center_prev;
            if (!ok)
            {
            	std::vector<float> ROI_shift(
		RF_data.begin() + start + shift ,
		RF_data.begin() + end  + shift
	    	);
	    	
	    	
	    	
	    	int shifted_nw= tracked_nw + shift;
	    	int shifted_fw= tracked_fw + shift;
	    	
            	bool ok2 = track_walls(ROI_shift, previous_roi, shifted_nw, shifted_fw);
                if(!ok2){
                  tracking_fail_count++;
                  //std::cout<<"track count->"<<tracking_fail_count<<std::endl;
                }
                else{
                  std::cout << ">>> saved by shifting \n";
                     
                   ROI= ROI_shift;
                
                }
		//tracking_fail_count++;
                if (tracking_fail_count > MAX_TRACK_FAIL)
                {
                    current_mode = Mode::DETECT;
                    tracked_nw = -1;
                    tracked_fw = -1;
                    std::cout << ">>> TRACK LOST → DETECT\n";
                    std::cout<<"track loss count->"<<++track_loss_count<<std::endl;
                }
            
            }
            else
            {
                tracking_fail_count = 0;
            }
        }

        previous_roi = ROI;

        //if (tracked_nw == -1 || tracked_fw == -1)
            
        //    return inference_data;

        // -------- VISUALIZATION --------
       cv::Mat plot = plot_rf_with_walls(ROI, tracked_nw, tracked_fw);
        cv::imshow("RF + Walls", plot);
        cv::waitKey(1);

        // -------- DISTENTION --------
        float dist = (float)(tracked_fw - tracked_nw);

        if (dist > 0)
            distention_buffer.push_back(dist);

        if (distention_buffer.size() > max_history)
            distention_buffer.erase(distention_buffer.begin());

        display_data_packet.dist_history = distention_buffer;
        display_data_packet.output_image = inference_data.output_image;
        
       
        	

        return display_data_packet;
    }
    
    /*
    
    Artery_locator_data process(const Artery_locator_data& inference_data) override
	{
	    // -------------------------------------------------
	    // ROI UPDATE ONLY DURING DETECTION
	    // -------------------------------------------------

	    if (current_mode == Mode::DETECT)
	    {
		x1_nbb = static_cast<int>(
		    x_rescaler * inference_data.detection.box.x
		) - 200;

		x2_nbb = static_cast<int>(
		    x_rescaler *
		    (inference_data.detection.box.x +
		     inference_data.detection.box.width)
		);

		x1_nbb = std::max(0, x1_nbb);

		x2_nbb = std::min(
		    (int)inference_data.latest_RF_frame.size(),
		    x2_nbb
		);
	    }

	    // -------------------------------------------------
	    // ROI EXTRACTION
	    // -------------------------------------------------

	    int start = std::max(0, x1_nbb);
	    int end   = std::min(
		(int)inference_data.latest_RF_frame.size(),
		x2_nbb
	    );

	    if (start >= end)
		return inference_data;

	    RF_data= inference_data.latest_RF_frame;

	    // -------------------------------------------------
	    // LOWPASS FILTER
	    // -------------------------------------------------

	    float* channels[1] = { RF_data.data() };
	    SP_handle.lowpass_filter_cut.process(RF_data.size(), channels);
	    
	    
	    std::vector<float> ROI(
		RF_data.begin() + start,
		RF_data.begin() + end
	    );

	    if (ROI.empty())
		return inference_data;

	    // -------------------------------------------------
	    // FIRST FRAME
	    // -------------------------------------------------

	    if (previous_roi.empty())
	    {
		previous_roi = ROI;
		return inference_data;
	    }

	    // -------------------------------------------------
	    // STATE MACHINE
	    // -------------------------------------------------

	    if (current_mode == Mode::DETECT)
	    {
		auto [nw, fw] = detect_walls(ROI, previous_roi);

		if (nw != -1 && fw != -1)
		{
		    tracked_nw = nw;
		    tracked_fw = fw;

		    // sanity
		    int dist = tracked_fw - tracked_nw;

		    if (dist > MIN_DISTENTION &&
		        dist < MAX_DISTENTION)
		    {
		        nw_history.clear();
		        fw_history.clear();

		        tracking_fail_count = 0;
			//if (center_prev==-1){
		        center_prev =
		            (tracked_nw + tracked_fw) / 2;//}

		        current_mode = Mode::TRACK;

		        std::cout
		            << ">>> DETECT -> TRACK\n";

		        std::cout
		            << "NW: " << tracked_nw
		            << " FW: " << tracked_fw
		            << " CENTER: " << center_prev
		            << std::endl;
		    }
		}
	    }
	    else
	    {
		// -------------------------------------------------
		// NORMAL TRACKING
		// -------------------------------------------------

		bool ok = track_walls(
		    ROI,
		    previous_roi,
		    tracked_nw,
		    tracked_fw
		);

		// -------------------------------------------------
		// SUCCESS
		// -------------------------------------------------

		if (ok)
		{
		    int dist =
		        tracked_fw - tracked_nw;

		    // hard sanity
		    if (dist < MIN_DISTENTION ||
		        dist > MAX_DISTENTION)
		    {
		        ok = false;
		    }
		}

		// -------------------------------------------------
		// TRACK FAILED
		// -------------------------------------------------

		if (!ok)
		{
		    tracking_fail_count++;

		    std::cout
		        << "TRACK FAIL COUNT -> "
		        << tracking_fail_count
		        << std::endl;

		    // ---------------------------------------------
		    // MOTION ARTIFACT RECOVERY
		    // ---------------------------------------------

		    int center_curr =
		        (tracked_nw + tracked_fw) / 2;

		    int shift =
		        center_curr - center_prev;

		    // try opposite direction too if needed
		    int shifted_start =
		        std::max(0, start - shift);

		    int shifted_end =
		        std::min(
		            (int)RF_data.size(),
		            end - shift
		        );

		    bool recovered = false;

		    if (shifted_start < shifted_end)
		    {
		        std::vector<float> ROI_shift(
		            RF_data.begin()
		                + shifted_start,

		            RF_data.begin()
		                + shifted_end
		        );

		        if (!ROI_shift.empty())
		        {
		            recovered = track_walls(
		                ROI_shift,
		                previous_roi,
		                tracked_nw,
		                tracked_fw
		            );
		        }
		        
		        ROI= ROI_shift;
		    }

		    // ---------------------------------------------
		    // RECOVERY SUCCESS
		    // ---------------------------------------------

		    if (recovered)
		    {
		        tracking_fail_count = 0;

		        center_prev =
		            (tracked_nw + tracked_fw) / 2;

		        std::cout
		            << ">>> RECOVERED VIA SHIFT\n";
		    }

		    // ---------------------------------------------
		    // TOTAL TRACK FAILURE
		    // ---------------------------------------------

		    else if (
		        tracking_fail_count >
		        MAX_TRACK_FAIL
		    )
		    {
		        current_mode = Mode::DETECT;

		        tracked_nw = -1;
		        tracked_fw = -1;

		        std::cout
		            << ">>> TRACK LOST -> DETECT\n";
		    }
		}

		// -------------------------------------------------
		// TRACK SUCCESS
		// -------------------------------------------------

		else
		{
		    tracking_fail_count = 0;

		    center_prev =
		        (tracked_nw + tracked_fw) / 2;
		}
	    }

	    // -------------------------------------------------
	    // UPDATE PREVIOUS FRAME
	    // -------------------------------------------------

	    previous_roi = ROI;

	    // -------------------------------------------------
	    // VISUALIZATION SAFETY
	    // -------------------------------------------------

	    if (tracked_nw >= 0 &&
		tracked_fw < ROI.size() &&
		tracked_fw > tracked_nw)
	    {
		cv::Mat plot =
		    plot_rf_with_walls(
		        ROI,
		        tracked_nw,
		        tracked_fw
		    );

		cv::imshow("RF + Walls", plot);
		cv::waitKey(1);
	    }

	    // -------------------------------------------------
	    // DISTENTION
	    // -------------------------------------------------

	    float dist =
		(float)(tracked_fw - tracked_nw);

	    if (dist > 0)
	    {
		distention_buffer.push_back(dist);
	    }

	    if (distention_buffer.size() >
		max_history)
	    {
		distention_buffer.erase(
		    distention_buffer.begin()
		);
	    }

	    // -------------------------------------------------
	    // OUTPUT
	    // -------------------------------------------------

	    display_data_packet.dist_history =
		distention_buffer;

	    display_data_packet.output_image =
		inference_data.output_image;

	    return display_data_packet;
	}*/
};
/*
class ArteryLocator : public ConnectProcessor<Artery_locator_data, Artery_locator_data>
{
private:
    float x_rescaler = 3700.0f / 800.0f;
    SignalPreprocessing SP_handle;

    std::vector<float> distention_buffer;
    int max_history = 800 / 3;

    // 🔷 STATE
    bool tracking = false;
    int NW = -1, FW = -1;

    std::vector<float> prev_frame;

    Artery_locator_data display_data_packet;

    // --------------------------------------------------
    int compute_shift(const std::vector<float>& curr,
                      const std::vector<float>& prev,
                      int center,
                      int half_window = 60)
    {
        if (prev.empty()) return 0;

        float best_corr = -1e9;
        int best_shift = 0;

        for (int s = -10; s <= 10; s++)
        {
            float corr = 0;

            for (int i = -half_window; i < half_window; i++)
            {
                int idx1 = center + i;
                int idx2 = idx1 + s;

                if (idx1 < 0 || idx1 >= curr.size()) continue;
                if (idx2 < 0 || idx2 >= prev.size()) continue;

                corr += curr[idx1] * prev[idx2];
            }

            if (corr > best_corr)
            {
                best_corr = corr;
                best_shift = s;
            }
        }
        return best_shift;
    }

public:
    ArteryLocator(CircularBuffer<Artery_locator_data>* in,
                  CircularBuffer<Artery_locator_data>* out)
        : ConnectProcessor(in, out) {}

    Artery_locator_data process(const Artery_locator_data& inference_data) override
    {
        // ---------------- ROI ----------------
        int x1 = x_rescaler * inference_data.detection.box.x   ;
        int x2 = x_rescaler * (inference_data.detection.box.x + inference_data.detection.box.width) ;

        if (x2 <= x1) return inference_data;

        int start = std::max(0, x1 - 150);
        int end   = std::min((int)inference_data.latest_RF_frame.size(), x2 + 100);

        if (start >= end) return inference_data;

        std::vector<float> ROI(
            inference_data.latest_RF_frame.begin() + start,
            inference_data.latest_RF_frame.begin() + end
        );

        if (ROI.empty()) return inference_data;

        // 🔷 Filter
        float* ch[1]; ch[0] = ROI.data();
        SP_handle.lowpass_filter_cut.process(ROI.size(), ch);

        // 🔷 Bootstrap
        if (prev_frame.empty())
        {
            prev_frame = ROI;
            return inference_data;
        }

        // ==================================================
        // 🔷 DETECTION PHASE
        // ==================================================
        if (!tracking)
        {
            auto [p1, p2] = SP_handle.findTwoProminentPeaks(ROI, 300, 0.0f);

            if (p1 != 0 || p2 != 0)
            {
                int s1 = compute_shift(ROI, prev_frame, p1);
                int s2 = compute_shift(ROI, prev_frame, p2);

                // 🔥 ONLY CONDITION: opposite motion
                if ((s1 != 0 && s2 != 0) && ((s1 > 0) ^ (s2 > 0)))
                {
                    NW = p1;
                    FW = p2;
                    tracking = true;

                    std::cout << "LOCKED → NW=" << NW << " FW=" << FW << std::endl;
                    std::cout << "LOCKED → NW=" << NW << " FW=" << FW << std::endl;
                    std::cout << "LOCKED → NW=" << NW << " FW=" << FW << std::endl;
                    std::cout << "LOCKED → NW=" << NW << " FW=" << FW << std::endl;
                    std::cout << "LOCKED → NW=" << NW << " FW=" << FW << std::endl;
                }
            }
        }

        // ==================================================
        // 🔷 TRACKING PHASE
        // ==================================================
        else
        {
            int half_region = std::max(30, std::min((FW - NW) / 6, 100));

            int sNW = compute_shift(ROI, prev_frame, NW, half_region);
            int sFW = compute_shift(ROI, prev_frame, FW, half_region);

            NW += sNW;
            FW += sFW;

            float dist = FW - NW;

            std::cout << "TRACK → NW=" << NW
                      << " FW=" << FW
                      << " dist=" << dist << std::endl;

            if (dist > 0)
                distention_buffer.push_back(dist);

            if (distention_buffer.size() > max_history)
                distention_buffer.erase(distention_buffer.begin());
        }

        prev_frame = ROI;

        display_data_packet.dist_history = distention_buffer;
        display_data_packet.output_image = inference_data.output_image;

        return display_data_packet;
    }
};*/

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
