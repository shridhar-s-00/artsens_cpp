





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
	         	
	         	file.open("/home/act/Project Senna/Signal files/999_13_2440_4180_PLRAW.sgl",std::ios::binary);
	         	
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
    
    
    bool validate_bbox_stability()
	{
	    if (bbox_center_history.size() <
		REQUIRED_BBOX_HISTORY)
	    {
		return false;
	    }

	    // ------------------------------------
	    // compute mean center + width
	    // ------------------------------------

	    float mean_center = 0;
	    float mean_width  = 0;

	    for (int i = 0;
		 i < bbox_center_history.size();
		 i++)
	    {
		mean_center += bbox_center_history[i];
		mean_width  += bbox_width_history[i];
	    }

	    mean_center /=
		bbox_center_history.size();

	    mean_width /=
		bbox_width_history.size();

	    // ------------------------------------
	    // check deviations
	    // ------------------------------------

	    for (int i = 0;
		 i < bbox_center_history.size();
		 i++)
	    {
		int center_diff =
		    std::abs(
		        bbox_center_history[i] -
		        mean_center
		    );

		int width_diff =
		    std::abs(
		        bbox_width_history[i] -
		        mean_width
		    );

		if (center_diff >
		    MAX_CENTER_DEVIATION)
		{
		    return false;
		}

		if (width_diff >
		    MAX_WIDTH_DEVIATION)
		{
		    return false;
		}
	    }

	    return true;
	}
    int x1_nbb;
    int x2_nbb;
    int one_time_use=0;
    int center_prev=-1;
    int center_curr;
    std::vector<float> RF_data;
    
    
	std::vector<int> bbox_center_history;
	std::vector<int> bbox_width_history;

	const int REQUIRED_BBOX_HISTORY = 5;

	const int MAX_CENTER_DEVIATION = 80;
	const int MAX_WIDTH_DEVIATION  = 100;
	int validate_bounding_box=0;
	
	
    std::pair<int, int> find_wall_peaks(
    const std::vector<float>& signal,
    int distance_tolerance,
    float min_peak_height = 0.0f
)
{
    // ----------------------------------------
    // safety
    // ----------------------------------------

    if (signal.size() < 10)
        return {-1, -1};

    int N = signal.size();
    int mid = N / 2;

    // ----------------------------------------
    // collect LEFT peaks
    // ----------------------------------------

    std::vector<std::pair<int, float>> left_peaks;

    for (int i = 1; i < mid - 1; i++)
    {
        if (signal[i] > signal[i - 1] &&
            signal[i] > signal[i + 1] &&
            signal[i] > min_peak_height)
        {
            left_peaks.push_back(
                {i, signal[i]}
            );
        }
    }

    // ----------------------------------------
    // collect RIGHT peaks
    // ----------------------------------------

    std::vector<std::pair<int, float>> right_peaks;

    for (int i = mid + 1; i < N - 1; i++)
    {
        if (signal[i] > signal[i - 1] &&
            signal[i] > signal[i + 1] &&
            signal[i] > min_peak_height)
        {
            right_peaks.push_back(
                {i, signal[i]}
            );
        }
    }

    // ----------------------------------------
    // no peaks
    // ----------------------------------------

    if (left_peaks.empty() ||
        right_peaks.empty())
    {
        return {-1, -1};
    }

    // ----------------------------------------
    // sort by height
    // tallest first
    // ----------------------------------------

    std::sort(
        left_peaks.begin(),
        left_peaks.end(),
        [](const auto& a, const auto& b)
        {
            return a.second > b.second;
        }
    );

    std::sort(
        right_peaks.begin(),
        right_peaks.end(),
        [](const auto& a, const auto& b)
        {
            return a.second > b.second;
        }
    );

    // ----------------------------------------
    // try combinations
    // ----------------------------------------

    for (const auto& left_peak : left_peaks)
    {
        for (const auto& right_peak : right_peaks)
        {
            int left_idx =
                left_peak.first;

            int right_idx =
                right_peak.first;

            // ----------------------------
            // ensure sufficient separation
            // ----------------------------

            if (std::abs(
                    right_idx - left_idx
                ) < distance_tolerance)
            {
                continue;
            }

            // valid pair
            return {
                left_idx,
                right_idx
            };
        }
    }

    // ----------------------------------------
    // nothing valid
    // ----------------------------------------

    return {-1, -1};
}

std::vector<std::pair<int,int>>
peak_history;

std::vector<int>
center_history;

std::vector<std::vector<float>>
aligned_roi_history;

const int REQUIRED_TEMPORAL_FRAMES = 20;

std::vector<float> shift_signal(
    const std::vector<float>& signal,
    int shift
)
{
    std::vector<float>
        shifted(signal.size(), 0);

    for (int i = 0;
         i < signal.size();
         i++)
    {
        int j = i - shift;

        if (j >= 0 &&
            j < signal.size())
        {
            shifted[i] = signal[j];
        }
    }

    return shifted;
}
public:
    ArteryLocator(CircularBuffer<Artery_locator_data>* in,
                  CircularBuffer<Artery_locator_data>* out)
        : ConnectProcessor(in, out) {}

   Artery_locator_data process(const Artery_locator_data& inference_data) override
    {
    
    
    	//check the bounding boxes first

	if( current_mode == Mode::DETECT)
		{
		    
		    x1_nbb = static_cast<int>(
			x_rescaler * inference_data.detection.box.x
		    );

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

		    // ----------------------------------------
		    // compute bbox properties
		    // ----------------------------------------

		    int bbox_center =
			(x1_nbb + x2_nbb) / 2;

		    int bbox_width =
			x2_nbb - x1_nbb;

		    // ----------------------------------------
		    // store bbox history
		    // ----------------------------------------

		    if(bbox_width > 300)
		    {
			bbox_center_history.push_back(
			    bbox_center
			);

			bbox_width_history.push_back(
			    bbox_width
			);
		    }

		    // ----------------------------------------
		    // maintain fixed history
		    // ----------------------------------------

		    if (bbox_center_history.size() >
			REQUIRED_BBOX_HISTORY)
		    {
			bbox_center_history.erase(
			    bbox_center_history.begin()
			);

			bbox_width_history.erase(
			    bbox_width_history.begin()
			);
		    }

		    // ----------------------------------------
		    // validate bbox stability
		    // ----------------------------------------

		    validate_bounding_box =
			validate_bbox_stability();

		    if (!validate_bounding_box)
		    {
			return inference_data;
		    }

		    // ----------------------------------------
		    // ROI extraction
		    // ----------------------------------------

		    int start =
			std::max(0, x1_nbb);

		    int end =
			std::min(
			    (int)inference_data.latest_RF_frame.size(),
			    x2_nbb
			);

		    if (start >= end)
			return inference_data;

		    RF_data =
			inference_data.latest_RF_frame;

		    float* channels[1] =
		    {
			RF_data.data()
		    };

		    SP_handle.lowpass_filter_cut.process(
			RF_data.size(),
			channels
		    );

		    std::vector<float> ROI(
			RF_data.begin() + start,
			RF_data.begin() + end
		    );

		    if (ROI.empty())
			return inference_data;

		    // ----------------------------------------
		    // PEAK DETECTION
		    // ----------------------------------------

		    auto [peak_l, peak_r] =
			find_wall_peaks(
			    ROI,
			    300,
			    0.1
			);

		    if (peak_l == -1 ||
			peak_r == -1)
		    {
			return inference_data;
		    }

		    std::cout
			<< "peak l->"
			<< peak_l
			<< " peak r->"
			<< peak_r
			<< std::endl;

		    // ----------------------------------------
		    // COMPUTE CENTER
		    // ----------------------------------------

		    int artery_center =
			(peak_l + peak_r) / 2;

		    // ----------------------------------------
		    // USE FIRST CENTER AS REFERENCE
		    // ----------------------------------------

		    if (center_history.empty())
		    {
			reference_center =
			    artery_center;
		    }

		    // ----------------------------------------
		    // COMPUTE SHIFT
		    // ----------------------------------------

		    int shift = 0;
			//artery_center -
			//reference_center;

		    // ----------------------------------------
		    // ALIGN SIGNAL SAFELY
		    // ----------------------------------------

		    std::vector<float>
			aligned_roi(
			    ROI.size(),
			    0
			);

		    for (int i = 0;
			 i < ROI.size();
			 i++)
		    {
			int j = i - shift;

			if (j >= 0 &&
			    j < ROI.size())
			{
			    aligned_roi[i] =
				ROI[j];
			}
		    }

		    // ----------------------------------------
		    // STORE TEMPORAL DATA
		    // ----------------------------------------

		    peak_history.push_back(
			{peak_l, peak_r}
		    );

		    center_history.push_back(
			artery_center
		    );

		    aligned_roi_history.push_back(
			aligned_roi
		    );

		    // ----------------------------------------
		    // MAINTAIN FIXED TEMPORAL BUFFER
		    // ----------------------------------------

		    if (peak_history.size() >
			REQUIRED_TEMPORAL_FRAMES)
		    {
			peak_history.erase(
			    peak_history.begin()
			);

			center_history.erase(
			    center_history.begin()
			);

			aligned_roi_history.erase(
			    aligned_roi_history.begin()
			);
		    }

		    // ----------------------------------------
		    // WAIT UNTIL ENOUGH FRAMES
		    // ----------------------------------------

		    if (aligned_roi_history.size() <
			REQUIRED_TEMPORAL_FRAMES)
		    {
			std::cout
			    << "Collecting temporal frames -> "
			    << aligned_roi_history.size()
			    << std::endl;

			return inference_data;
		    }

		    // ----------------------------------------
		    // DISPLAY ALL ALIGNED FRAMES
		    // ----------------------------------------

		    static int playback_index = 0;

			// safety
			if (aligned_roi_history.empty())
			{
			    return inference_data;
			}

			// wrap index
			playback_index =
			    playback_index %
			    aligned_roi_history.size();

			// current frame
			const auto& signal =
			    aligned_roi_history[playback_index];

			// current peaks
			int peak_l1 =
			    peak_history[playback_index].first;

			int peak_r1 =
			    peak_history[playback_index].second;

			// ----------------------------------------
			// CREATE PLOT
			// ----------------------------------------

			cv::Mat plot =
			    plot_rf_with_walls(
				signal,
				peak_l1,
				peak_r1
			    );

			// ----------------------------------------
			// DISPLAY INFO
			// ----------------------------------------

			cv::putText(
			    plot,
			    "Frame: " +
			    std::to_string(playback_index),
			    cv::Point(20, 30),
			    cv::FONT_HERSHEY_SIMPLEX,
			    0.7,
			    cv::Scalar(255,255,255),
			    2
			);

			// ----------------------------------------
			// SHOW SINGLE WINDOW
			// ----------------------------------------

			cv::imshow(
			    "Aligned RF Playback",
			    plot
			);

			// small delay for playback feel
			cv::waitKey(30);

			// next frame
			playback_index++;
		}
	
	
	return inference_data;
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
           
            /*dist_canvas.setTo(cv::Scalar(0, 0, 0));

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
            }*/

            //cv::imshow("Live Signal", dist_canvas);
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
