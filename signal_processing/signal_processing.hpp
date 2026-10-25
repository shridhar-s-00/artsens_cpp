
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/signal_processing/data_conversions.hpp"
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/utils/utils.hpp"


#define lowcut 1e6
#define highcut 20e6
#define fs 80e6
#define filt_order 4

int HILBERT_LEN = 51;   


#define low_cutoff 300000

#define OVERLAP 5
#define NO_OF_FRAMES 20
#define DEAD_BAND_WIDTH 3.5


class SignalPreprocessing
{
	public:
		Dsp::SimpleFilter <Dsp::Butterworth::BandPass<4>,1>bandpass_filter;
		Dsp::SimpleFilter <Dsp::Butterworth::LowPass<4>,1>lowpass_filter_cut;
		std::vector<float> hilbert_coefficient;
		std::vector<float> hilbert_state;

	SignalPreprocessing(){
	
		 bandpass_filter.setup(filt_order,fs,(lowcut+highcut)/2,highcut-lowcut);
		 lowpass_filter_cut.setup(filt_order,fs,low_cutoff);
		 hilbert_coefficient = design_hilbert_fir(HILBERT_LEN);
		 hilbert_state.assign(HILBERT_LEN, 0.0f);
	}

		
	std::vector<float> design_hilbert_fir(int N) {
	    std::vector<float> h(N);
	    int M = (N - 1) / 2;

	    for (int n = 0; n < N; n++) {
		if (n == M) {
		    h[n] = 0.0;
		} else {
		    int k = n - M;
		    h[n] = 2.0 / (M_PI * k);
		}

		// Zero out even taps (Hilbert property)
		if ((n % 2) == 0)
		    h[n] = 0.0;
	    }
	    return h;
	 }


	 float envelope(float real, float imag) {
	    return std::sqrt(real * real + imag * imag);
	 }



	 float hilbert_sample(
	    float x,
	    const std::vector<float>& h,
	    std::vector<float>& state
	  ) {
	    // shift state
	    for (int i = state.size() - 1; i > 0; i--)
		state[i] = state[i - 1];
	    state[0] = x;

	    float y = 0.0;
	    for (size_t i = 0; i < h.size(); i++)
		y += h[i] * state[i];

	    return y;
	   }
	   
	   
	   void filter_RF_frame(const std::vector<uint8_t>& raw_frame, std::vector<float>& hilbert_filtered_frame){
	
		hilbert_filtered_frame.clear();
		
		auto floats = IEEE_to_floatingPoint_recorded_data(raw_frame);
		
		float* channels[1];
		channels[0] = floats.data();

		bandpass_filter.process(floats.size(), channels);
		float* data = floats.data();
		size_t N = floats.size();
		
	     


		for (size_t i = 0; i < N; i++) {
			float x = abs(data[i]);

			float imag = hilbert_sample(x, hilbert_coefficient, hilbert_state);
			float env  = std::sqrt(x * x + imag * imag);
				  
			hilbert_filtered_frame.push_back(env);
		}
		
	     }
	     
	     
	     
	     
	   void frame_buffer_to_image(std::vector<std::vector<float>> frame_buffer,cv::Mat& grayscale_image){
	   
	   	 std::vector<float> envelope_signal_flat;
	         
	         for(const auto& row : frame_buffer){
	            envelope_signal_flat.insert(envelope_signal_flat.end(),row.begin(),row.end());
	         }
	         
	         std::vector<float> envelope_signal_normalized(envelope_signal_flat.size());
	         
	         auto min_it= std::min_element(envelope_signal_flat.begin(),envelope_signal_flat.end());
	         auto max_it= std::max_element(envelope_signal_flat.begin(),envelope_signal_flat.end());
	         
	         float min_val= *min_it;
	         float max_val= *max_it;
	         
	         for( int i=0; i< envelope_signal_flat.size();i++){
	            
	        	envelope_signal_normalized[i]= (envelope_signal_flat[i] - min_val)/ (max_val - min_val) ;
	        }
	        
	    	cv::Mat data_image(20, 3700, CV_32F, envelope_signal_normalized.data());

		  
		    
		cv::Mat img;
		cv::resize(data_image, img, cv::Size(1024, 128));
		
		cv::Mat img8;
   		img.convertTo(img8, CV_8U, 255.0);
		   		
			
	   	
	        cv::cvtColor(img8, grayscale_image, cv::COLOR_GRAY2RGB);
	      
	     }
	     
		  	
	std::pair<int, int> findTwoProminentPeaks(
	    const std::vector<float>& data,
	    int   min_distance,
	    float height_threshold)
		{
		    if (data.size() < 3)
			throw std::invalid_argument("Data must have at least 3 elements.");
		    if (min_distance < 1)
			throw std::invalid_argument("min_distance must be >= 1.");

		    // --- 1. Collect all peaks (local maxima above threshold) ----------------
		    // A peak at index i satisfies: data[i] > data[i-1] && data[i] > data[i+1]
		    std::vector<int> peaks;

		    for (int i = 1; i < (int)data.size() - 1; ++i)
			if (data[i] > data[i - 1] && data[i] > data[i + 1]
				&& data[i] >= height_threshold)
			    peaks.push_back(i);

		    if (peaks.size() < 2)
			//throw std::runtime_error("Fewer than 2 peaks found above the threshold.");
			return {0,0};
		    // --- 2. Sort peaks by height descending ---------------------------------
		    std::sort(peaks.begin(), peaks.end(), [&](int a, int b) {
			return data[a] > data[b];
		    });

		    // --- 3. Greedily pick the tallest pair separated by min_distance --------
		    int first  = -1;
		    int second = -1;

		    for (int i = 0; i < (int)peaks.size() && second == -1; ++i) {
			if (first == -1) {
			    first = peaks[i];
			    continue;
			}
			if (std::abs(peaks[i] - first) >= min_distance)
			    second = peaks[i];
		    }

		    if (second == -1)
			//throw std::runtime_error(
			//    "No two peaks satisfy the minimum distance constraint.");
			return {0,0};
		    // Return in left-to-right order
		    if (first > second) std::swap(first, second);
		    return {first, second};
		}
	
	
};
	


