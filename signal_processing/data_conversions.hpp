#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/utils/utils.hpp"




std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delimiter)) {
        tokens.push_back(item);
    }

    return tokens;
}


struct ImageInfo {
    std::string subject;
    std::string trial;
    std::string frame_start;
    std::string frame_end;
    int lid;
};



ImageInfo parse_image_name(const std::string& img_name) {
    auto x = split(img_name, '_');

    ImageInfo info;
    info.subject     = x[0];
    info.trial       = x[1];
    info.frame_start = x[2];
    info.frame_end   = x[3];
    info.lid         = std::stoi(x[5]);

    return info;
}


std::vector<uint8_t> read_one_frame(
    std::ifstream& file,
    int frame_size,
    int dead_band_width
) {
    int dead_bytes = (dead_band_width * frame_size) / 40;
    int data_bytes = frame_size - dead_bytes;

    // Skip dead band
    file.ignore(dead_bytes);

    // Allocate buffer for valid data
    std::vector<uint8_t> frame_data(data_bytes);

    // Read valid data
    file.read(reinterpret_cast<char*>(frame_data.data()), data_bytes);

    // Check if full frame was read
    if (file.gcount() < data_bytes) {
        return {};  // EOF or incomplete frame
    }

    return frame_data;
}


std::vector<float> IEEE_to_floatingPoint_recorded_data(const std::vector<uint8_t>& binary_data) {
    
    std::vector<float> floating_values;

    
    /*if (binary_data.size() % 4 != 0) {
        return NULL;  // return empty on error
    }*/

    for (size_t i = 0; i < binary_data.size(); i += 4) {
        uint8_t bytes[4];

        // Reverse byte order (Python's chunk[::-1])
        bytes[0] = binary_data[i + 3];
        bytes[1] = binary_data[i + 2];
        bytes[2] = binary_data[i + 1];
        bytes[3] = binary_data[i + 0];

        float value;
        memcpy(&value, bytes, sizeof(float));

        floating_values.push_back(value);
    }

   return floating_values;
}
	
