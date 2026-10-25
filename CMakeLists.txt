cmake_minimum_required(VERSION 3.10)
project(TensorRT_Test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)


# 🔥 REQUIRED
enable_language(CUDA)

# ---- Find packages ----
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

file(GLOB DSP_SOURCES
    "/home/act/Vessel_Vision_live_track/project_version_v1r0/DSPFilters/source/*.cpp"
)

# ---- Executable ----
add_executable(main
    
   main_V1.cpp
   yolo_infer_older.cpp
   MotionTracking.cpp

    ${DSP_SOURCES}
)

target_compile_options(main PRIVATE -O3 -march=native -ffast-math)


# ---- Includes ----
target_include_directories(main PRIVATE
    /usr/include/x86_64-linux-gnu
    /usr/include/libusb-1.0
     "/home/act/Vessel_Vision_live_track/project_version_v1r0/DSPFilters/include"
     "/usr/include/python3.10"
     "/home/act/.local/lib/python3.10/site-packages/pybind11/include"
     ${Python3_INCLUDE_DIRS}
)



# ---- Linking ----
target_link_libraries(main PRIVATE
    CUDA::cudart
    nvinfer
    usb-1.0
    ${OpenCV_LIBS}
    Python3::Python
)
