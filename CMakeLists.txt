cmake_minimum_required(VERSION 3.20)
project(cpp_video_inference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_local_opencv_dir "${CMAKE_CURRENT_SOURCE_DIR}/../toolchains/opencv-4.13/opencv/build")
if(NOT OpenCV_DIR AND EXISTS "${_local_opencv_dir}/OpenCVConfig.cmake")
  set(OpenCV_DIR "${_local_opencv_dir}")
endif()
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation directory")
if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT installation directory" FORCE)
endif()
if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
  message(FATAL_ERROR "Set -DTENSORRT_ROOT to a TensorRT 10 installation containing include/NvInfer.h")
endif()
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${TENSORRT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(TENSORRT_LIBRARY NAMES nvinfer_10 nvinfer HINTS "${TENSORRT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(video_infer
  src/main.cpp
  src/preprocess.cpp
  src/trt_engine.cpp
  src/postprocess.cpp
  src/video_pipeline.cpp
  src/multi_stream_pipeline.cpp)
target_compile_features(video_infer PRIVATE cxx_std_17)
target_include_directories(video_infer PRIVATE ${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(video_infer PRIVATE ${OpenCV_LIBS} ${TENSORRT_LIBRARY} CUDA::cudart)

string(REPLACE "." "" _opencv_version_compact "${OpenCV_VERSION}")
set(_opencv_world_dll "${OpenCV_DIR}/x64/vc16/bin/opencv_world${_opencv_version_compact}.dll")
if(WIN32 AND EXISTS "${_opencv_world_dll}")
  add_custom_command(TARGET video_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${_opencv_world_dll}"
      "$<TARGET_FILE_DIR:video_infer>")
endif()

# OpenCV videoio loads FFmpeg as a runtime plugin on Windows. Keep it next to
# the executable so AVI/MP4 metadata and decoding work without PATH changes.
if(WIN32)
  file(GLOB _opencv_ffmpeg_plugins
    "${OpenCV_DIR}/x64/vc16/bin/opencv_videoio_ffmpeg*_64.dll")
  foreach(_opencv_ffmpeg_plugin IN LISTS _opencv_ffmpeg_plugins)
    add_custom_command(TARGET video_infer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_opencv_ffmpeg_plugin}" "$<TARGET_FILE_DIR:video_infer>")
  endforeach()
endif()

if(WIN32 AND EXISTS "${TENSORRT_ROOT}/lib/nvinfer_10.dll")
  add_custom_command(TARGET video_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${TENSORRT_ROOT}/lib/nvinfer_10.dll"
      "$<TARGET_FILE_DIR:video_infer>")
endif()

if(MSVC)
  target_compile_options(video_infer PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(video_infer PRIVATE -Wall -Wextra -Wpedantic)
endif()
