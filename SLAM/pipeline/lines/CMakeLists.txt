cmake_minimum_required(VERSION 3.16)
project(line_extract CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
# ELSED's own CMakeLists also builds a pybind11 module, which needs python-dev;
# we only want the library, so its sources are compiled in directly.
file(GLOB ELSED_SRC ${ELSED_DIR}/src/*.cpp)
list(FILTER ELSED_SRC EXCLUDE REGEX "PYAPI|main")
add_executable(line_extract line_extract.cc ${ELSED_SRC})
target_include_directories(line_extract PRIVATE ${ELSED_DIR}/src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(line_extract ${OpenCV_LIBS})

# 3-frame lifecycle test: links the SLAM's OWN MapLine + the same ELSED,
# with the LineExtractor matcher. Needs the ORB_SLAM3 tree for headers.
if(DEFINED ORB_DIR)
  find_package(Eigen3 REQUIRED)
  add_executable(line_track_test line_track_test.cc ${ELSED_SRC}
                 ${ORB_DIR}/src/MapLine.cc ${ORB_DIR}/src/LineExtractor.cc)
  target_include_directories(line_track_test PRIVATE
      ${ELSED_DIR}/src ${OpenCV_INCLUDE_DIRS} ${ORB_DIR}/include
      ${ORB_DIR} ${ORB_DIR}/Thirdparty/Sophus ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(line_track_test ${OpenCV_LIBS})
endif()
