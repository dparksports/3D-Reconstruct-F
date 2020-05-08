cmake_minimum_required(VERSION 3.16)
project(fundamentalReconstruct)

set(CMAKE_CXX_STANDARD 20)

#add_executable(fundamentalReconstruct main.cpp)
find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)