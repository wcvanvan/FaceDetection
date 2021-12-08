cmake_minimum_required(VERSION 3.16.3)
project(CNN)

find_package(OpenCV REQUIRED)
add_executable(CNN src/simd_replace.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})