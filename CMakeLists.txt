cmake_minimum_required(VERSION 3.16.3)
project(CNN)

find_package(OpenCV REQUIRED)
add_executable(CNN src/mat_mul.cpp "src/neon_replace.hpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})