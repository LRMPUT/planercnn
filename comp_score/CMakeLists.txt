cmake_minimum_required(VERSION 2.8.3)
project(comp_score_py)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(pybind11)
pybind11_add_module(comp_score_py
        compScorePy.cpp)

target_link_libraries(comp_score_py
        PRIVATE)

