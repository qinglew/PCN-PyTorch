cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(sample)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (mesh_sampling mesh_sampling.cpp)
target_link_libraries (mesh_sampling ${PCL_LIBRARIES})
