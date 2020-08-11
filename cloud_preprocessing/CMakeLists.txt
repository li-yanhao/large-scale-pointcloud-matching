cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(region_growing_rgb_segmentation)

find_package(PCL 1.5 REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_BUILD_TYPE "Release")

include_directories(
  ./
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# add_library(descriptors descriptors.h)
# target_link_libraries (descriptors ${PCL_LIBRARIES})

add_executable (region_growing_rgb_segmentation region_growing_rgb_segmentation.cpp)
target_link_libraries (region_growing_rgb_segmentation ${PCL_LIBRARIES})

add_executable (euclidean_cluster_extraction euclidean_cluster_extraction.cpp)
target_link_libraries (euclidean_cluster_extraction ${PCL_LIBRARIES})


add_executable (conditional_euclidean_clustering descriptors.hpp conditional_euclidean_clustering.cpp)
target_link_libraries (conditional_euclidean_clustering ${PCL_LIBRARIES})

add_executable (mincut_segmentation mincut_segmentation.cpp)
target_link_libraries (mincut_segmentation ${PCL_LIBRARIES})

add_executable (segment_correspondences segment_correspondences.cpp)
target_link_libraries (segment_correspondences ${PCL_LIBRARIES})

add_executable (submap_correspondences submap_correspondences.cpp)
target_link_libraries (submap_correspondences ${PCL_LIBRARIES})

add_executable (bird_view_poi bird_view_poi.cpp)
target_link_libraries (bird_view_poi ${PCL_LIBRARIES})

add_executable (ground_filtering ground_filtering.cpp)
target_link_libraries (ground_filtering ${PCL_LIBRARIES})

