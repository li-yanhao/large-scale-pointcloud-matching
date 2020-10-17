cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(region_growing_rgb_segmentation)

SET(OpenCV_DIR /usr/local/share/OpenCV)
SET(OPENCV_ENABLE_NONFREE ON)

find_package(PCL 1.7 REQUIRED)
# find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -lstdc++")

message(STATUS "PCL_INCLUDE_DIRS:" ${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARY_DIRS:" ${PCL_LIBRARY_DIRS})
message(STATUS "PCL_DEFINITIONS:" ${PCL_DEFINITIONS})
message(STATUS "OpenCV_INCLUDE_DIRS:" ${OpenCV_INCLUDE_DIRS})
# message(STATUS "HDF5_INCLUDE_DIRS:" ${HDF5_INCLUDE_DIRS})

include_directories(
  ./
  /usr/include/hdf5/serial
)

# include_directories(${HDF5_INCLUDE_DIRS})
# link_directories( ${HDF5_LIBRARY_DIRS} )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories( ${OpenCV_LIBRARY_DIRS} )

# include_directories(/usr/include/hdf5/serial)
# include_directories(${HDF5_CXX_INCLUDE_DIRS})

# link_directories(${HDF5_LIBRARY_DIRS})
# link_directories(${HDF5_CXX_LIBRARY_DIRS})

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
target_link_libraries (submap_correspondences
  ${PCL_LIBRARIES} 
  # ${HDF5_CXX_LIBRARIES} 
  # ${HDF5_LIBRARIES}
  boost_system
  mpi_cxx
  glog gflags 
  pthread
  # /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
  # /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.so
  # /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so
)
# install(TARGETS submap_correspondences DESTINATION bin)

add_executable (bird_view_poi bird_view_poi.cpp)
target_link_libraries (bird_view_poi ${PCL_LIBRARIES})

add_executable (ground_filtering ground_filtering.cpp)
target_link_libraries (ground_filtering ${PCL_LIBRARIES} )

add_executable (ground_removal ground_removal.cpp)
target_link_libraries (ground_removal ${PCL_LIBRARIES} )


add_executable (json_tutorial json_tutorial.cpp)
# target_link_libraries (json_tutorial ${PCL_LIBRARIES})

add_executable (merge_clouds merge_clouds.cpp)
target_link_libraries (merge_clouds ${PCL_LIBRARIES} )

add_executable(segmatch_to_pcds segmatch_to_pcds.cpp)
target_link_libraries (segmatch_to_pcds ${PCL_LIBRARIES} )

add_executable(bird_view_image bird_view_image.cpp)
target_link_libraries (bird_view_image ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(uniform_sample uniform_sample.cpp)
target_link_libraries(uniform_sample ${PCL_LIBRARIES})

add_executable(bird_view_dataset bird_view_dataset.cpp)
target_link_libraries(bird_view_dataset ${PCL_LIBRARIES} glog gflags mpi_cxx)

