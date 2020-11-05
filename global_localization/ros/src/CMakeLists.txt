cmake_minimum_required(VERSION 2.8.3)
project(cloud_to_spi)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
find_package(catkin REQUIRED COMPONENTS  
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  image_transport
  cv_bridge
#   message_filters
)

find_package(OpenCV REQUIRED)

catkin_package()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ./
  ${catkin_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${libelas_INCLUDE_DIRS}
)


add_library(cloud_to_spi cloud_to_spi.cc spi_utility.cc)
target_link_libraries(cloud_to_spi ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} )


add_executable(node node.cc)
target_link_libraries(node ${catkin_LIBRARIES} cloud_to_spi)

