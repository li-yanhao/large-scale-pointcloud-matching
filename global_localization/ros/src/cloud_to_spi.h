#pragma once
#include <deque>
#include <string>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "tf2_ros/buffer.h"
// #include <tf2_msgs/TFMessage.h>

const std::string kPointCloudTopic = "/velodyne_points";
const std::string kSpiTopic = "/spi_image/compressed";

class CloudToSpi {
public:
    CloudToSpi(const ros::NodeHandle& node_handler);
    void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void load_tf_buffer();
    void play_spi();

private:
    ros::NodeHandle nh_;
    ros::Subscriber ptcloud_subscriber_;
    ros::Publisher spi_publisher_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    std::string world_frame_;
    std::string lidar_frame_;

    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_queue_;
    std::deque<Eigen::Matrix4d> poses_lidar_to_world_;
    int num_accumulated_clouds_;
    int inter_cloud_diff_;
    float max_z_;
    float min_z_;
    float scale_;
    float meters_per_pixel_;

    double x_offset_;
    double y_offset_;
    double z_offset_;
    std::string lidar_trajectory_file_;
};
