#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/CompressedImage.h>

#include "cloud_to_spi.h"
#include "spi_utility.h"

namespace {
std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do {
        pos = str.find(delim, prev);
        if (pos == std::string::npos)
            pos = str.length();
        std::string token = str.substr(prev, pos - prev);
        if (!token.empty())
            tokens.push_back(token);
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());

    return tokens;
}
}  // namespace

CloudToSpi::CloudToSpi(const ros::NodeHandle& node_handler) : nh_(node_handler)
{
    tf_buffer_.reset(new tf2_ros::Buffer(ros::Duration(3600)));

    nh_.param<std::string>("world_frame", world_frame_, "world");
    nh_.param<std::string>("lidar_frame", lidar_frame_, "velodyne");
    nh_.param<int>("num_accumulated_clouds", num_accumulated_clouds_, 600);
    nh_.param<int>("inter_cloud_diff", inter_cloud_diff_, 200);
    nh_.param<float>("max_z", max_z_, 10.f);
    nh_.param<float>("min_z", min_z_, 0.f);
    nh_.param<float>("scale", scale_, 100.f);
    nh_.param<float>("meters_per_pixel", meters_per_pixel_, 0.1f);
    nh_.param<double>("x_offset", x_offset_, -2861852);
    nh_.param<double>("y_offset", y_offset_, 4651685);
    nh_.param<double>("z_offset", z_offset_, 3283262);

    nh_.param<std::string>("lidar_trajectory_file",
                           lidar_trajectory_file_,
                           "/media/li/lavie/dataset/20201023114733_Paul_Locate.bag_lidar_trajectory.txt");

    ROS_ASSERT(num_accumulated_clouds_ >= inter_cloud_diff_);

    ptcloud_subscriber_ = nh_.subscribe(kPointCloudTopic, 100, &CloudToSpi::pointcloud_callback, this);
    spi_publisher_ = nh_.advertise<sensor_msgs::CompressedImage>(kSpiTopic, 2);

    ROS_INFO_STREAM("Start loading tf_buffer ...");
    load_tf_buffer();
    ROS_INFO_STREAM("Loading tf_buffer finished.");
    // play_spi();
}

void CloudToSpi::pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& ptcloud_msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*ptcloud_msg, *point_cloud);
    try {
        auto stamped_transform = tf_buffer_->lookupTransform(world_frame_, lidar_frame_, ptcloud_msg->header.stamp);
        // LOG(INFO) << stamped_transform;
        // Eigen::Isometry3d lidar_to_world = Eigen::Isometry3d::Identity();
        Eigen::Quaterniond quaternion(stamped_transform.transform.rotation.w,
                                      stamped_transform.transform.rotation.x,
                                      stamped_transform.transform.rotation.y,
                                      stamped_transform.transform.rotation.z);
        Eigen::Matrix4d lidar_to_world = Eigen::Matrix4d::Identity();
        lidar_to_world.block(0, 0, 3, 3) = quaternion.matrix();
        lidar_to_world.block(0, 3, 3, 1) << stamped_transform.transform.translation.x,
                stamped_transform.transform.translation.y, stamped_transform.transform.translation.z;
        // ROS_INFO_STREAM("lidar_to_world: \n" << lidar_to_world);

        // 1. transform cloud to world frame
        // filter_cloud_by_z_value(pcl_point_cloud, FLAGS_filer_min_z, FLAGS_filer_max_z);
        filter_cloud_by_z_value(*point_cloud, min_z_, max_z_);
        pcl::transformPointCloud(*point_cloud, *point_cloud, lidar_to_world);

        cloud_queue_.push_back(point_cloud);
        poses_lidar_to_world_.push_back(lidar_to_world);

        // ROS_INFO_STREAM("point_cloud size: " << point_cloud->size());

        if (cloud_queue_.size() >= num_accumulated_clouds_ + inter_cloud_diff_) {
            ROS_ASSERT(cloud_queue_.size() == poses_lidar_to_world_.size());
            int ref_index = num_accumulated_clouds_ / 2;

            Eigen::Matrix4d lidar_ref_to_world = poses_lidar_to_world_[ref_index];
            pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_accumulated(new pcl::PointCloud<pcl::PointXYZ>());
            for (int i = 0; i < num_accumulated_clouds_; ++i) {
                *point_cloud_accumulated += *(cloud_queue_[i]);
            }

            pcl::transformPointCloud(*point_cloud_accumulated,
                                     *point_cloud_accumulated,
                                     lidar_ref_to_world.inverse().cast<float>());

            static int submap_id = 0;
            cv::Mat spi_image;
            // spi_image =
            //         cv::imread("/media/li/lavie/dataset/birdview_dataset/08/submap_" + std::to_string(submap_id++) +
            //         ".png");
            // ROS_INFO_STREAM("point_cloud_accumulated len: " << point_cloud_accumulated->size());
            cloud_to_birdview_image(*point_cloud_accumulated, spi_image, min_z_, max_z_, scale_, meters_per_pixel_);
            spi_image.convertTo(spi_image, CV_8U, 255);
            cv::imshow("spi", spi_image);
            cv::waitKey(1);

            sensor_msgs::CompressedImage img_msg;
            img_msg.header = ptcloud_msg->header;
            img_msg.format = "png";
            cv::imencode(".png", spi_image, img_msg.data);

            spi_publisher_.publish(img_msg);
            ROS_INFO_STREAM("Published spi image.");

            for (int i = 0; i < inter_cloud_diff_; ++i) {
                cloud_queue_.pop_front();
                poses_lidar_to_world_.pop_front();
            }
        }
    } catch (const tf2::TransformException& ex) {
        ROS_INFO(ex.what());
    }
}

void CloudToSpi::load_tf_buffer()
{
    std::ifstream lidar_trajectory_file(lidar_trajectory_file_);
    std::string line;
    int seq = 0;
    while (std::getline(lidar_trajectory_file, line)) {
        auto splitted = split(line, ",");
        geometry_msgs::TransformStamped transform_stamped_msg;
        transform_stamped_msg.header.frame_id = world_frame_;
        transform_stamped_msg.child_frame_id = lidar_frame_;
        transform_stamped_msg.header.seq = seq++;
        // transform_stamped_msg.header.stamp = ros::Time::fromSec(atof(splitted[0].c_str()));
        // transform_stamped_msg.transform.rotation.w = pose_stamped_msg.pose.orientation;
        transform_stamped_msg.header.stamp.fromSec(atof(splitted[0].c_str()));
        transform_stamped_msg.transform.translation.x = atof(splitted[1].c_str()) - x_offset_;
        transform_stamped_msg.transform.translation.y = atof(splitted[2].c_str()) - y_offset_;
        transform_stamped_msg.transform.translation.z = atof(splitted[3].c_str()) - z_offset_;
        transform_stamped_msg.transform.rotation.w = atof(splitted[4].c_str());
        transform_stamped_msg.transform.rotation.x = atof(splitted[5].c_str());
        transform_stamped_msg.transform.rotation.y = atof(splitted[6].c_str());
        transform_stamped_msg.transform.rotation.z = atof(splitted[7].c_str());
        tf_buffer_->setTransform(transform_stamped_msg, "li", false);
    }
}


void CloudToSpi::play_spi()
{
    static int submap_id = 0;
    ros::Rate rate(2.5);
    while (ros::ok()) {
        cv::Mat spi_image;
        const std::string spi_filename =
                "/media/li/lavie/dataset/birdview_dataset/08/submap_" + std::to_string(submap_id++) + ".png";
        spi_image = cv::imread(spi_filename, cv::IMREAD_GRAYSCALE);
        ROS_INFO_STREAM(spi_image.channels());
        ROS_INFO_STREAM(spi_image.size);
        cv::imshow("spi raw", spi_image);
        cv::waitKey(1);
        // cloud_to_birdview_image(*point_cloud, spi_image);

        sensor_msgs::CompressedImage img_msg;
        img_msg.format = "png";
        cv::imencode(".png", spi_image, img_msg.data);

        spi_publisher_.publish(img_msg);
        ROS_INFO_STREAM("Played " << spi_filename);
        rate.sleep();
    }
}