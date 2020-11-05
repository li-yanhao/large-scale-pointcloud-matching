#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "opencv2/core/core.hpp"

// using TransformStamped = geometry_msgs::TransformStamped;
using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;


template<typename T>
class Grid : public std::vector<T> {
public:
    Grid(std::size_t m, std::size_t n) : m_(m), n_(n)
    {
        this->resize(m_ * n_);
    }

    Grid(std::size_t m, std::size_t n, T val) : m_(m), n_(n)
    {
        this->resize(m_ * n_, val);
    }

    T& at(std::size_t x, std::size_t y)
    {
        return (*this)[x * n_ + y];
    }

private:
    std::size_t m_;
    std::size_t n_;
};


template<typename PointT>
Grid<std::vector<std::size_t>> make_grid_with_indices(const pcl::PointCloud<PointT>& cloud,
                                                      float scale,
                                                      float resolution)
{
    // make a grid
    int grid_size = int(scale / resolution);
    Grid<std::vector<std::size_t>> grid(grid_size, grid_size);
    // std::cout << "grid initialization success\n";

    for (std::size_t i = 0; i < cloud.size(); ++i) {
        std::size_t x_index = (cloud.points[i].x + scale / 2) / resolution;
        std::size_t y_index = (cloud.points[i].y + scale / 2) / resolution;
        if (0 <= x_index && x_index < grid_size && 0 <= y_index && y_index < grid_size) {
            grid.at(x_index, y_index).emplace_back(i);
        }
    }

    return grid;
}


void cloud_to_birdview_image(const PointCloud& in_cloud,
                             cv::Mat& out_image,
                             float min_z,
                             float max_z,
                             float scale,
                             float meters_per_pixel);


template<typename PointT>
void filter_cloud_by_z_value(pcl::PointCloud<PointT>& cloud, double min_z, double max_z)
{
    pcl::PointCloud<PointT> cloud_new;
    cloud_new.reserve(cloud.size());
    for (auto& point : cloud.points) {
        if (point.z > min_z && point.z < max_z) {
            cloud_new.push_back(point);
        }
    }
    cloud = cloud_new;
}