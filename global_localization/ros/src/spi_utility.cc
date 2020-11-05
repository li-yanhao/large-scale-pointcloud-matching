#include "spi_utility.h"


void cloud_to_birdview_image(const PointCloud& in_cloud,
                             cv::Mat& out_image,
                             float min_z,
                             float max_z,
                             float scale,
                             float meters_per_pixel)
{
    auto grid = make_grid_with_indices<PointT>(in_cloud, scale, meters_per_pixel);
    int grid_size = int(scale / meters_per_pixel);
    cv::Mat image = cv::Mat::zeros(grid_size, grid_size, CV_32FC1);
    cv::Mat image_normalized = cv::Mat::zeros(grid_size, grid_size, CV_32FC1);
    float interval = max_z - min_z;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            // maximum height of the points bin
            const auto& indices = grid.at(i, j);

            float max_height = min_z;
            for (auto index : indices) {
                max_height = in_cloud.points[index].z > max_height ? in_cloud.points[index].z : max_height;
            }
            max_height = std::min(max_height, max_z);

            image.at<float>(i, j) = (max_height - min_z) / interval;
        }
    }

    // cv::cvtColor(image, image_normalized, );
    // image.convertTo(out_image, CV_32FC1);
    image.copyTo(out_image);
}