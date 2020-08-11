
#include <pcl/io/pcd_io.h>
// #include <pcl/filters/approximate_voxel_grid.h>


template<typename PointT>
pcl::PointXYZ calculate_center(const pcl::PointCloud<PointT>& cloud)
{
    double x(0), y(0), z(0);
    for (const auto& point : cloud.points) {
        x += point.x;
        y += point.y;
        z += point.z;
    }
    x /= cloud.size();
    y /= cloud.size();
    z /= cloud.size();

   return pcl::PointXYZ(x, y, z);
}

template<typename PointT>
pcl::PointXYZ calculate_center_by_boundary(const pcl::PointCloud<PointT>& cloud)
{
    float x_max(std::numeric_limits<float>::min());
    float y_max(std::numeric_limits<float>::min());
    float z_max(std::numeric_limits<float>::min());
    float x_min(std::numeric_limits<float>::max());
    float y_min(std::numeric_limits<float>::max());
    float z_min(std::numeric_limits<float>::max());
    
    for (const auto& point : cloud.points) {
        x_max = std::max(point.x, x_max);
        y_max = std::max(point.y, y_max);
        z_max = std::max(point.z, z_max);
        x_min = std::min(point.x, x_min);
        y_min = std::min(point.x, y_min);
        z_min = std::min(point.x, z_min);
    }

    return pcl::PointXYZ((x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2);
}


template<typename PointT>
float calculate_distance(const PointT& P, const PointT& Q)
{
    float dx = P.x - Q.x;
    float dy = P.y - Q.y;
    float dz = P.z - Q.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

template<typename PointT>
bool are_submaps_close(const pcl::PointCloud<PointT>& cloud_A, const pcl::PointCloud<PointT>& cloud_B, float threshold)
{
    
    pcl::PointXYZ center_A = calculate_center(cloud_A);
    pcl::PointXYZ center_B = calculate_center(cloud_B);

    return calculate_distance(center_A, center_B) < threshold;
}


int main(int argc, char** argv)
{   
    std::vector<pcl::PointXYZ> submap_centers;

    std::stringstream file_ss;
    const int id_max = 356;
    const std::string file_prefix = "/media/admini/My_data/0629/lidar_calib/20200617160311_lyu_shengda_calib.bag_segment_"; 
    for (int i = 0; i < id_max; ++i) {
        file_ss << file_prefix << std::to_string(i) << ".pcd";

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::io::loadPCDFile(file_ss.str(), *cloud_in);

        // submap_centers.push_back(calculate_center_by_boundary(*cloud_in));
        submap_centers.push_back(calculate_center(*cloud_in));
        
        file_ss.str("");
    }

    std::vector<std::pair<int, int>> submap_correspondences;
    const float threshold_distance = 50.f;
    for (std::size_t i = 0; i < submap_centers.size(); ++i) {
        const auto& center_i = submap_centers[i];
        for (std::size_t j = 0; j < submap_centers.size(); ++j) {
            const auto& center_j = submap_centers[j];
            float distance_xy = std::sqrt(std::pow(center_i.x - center_j.x, 2) + std::pow(center_i.y - center_j.y, 2));

            if (distance_xy < threshold_distance) {
                submap_correspondences.push_back(std::make_pair(i, j));
            }
        }
    }

    for (const auto& correspondence : submap_correspondences) {
        std::cout << "(" << correspondence.first << "," << correspondence.second << ")\n";
    }

    // TODO: save the submap correspondences in a file (e.g. txt file)

    return 0;
}