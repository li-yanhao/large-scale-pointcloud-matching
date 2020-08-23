
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <string>

using PointT = pcl::PointXYZ;

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "usage: ./ground_removal pcd-file\n";

        return 0;
    }

    // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PointCloud<PointT>::Ptr in_cloud(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr out_cloud(new pcl::PointCloud<PointT>());

    reader.read(argv[1], *in_cloud);
    for (const auto& point : in_cloud->points) {
        if (point.z > 0) {
            out_cloud->points.emplace_back(point);
        }
    }

    out_cloud->width = out_cloud->points.size();
    out_cloud->height = 1;
    out_cloud->is_dense = true;

    pcl::PCDWriter writer;
    std::string out_file_name(argv[1]);
    out_file_name = out_file_name.substr(0, out_file_name.size()-4) + "_wo_ground.pcd";
    writer.write<PointT>(out_file_name, *out_cloud, false);
    
    return (0);
}