// #include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/pcd_io.h>

using PointT = pcl::PointXYZ;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: ./uniform_sampling <xxx.pcd> \n";

        return 0;
    }

    const std::string cloud_filename = argv[1];
    pcl::PointCloud<PointT>::Ptr cloud_in(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(cloud_filename, *cloud_in);

    pcl::RandomSample<PointT> rs;
    rs.setInputCloud(cloud_in);
    rs.setSample(512);

    pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>());
    rs.filter(*cloud_out);
    pcl::io::savePCDFile("out.pcd", *cloud_out, true);

    return 0;
}