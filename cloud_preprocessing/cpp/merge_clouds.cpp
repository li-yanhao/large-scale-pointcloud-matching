#include <dirent.h>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/random_sample.h>

using PointT = pcl::PointXYZ;

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "usage: ./merge_clouds <pcd-cloud-dir>\n";

        return 0;
    }

    boost::filesystem::path dir_path(argv[1]);

    struct dirent *ptr;
    DIR *dir;
    dir = opendir(argv[1]);
    std::vector<std::string> filenames;
    std::cout << "文件列表: "<< std::endl;
    
    while((ptr=readdir(dir))!=NULL)
    {
        std::string file_name = ptr->d_name;
        //跳过'.'和'..'两个目录
        if (file_name.size() <= 4 || file_name.substr(file_name.size()-4, file_name.size()) != ".pcd")
            continue;
        filenames.push_back(ptr->d_name);
    }
    
    for (int i = 0; i < filenames.size(); ++i)
    {
        std::cout << filenames[i] << std::endl;
    }


    pcl::PCDReader reader;
    pcl::PointCloud<PointT>::Ptr out_cloud(new pcl::PointCloud<PointT>());
    pcl::RandomSample<pcl::PointXYZ> downSizeFilter;
    for (const auto& filename : filenames) {
        pcl::PointCloud<PointT>::Ptr in_cloud(new pcl::PointCloud<PointT>());
        boost::filesystem::path pcd_path(filename);
        reader.read((dir_path / pcd_path).c_str(), *in_cloud);
        // Downsample

        downSizeFilter.setSample(in_cloud->points.size() * 0.1);
        downSizeFilter.setInputCloud(in_cloud);
        // pcl::PointCloud<pcl::PointXYZ>::Ptr outputDS;
        
        // downSizeFilter.filter(*in_cloud);

        // Add to out_cloud
        *out_cloud += *in_cloud;

    }

    out_cloud->width = out_cloud->points.size();
    out_cloud->height = 1;
    out_cloud->is_dense = true;

    pcl::PCDWriter writer;
    std::string out_file_name("remove_ground.pcd");
    // out_file_name = out_file_name.substr(0, out_file_name.size()-4) + "_wo_ground.pcd";
    writer.write<PointT>((dir_path / out_file_name).c_str(), *out_cloud, false);
    
    
    return (0);
}