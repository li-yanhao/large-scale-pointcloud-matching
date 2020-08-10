#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <algorithm>

int main (int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "usage: ./mincut_segmentation your-point-cloud.pcd" << std::endl;
    }
    pcl::PointCloud <pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZ>);
    if ( pcl::io::loadPCDFile <pcl::PointXYZ> (argv[1], *cloud) == -1 )
    {
        std::cout << "Cloud reading failed." << std::endl;
        return (-1);
    }

    pcl::PointXYZ center(0,0,0);
    for (const auto& point : cloud->points) {
        center.x += point.x;
        center.y += point.y;
        center.z += point.z;
    }

    center.x /= cloud->points.size();
    center.y /= cloud->points.size();
    center.z /= cloud->points.size();

    for (auto& point : cloud->points) {
        point.x -= center.x;
        point.y -= center.y;
        point.z -= center.z;
    }

    std::sort(cloud->points.begin(), cloud->points.end(), [](pcl::PointXYZ p1, pcl::PointXYZ p2){
        return p1.z < p2.z;
    });

    pcl::IndicesPtr indices (new std::vector <int>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-1000, 1000.0);
    pass.filter (*indices);

    pcl::MinCutSegmentation<pcl::PointXYZ> seg;
    seg.setInputCloud (cloud);
    seg.setIndices (indices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr foreground_points(new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointXYZ point = cloud->points.back();
    // point.x = 0;
    // point.y = 0;
    // point.z = 5;
    foreground_points->points.push_back(point);
    seg.setForegroundPoints (foreground_points);

    seg.setSigma (0.25);
    seg.setRadius (3.0433856);
    seg.setNumberOfNeighbours (14);
    seg.setSourceWeight (0.8);

    std::vector <pcl::PointIndices> clusters;
    seg.extract (clusters);

    std::cout << "Maximum flow is " << seg.getMaxFlow () << std::endl;

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = seg.getColoredCloud ();
    pcl::visualization::CloudViewer viewer ("Cluster viewer");
    viewer.showCloud(colored_cloud);
    while (!viewer.wasStopped ())
    {
    }

    return (0);
}