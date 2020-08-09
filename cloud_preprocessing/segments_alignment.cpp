#include <vector>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>



// To tell if two clouds are overlapped with a minimum ratio
bool is_overlapped(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_A, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_B, float ratio)
{
    
}

// input: two vectors of clusters
// output: vector of correspondences (cluster_A, cluster_B)
std::vector<std::pair<int, int>> clusters_alignment(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters_A, 
                        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters_B)
{

}

int 
main (int argc, char** argv)
{
    // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    reader.read (argv[1], *cloud);
    std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (cloud);
    vg.setLeafSize (0.1f, 0.1f, 0.1f);
    vg.filter (*cloud_filtered);

    // *cloud_filtered = *cloud;
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*


    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    ec.setClusterTolerance (0.5);
    ec.setMinClusterSize (100);
    // ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);

    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        // if (it->indices.size() > 10000) continue;
        int r = rand() % 192 + 64;
        int g = rand() % 192 + 64;
        int b = rand() % 192 + 64;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
        pcl::PointXYZRGB point_rgb(r, g, b);
        point_rgb.x = cloud_filtered->points[*pit].x;
        point_rgb.y = cloud_filtered->points[*pit].y;
        point_rgb.z = cloud_filtered->points[*pit].z;
        cloud_cluster->points.push_back (point_rgb); //*
        }

        // if (cloud_cluster->points.size() > 1000) {
        //   tree->setInputCloud (cloud_cluster);
        //   ec.setClusterTolerance (0.2);
        //   ec.setMinClusterSize (100);
        //   // ec.setMaxClusterSize (25000);
        //   ec.setSearchMethod (tree);
        //   ec.setInputCloud (cloud_cluster);
        //   ec.extract (cluster_indices);
        // }

        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
        std::stringstream ss;
        ss << "cloud_cluster_" << j << ".pcd";
        writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster, false); //*
        j++;
    }

    return (0);
}