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
bool is_overlapped(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_A, const  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_B)
{
    const float search_radius = 0.5f;
    const float overlap_ratio = 0.9f;
    const auto& cloud_large = cloud_A->size() > cloud_B->size() ? cloud_A : cloud_B;
    const auto& cloud_small = cloud_A->size() <= cloud_B->size() ? cloud_A : cloud_B;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    tree->setInputCloud (cloud_large);
    int num_overlapped_pts = 0;
    for (const auto& point : cloud_small->points) {
        std::vector<int> k_indices;
        std::vector<float> k_sqr_distances;
        if (tree->radiusSearch(point, search_radius, k_indices, k_sqr_distances) > 0) {
            ++num_overlapped_pts;
        }
    }

    return float(num_overlapped_pts) / cloud_small->size() > overlap_ratio;

}

// input: two vectors of clusters
// output: vector of correspondences (cluster_A, cluster_B)
std::vector<std::pair<int, int>> make_correspondences(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters_A, 
                                                    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters_B)
{
    std::vector<std::pair<int, int>> correspondences;
    for (std::size_t i = 0; i < clusters_A.size(); ++i) {
        const auto& cluster_A = clusters_A[i];
        for (std::size_t j = 0; j < clusters_B.size(); ++j) {
            const auto& cluster_B = clusters_B[j];
            if (is_overlapped(cluster_A, cluster_B)) {
                correspondences.push_back(std::make_pair(i, j));
            }
        }
    }

    return correspondences;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> segment(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float max_distance, int min_size)
{
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    // tree->setInputCloud (cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    ec.setClusterTolerance (max_distance);
    ec.setMinClusterSize (min_size);
    // ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        if (it->indices.size() > 6000) continue;

        int r = rand() % 192 + 64;
        int g = rand() % 192 + 64;
        int b = rand() % 192 + 64;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
            pcl::PointXYZRGB point_rgb(r, g, b);
            point_rgb.x = cloud->points[*pit].x;
            point_rgb.y = cloud->points[*pit].y;
            point_rgb.z = cloud->points[*pit].z;
            cloud_cluster->points.push_back (point_rgb); //*
        }

        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        clusters.push_back(cloud_cluster);

        // std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
        // std::stringstream ss;
        // ss << "cloud_cluster_" << j << ".pcd";
        // writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster, false); //*
        // ++j;
    }

    return clusters;
}

void visualize_correspondences(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters_A,
                               const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters_B,
                               const std::vector<std::pair<int, int>>& correspondences)
{
    return;
}

// 0. downsample
// 1. euclidean cluster
// 2. correspondences
// 3. visualization

int main (int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "usage: ./segments_correspondences cloud_A.pcd cloud_B.pcd" << std::endl;
    }

    // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_A(new pcl::PointCloud<pcl::PointXYZ>), cloud_B(new pcl::PointCloud<pcl::PointXYZ>);

    reader.read (argv[1], *cloud_A);
    std::cout << "cloud_A has: " << cloud_A->points.size () << " data points." << std::endl;

    reader.read (argv[2], *cloud_B);
    std::cout << "cloud_B has: " << cloud_B->points.size () << " data points." << std::endl;

    const float voxel_size = 0.2f;
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setLeafSize (voxel_size, voxel_size, voxel_size);
    vg.setInputCloud (cloud_A);
    vg.filter (*cloud_A);

    vg.setInputCloud (cloud_B);
    vg.filter (*cloud_B);

    // *cloud_filtered = *cloud;
    std::cout << "cloud_A after filtering has: " << cloud_A->points.size ()  << " data points." << std::endl;
    std::cout << "cloud_B after filtering has: " << cloud_B->points.size ()  << " data points." << std::endl;

    const float cluster_distance = 0.8f;
    const int cluster_min_size = 100;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters_A = segment(cloud_A, cluster_distance, cluster_min_size);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters_B = segment(cloud_B, cluster_distance, cluster_min_size);
    
    std::cout << "cloud A has " << clusters_A.size() << " clusters." << std::endl;
    std::cout << "cloud B has " << clusters_B.size() << " clusters." << std::endl;
    
    std::vector<std::pair<int, int>> correspondences = make_correspondences(clusters_A, clusters_B);



    visualize_correspondences(clusters_A, clusters_B, correspondences);

    {
        int j = 0;
        for (const auto& cluster : clusters_A) {
            std::cout << "PointCloud representing the Cluster: " << cluster->points.size() << " data points." << std::endl;
            std::stringstream ss;
            ss << "cloud_cluster_A_" << j << ".pcd";
            writer.write<pcl::PointXYZRGB> (ss.str (), *cluster, false); //*
            j++;
        }

        j = 0;
        for (const auto& cluster : clusters_B) {
            std::cout << "PointCloud representing the Cluster: " << cluster->points.size() << " data points." << std::endl;
            std::stringstream ss;
            ss << "cloud_cluster_B_" << j << ".pcd";
            writer.write<pcl::PointXYZRGB> (ss.str (), *cluster, false); //*
            j++;
        }

    }

    std::cout << "correspondences size: " << correspondences.size() << std::endl;
    for (const auto& correspondence : correspondences) {
        std::cout << "(" << correspondence.first << "," << correspondence.second << ")\n"; 
    }

    return (0);
}