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


pcl::PointXYZ calculate_center(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
{
    double x(0), y(0), z(0);
    for (const auto& point : cloud->points) {
        x += point.x;
        y += point.y;
        z += point.z;
    }
    x /= cloud->size();
    y /= cloud->size();
    z /= cloud->size();

   return pcl::PointXYZ(x, y, z);
}

template<typename PointT>
float calculate_distance(const PointT& P, const PointT& Q)
{
    float dx = P.x - Q.x;
    float dy = P.y - Q.y;
    float dz = P.z - Q.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// To tell if two clouds are overlapped with a minimum ratio
bool is_overlapped(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_A, 
                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_B,
                   float overlap_ratio)
{
    const float search_radius = 0.5f;
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
    const float overlap_ratio = 0.6f;
    const float max_center_distance(5.f);
    std::vector<pcl::PointXYZ> centers_A;
    std::vector<pcl::PointXYZ> centers_B;
    for (const auto& cluster : clusters_A) {
        centers_A.emplace_back(calculate_center(cluster));
    }
    for (const auto& cluster : clusters_B) {
        centers_B.emplace_back(calculate_center(cluster));
    }

    std::vector<std::pair<int, int>> correspondences;
    for (std::size_t i = 0; i < clusters_A.size(); ++i) {
        const auto& cluster_A = clusters_A[i];
        for (std::size_t j = 0; j < clusters_B.size(); ++j) {
            if (calculate_distance(centers_A[i], centers_B[j]) > max_center_distance)
                continue;
            const auto& cluster_B = clusters_B[j];
            if (is_overlapped(cluster_A, cluster_B, overlap_ratio)) {
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

void draw_connected_line(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cluster_A,
                         const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cluster_B,
                         float supplement_z,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr& out_cloud)
{
    pcl::PointXYZ center_A(0, 0, 0);
    pcl::PointXYZ center_B(0, 0, 0);
    for (const auto& point : cluster_A->points) {
        center_A.x += point.x;
        center_A.y += point.y;
        center_A.z += point.z;
    }
    center_A.x /= cluster_A->size();
    center_A.y /= cluster_A->size();
    center_A.z /= cluster_A->size();

    for (const auto& point : cluster_B->points) {
        center_B.x += point.x;
        center_B.y += point.y;
        center_B.z += point.z;
    }
    center_B.x /= cluster_B->size();
    center_B.y /= cluster_B->size();
    center_B.z /= cluster_B->size();
    center_B.z += supplement_z;

    float interval = 0.2f;
    pcl::PointXYZ direction(center_B.x - center_A.x, center_B.y - center_A.y, center_B.z - center_A.z);
    float distance = std::sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    direction.x /= distance;
    direction.y /= distance;
    direction.z /= distance;
    float incre_x(direction.x * interval);
    float incre_y(direction.y * interval);
    float incre_z(direction.z * interval);
    int i = 0;

    // add red line to show connection of two centers
    for (int i = 0; i < distance/interval; ++i) {
        pcl::PointXYZRGB point(255, 0, 0);
        point.x = center_A.x + i * incre_x;
        point.y = center_A.y + i * incre_y;
        point.z = center_A.z + i * incre_z;
        out_cloud->points.push_back(point);
    }
}

void visualize_correspondences(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters_A,
                        const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters_B,
                        const std::vector<std::pair<int, int>>& correspondences,
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& out_cloud)
{
    // 1. draw correponding clusters
    const float supplement_z = 10.f;
    out_cloud->clear();
    for (const auto& pair : correspondences) {
        const auto& cluster_A = clusters_A[pair.first];
        const auto& cluster_B = clusters_B[pair.second];
        int r = rand() % 192 + 64;
        int g = rand() % 192 + 64;
        int b = rand() % 192 + 64;
        for (const auto& p : cluster_A->points) {
            pcl::PointXYZRGB point(r, g, b);
            point.x = p.x;
            point.y = p.y;
            point.z = p.z;
            out_cloud->points.push_back(point);
        }

        // translate cluster B to a higher place
        for (const auto& p : cluster_B->points) {
            pcl::PointXYZRGB point(r, g, b);
            point.x = p.x;
            point.y = p.y;
            point.z = p.z + supplement_z;
            out_cloud->points.push_back(point);
        }
        draw_connected_line(cluster_A, cluster_B, supplement_z, out_cloud);
    }

    // 2. draw the remaining clusters
    std::vector<bool> drawn_bitmap_A(clusters_A.size(), false);
    std::vector<bool> drawn_bitmap_B(clusters_B.size(), false);
    for (const auto& pair : correspondences) {
        drawn_bitmap_A[pair.first] = true;
        drawn_bitmap_B[pair.second] = true;
    }
    
    for (std::size_t i = 0; i < clusters_A.size(); ++i) {
        if (drawn_bitmap_A[i]) continue;
        const auto& cluster = clusters_A[i];
        for (const auto& p : cluster->points) {
            pcl::PointXYZRGB point(63, 63, 63);
            point.x = p.x;
            point.y = p.y;
            point.z = p.z;
            out_cloud->points.push_back(point);
        }
    }

    for (std::size_t i = 0; i < clusters_B.size(); ++i) {
        if (drawn_bitmap_B[i]) continue;
        const auto& cluster = clusters_B[i];
        for (const auto& p : cluster->points) {
            pcl::PointXYZRGB point(63, 63, 63);
            point.x = p.x;
            point.y = p.y;
            point.z = p.z;
            out_cloud->points.push_back(point);
        }
    }
    
}

// 0. downsample
// 1. euclidean cluster
// 2. correspondences
// 3. visualization

int main (int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "usage: ./segments_correspondences cloud_A.pcd cloud_B.pcd" << std::endl;

        return (-1);
    }

    // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_A(new pcl::PointCloud<pcl::PointXYZ>), cloud_B(new pcl::PointCloud<pcl::PointXYZ>);

    reader.read (argv[1], *cloud_A);
    std::cout << "cloud_A has: " << cloud_A->points.size () << " data points." << std::endl;

    reader.read (argv[2], *cloud_B);
    std::cout << "cloud_B has: " << cloud_B->points.size () << " data points." << std::endl;

    // 0. downsample
    const float voxel_size = 0.05f;
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

    // 1. euclidean cluster
    const float cluster_distance = 0.8f;
    const int cluster_min_size = 100;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters_A = segment(cloud_A, cluster_distance, cluster_min_size);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters_B = segment(cloud_B, cluster_distance, cluster_min_size);
    
    std::cout << "cloud A has " << clusters_A.size() << " clusters." << std::endl;
    std::cout << "cloud B has " << clusters_B.size() << " clusters." << std::endl;
    
    // 2. correspondences
    // std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointXYZ>> clusters_with_centers_A;
    // std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointXYZ>> clusters_with_centers_B;

    std::vector<std::pair<int, int>> correspondences = make_correspondences(clusters_A, clusters_B);

    std::cout << "correspondences size: " << correspondences.size() << std::endl;
    for (const auto& correspondence : correspondences) {
        std::cout << "(" << correspondence.first << "," << correspondence.second << ")\n"; 
    }

    // 3. visualization
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    visualize_correspondences(clusters_A, clusters_B, correspondences, out_cloud);



    // 4. save pcd file
    out_cloud->width = 1;
    out_cloud->height = out_cloud->size();
    if ( pcl::io::savePCDFile <pcl::PointXYZRGB> ("correspondences_output.pcd", *out_cloud) == -1 )
    {
        std::cout << "Cloud writing failed." << std::endl;
        return (-1);
    }

    return (0);
}