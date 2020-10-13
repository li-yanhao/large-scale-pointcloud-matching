#include <H5Cpp.h>
#include <H5DataSet.h>
#include <fstream>
#include <iostream>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/random_sample.h>

#include "thread_pool.h"

using namespace H5;

// DEFINE_string("out_dir", "", "out_dir")
DEFINE_string(h5filename, "", "h5filename");
DEFINE_string(file_prefix, "", "out_dir + id + .pcd = filename");
DEFINE_string(correspondence_filename, "", "correspondence_filename");
DEFINE_int32(id_max, 0, "id_max");
DEFINE_int32(num_thread, 7, "num_thread");


constexpr float kClusterTolerance = 0.3f;  // unit: meter
constexpr int kMinClusterSize = 100;       // unit: (number of points)
constexpr int kMaxClusterSize = 512;      // unit: (number of points)

constexpr float kMaxInterSubmapDistance = 30.f;       // unit: meter
constexpr float kMaxInterSegmentDistance = 5.f;       // unit: meter
constexpr float kOverlapTolerance = 0.3f;             // unit: meter
constexpr float kMinInterSegmentOverlapRatio = 0.8f;  // unit: none


template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> extract_segments(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                                                                    float max_distance,
                                                                    int min_size,
                                                                    int max_size);

template<typename PointT>
pcl::PointXYZ calculate_center(const pcl::PointCloud<PointT>& cloud);

template<typename PointT>
float calculate_distance(const PointT& P, const PointT& Q);

template<typename PointT>
bool is_overlapped(const typename pcl::PointCloud<PointT>::Ptr& cloud_A,
                   const typename pcl::PointCloud<PointT>::Ptr& cloud_B,
                   float overlap_ratio);


template<typename PointT>
struct Submap {
    typedef std::shared_ptr<Submap<PointT>> Ptr;
    struct Segment {
        int id;
        typename pcl::PointCloud<PointT>::Ptr cloud;
        PointT center;
    };

    // pcl::PointCloud<PointT>::Ptr cloud;
    int id;
    PointT center;
    std::vector<Segment> segments;

    Submap(const typename pcl::PointCloud<PointT>::Ptr& cloud, int submap_id) :
        id(submap_id),
        center(calculate_center(*cloud))
    {

        std::vector<typename pcl::PointCloud<PointT>::Ptr> segment_clouds =
                extract_segments<PointT>(cloud, kClusterTolerance, kMinClusterSize, kMaxClusterSize);

        int segment_id = 0;
        for (const auto& segment_cloud : segment_clouds) {
            Segment segment{segment_id++, segment_cloud, calculate_center(*segment_cloud)};
            segments.push_back(std::move(segment));
        }
    }
};

struct Correspondence {
    std::pair<int, int> submap_pair;
    std::vector<std::pair<int, int>> segment_pairs;

    template<typename PointT>
    Correspondence(const Submap<PointT>& submap_i, const Submap<PointT>& submap_j);
};


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

    return pcl::PointXYZ((x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2);
}


template<typename PointT>
float calculate_distance(const PointT& P, const PointT& Q)
{
    float dx = P.x - Q.x;
    float dy = P.y - Q.y;
    float dz = P.z - Q.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

template<typename PointT>
bool are_submaps_close(const pcl::PointCloud<PointT>& cloud_A, const pcl::PointCloud<PointT>& cloud_B, float threshold)
{

    pcl::PointXYZ center_A = calculate_center(cloud_A);
    pcl::PointXYZ center_B = calculate_center(cloud_B);

    return calculate_distance(center_A, center_B) < threshold;
}


template<typename PointT>
bool are_submaps_close(const Submap<PointT>& submap_A, const Submap<PointT>& submap_B, float threshold)
{
    return calculate_distance(submap_A.center, submap_B.center) < threshold;
}

// To tell if two clouds are overlapped with a minimum ratio
template<typename PointT>
bool is_overlapped(const typename pcl::PointCloud<PointT>::Ptr& cloud_A,
                   const typename pcl::PointCloud<PointT>::Ptr& cloud_B,
                   float overlap_tolerance,
                   float min_overlap_ratio)
{
    const auto& cloud_large = cloud_A->size() > cloud_B->size() ? cloud_A : cloud_B;
    const auto& cloud_small = cloud_A->size() <= cloud_B->size() ? cloud_A : cloud_B;
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud_large);
    int num_overlapped_pts = 0;
    for (const auto& point : cloud_small->points) {
        std::vector<int> k_indices;
        std::vector<float> k_sqr_distances;
        if (tree->radiusSearch(point, overlap_tolerance, k_indices, k_sqr_distances) > 0) {
            ++num_overlapped_pts;
        }
    }

    return float(num_overlapped_pts) / cloud_small->size() > min_overlap_ratio;
}

template<typename PointT>
Correspondence::Correspondence(const Submap<PointT>& submap_i, const Submap<PointT>& submap_j)
{
    submap_pair = std::make_pair(submap_i.id, submap_j.id);
    for (const auto& segment_i : submap_i.segments) {
        for (const auto& segment_j : submap_j.segments) {
            if (calculate_distance(segment_i.center, segment_j.center) > kMaxInterSegmentDistance)
                continue;
            if (is_overlapped<PointT>(segment_i.cloud,
                                      segment_j.cloud,
                                      kOverlapTolerance,
                                      kMinInterSegmentOverlapRatio)) {

                segment_pairs.push_back(std::make_pair(segment_i.id, segment_j.id));
            }
        }
    }
}


template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> extract_segments(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                                                                    float cluster_tolerance,
                                                                    int min_size,
                                                                    int max_size)
{
    // Creating the KdTree object for the search method of the extraction
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;

    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_size);
    // ec.setMaxClusterSize(max_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end();
         ++it) {
        // if (it->indices.size() > 6000)
        //     continue;

        typename pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
            cloud_cluster->points.push_back(cloud->points[*pit]);  //*
        }

        cloud_cluster->height = cloud_cluster->points.size();
        cloud_cluster->width = 1;
        cloud_cluster->is_dense = true;

        pcl::RandomSample<PointT> rs;
        rs.setInputCloud(cloud_cluster);
        rs.setSample(max_size);
        rs.filter(*cloud_cluster);

        clusters.push_back(std::move(cloud_cluster));
    }

    return clusters;
}

template<typename PointT>
void write_submaps(std::string filename, std::vector<typename Submap<PointT>::Ptr> submaps)
{
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    for (const auto& submap : submaps) {

        const std::string submap_group = "submap_" + std::to_string(submap->id);
        H5Gcreate(file_id, submap_group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /// Create a seperate dataset to denote the number of segments
        {
            hsize_t dims[] = {1};
            hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
            std::string dataset_name = submap_group + "/num_segments";
            hid_t dataset_id = H5Dcreate(file_id,
                                         dataset_name.c_str(),
                                         H5T_NATIVE_INT32,
                                         dataspace_id,
                                         H5P_DEFAULT,
                                         H5P_DEFAULT,
                                         H5P_DEFAULT);

            const int num_segments = submap->segments.size();
            auto status = H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_segments);
            if (status != 0) {
                LOG(INFO) << "Failed to write data for \"" << dataset_name << "\"";
            }

            CHECK_EQ(H5Dclose(dataset_id), 0);
            CHECK_EQ(H5Sclose(dataspace_id), 0);
        }

        for (const auto& segment : submap->segments) {
            const std::string segment_dataset_name = submap_group + "/segment_" + std::to_string(segment.id);
            // H5Gcreate(file_id, segment_group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            // const auto& segment = segments[i];
            const uint32_t data_dim = 3;
            std::vector<double> segment_data(segment.cloud->size() * data_dim);
            for (uint32_t j = 0; j < segment.cloud->size(); ++j) {
                segment_data[j * data_dim] = segment.cloud->points[j].x;
                segment_data[j * data_dim + 1] = segment.cloud->points[j].y;
                segment_data[j * data_dim + 2] = segment.cloud->points[j].z;
            }
            unsigned rank = 2;
            hsize_t dims[] = {segment.cloud->size(), 3};
            hid_t dataspace_id = H5Screate_simple(rank, dims, NULL);
            hid_t dataset_id = H5Dcreate(file_id,
                                         segment_dataset_name.c_str(),
                                         H5T_NATIVE_DOUBLE,
                                         dataspace_id,
                                         H5P_DEFAULT,
                                         H5P_DEFAULT,
                                         H5P_DEFAULT);

            auto status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, segment_data.data());
            if (status != 0) {
                LOG(INFO) << "Failed to write data for \"" << segment_dataset_name << "\"";
            }

            CHECK_EQ(H5Dclose(dataset_id), 0);
            CHECK_EQ(H5Sclose(dataspace_id), 0);
        }
    }

    CHECK_EQ(H5Fclose(file_id), 0);
}


// void write_segments_to_hdf5(const std::vector<pcl::PointCloud<pcl::PointXYZI>>& segments, hid_t file_id, uint32_t
// submap_id)
// {
//     LOG(INFO) << "Submap contains " << segments.size() << " segments." << std::endl;
//     LOG(INFO) << "file_id: " << file_id;


//     const std::string submap_group = "submap_" + std::to_string(submap_id);
//     H5Gcreate(file_id, submap_group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

//     // CHECK(H5Gcreate(file_id, submap_group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) == 0)
//     //         << "Failed to create group \"" << submap_group << "\"";

//     {
//         // Create a seperate dataset to denote the number of segments
//         hsize_t dims[] = {1};
//         hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
//         std::string dataset_name = submap_group + "/num_segments";
//         hid_t dataset_id = H5Dcreate(file_id, dataset_name.c_str(), H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT,
//         H5P_DEFAULT, H5P_DEFAULT);

//         const int num_segments = segments.size();
//         auto status = H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_segments);
//         if (status != 0) {
//             LOG(INFO) << "Failed to write data for \"" << dataset_name << "\"";
//         }

//         CHECK_EQ(H5Dclose(dataset_id), 0);
//         CHECK_EQ(H5Sclose(dataspace_id), 0);
//     }

//     for (uint32_t i = 0; i < segments.size(); ++i) {

//         const std::string segment_dataset_name = submap_group + "/segment_" + std::to_string(i);
//         // H5Gcreate(file_id, segment_group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

//         const auto& segment = segments[i];
//         const uint32_t data_dim = 4;
//         std::vector<double> segment_data(segment.size() * data_dim);
//         for (uint32_t j = 0; j < segment.size(); ++j) {
//             segment_data[j * data_dim] = segment.points[j].x;
//             segment_data[j * data_dim + 1] = segment.points[j].y;
//             segment_data[j * data_dim + 2] = segment.points[j].z;
//             segment_data[j * data_dim + 3] = segment.points[j].intensity;
//         }

//         unsigned rank = 2;
//         hsize_t dims[] = {segment.size(), 4};
//         hid_t dataspace_id = H5Screate_simple(rank, dims, NULL);
//         hid_t dataset_id = H5Dcreate(file_id, segment_dataset_name.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
//         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

//         auto status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, segment_data.data());
//         if (status != 0) {
//             LOG(INFO) << "Failed to write data for \"" << segment_dataset_name << "\"";
//         }

//         CHECK_EQ(H5Dclose(dataset_id), 0);
//         CHECK_EQ(H5Sclose(dataspace_id), 0);
//     }
// }

void write_correspondences(std::string filename, std::vector<Correspondence> correspondences)
{
    boost::property_tree::ptree pt;

    // TODO: avoid hard-coded value
    pt.put("submap_segment_h5_file", "submap_segments.h5");

    boost::property_tree::ptree correspondences_tree;
    for (const auto& correspondence : correspondences) {
        boost::property_tree::ptree correspondence_tree;

        std::stringstream submap_pair_value;
        submap_pair_value << correspondence.submap_pair.first << "," << correspondence.submap_pair.second;
        correspondence_tree.put("submap_pair", submap_pair_value.str());

        std::stringstream segment_pairs_value;
        for (const auto& segment_pair : correspondence.segment_pairs) {
            segment_pairs_value << segment_pair.first << "," << segment_pair.second << ",";
        }
        correspondence_tree.put("segment_pairs", segment_pairs_value.str());

        correspondences_tree.push_back(std::make_pair("", correspondence_tree));
    }

    pt.put_child("correspondences", correspondences_tree);

    std::stringstream json_ss;
    boost::property_tree::json_parser::write_json(json_ss, pt);

    std::ofstream file;
    file.open(filename);
    file << json_ss.str();
    file.close();
}


// int test_submaps(int argc, char** argv)
// {
//     std::vector<pcl::PointXYZ> submap_centers;

//     std::stringstream file_ss;
//     const int id_max = 356;
//     const std::string file_prefix = "/media/li/LENOVO/dataset/kitti/lidar_odometry/extracted_submaps/00/submap_";
//     for (int i = 0; i < id_max; ++i) {
//         file_ss << file_prefix << std::to_string(i) << ".pcd";

//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
//         pcl::io::loadPCDFile(file_ss.str(), *cloud_in);

//         // submap_centers.push_back(calculate_center_by_boundary(*cloud_in));
//         submap_centers.push_back(calculate_center(*cloud_in));

//         file_ss.str("");
//     }

//     std::vector<std::pair<int, int>> submap_correspondences;
//     const float threshold_distance = 50.f;
//     for (std::size_t i = 0; i < submap_centers.size(); ++i) {
//         const auto& center_i = submap_centers[i];
//         for (std::size_t j = 0; j < submap_centers.size(); ++j) {
//             const auto& center_j = submap_centers[j];
//             float distance_xy = std::sqrt(std::pow(center_i.x - center_j.x, 2) + std::pow(center_i.y - center_j.y,
//             2));

//             if (distance_xy < threshold_distance) {
//                 submap_correspondences.push_back(std::make_pair(i, j));
//             }
//         }
//     }

//     for (const auto& correspondence : submap_correspondences) {
//         std::cout << "(" << correspondence.first << "," << correspondence.second << ")\n";
//     }

//     // TODO: save the submap correspondences in a file (e.g. txt file)

//     return 0;
// }


int test_submaps_and_segments(int argc, char** argv)
{

    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    if (FLAGS_file_prefix == "" || FLAGS_h5filename == "" || FLAGS_correspondence_filename == "" || FLAGS_id_max == 0) {
        LOG(INFO) << "Prepare submaps-segments database in the form of h5 file, and generate a correspondence file.";
        LOG(INFO) << "usage: ./submap_correspondences -file_prefix=<submap-cloud-prefix> "
                     "-h5filename=<xxx.h5> "
                     "-correspondence_filename=<xxx.txt>";

        return 0;
    }


    using Submap = Submap<pcl::PointXYZ>;


    std::stringstream file_ss;
    const int id_max = FLAGS_id_max;
    // const std::string file_prefix = "/media/li/LENOVO/dataset/kitti/lidar_odometry/extracted_submaps/00/submap_";
    const std::string file_prefix = FLAGS_file_prefix;

    std::cout << "Starting to create submaps ..." << std::endl;
    ThreadPool pool(FLAGS_num_thread);
    pool.init();
    // auto create_submap = [](const std::string cloud_filename, int id){
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
    //     pcl::io::loadPCDFile(cloud_filename, *cloud_in);

    //     pcl::ApproximateVoxelGrid<pcl::PointXYZ> avg;
    //     avg.setInputCloud(cloud_in);
    //     avg.setLeafSize(0.05f, 0.05f, 0.05f);
    //     avg.filter(*cloud_in);

    //     // submap_centers.push_back(calculate_center_by_boundary(*cloud_in));
    //     Submap submap(cloud_in, id);

    //     return submap;
    // };


    std::vector<Submap::Ptr> submaps;
    submaps.resize(id_max + 1);
    auto initialize_submap = [&submaps](const std::string cloud_filename, int id) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::io::loadPCDFile(cloud_filename, *cloud_in);

        pcl::ApproximateVoxelGrid<pcl::PointXYZ> avg;
        avg.setInputCloud(cloud_in);
        avg.setLeafSize(0.05f, 0.05f, 0.05f);
        avg.filter(*cloud_in);
        // submap.reset(new Submap(cloud_in, id));

        submaps[id] = std::make_shared<Submap>(cloud_in, id);
        LOG(INFO) << "Created submap " << id << " with " << submaps[id]->segments.size() << " segments.";
    };

    // std::vector<std::future<Submap>> futures;
    for (int i = 0; i <= id_max; ++i) {
        file_ss.str("");
        file_ss << file_prefix << std::to_string(i) << ".pcd";

        pool.submit(initialize_submap, file_ss.str(), i);
    }


    pool.shutdown();
    LOG(INFO) << submaps[0]->id;


    std::cout << "Starting to make correspondences ..." << std::endl;

    std::vector<Correspondence> correspondences;
    for (const auto& submap_i : submaps) {
        for (const auto& submap_j : submaps) {
            if (are_submaps_close(*submap_i, *submap_j, kMaxInterSubmapDistance) && (submap_j->id - submap_i->id) > 1) {
                correspondences.emplace_back(*submap_i, *submap_j);
                const Correspondence& correspondence = correspondences.back();
                std::cout << "Found correspondence between submaps " << correspondence.submap_pair.first << " and "
                          << correspondence.submap_pair.second << " with " << correspondence.segment_pairs.size()
                          << " segment pairs\n";
            }
        }
    }

    write_submaps<pcl::PointXYZ>(FLAGS_h5filename, submaps);
    write_correspondences(FLAGS_correspondence_filename, correspondences);
}


int main(int argc, char** argv)
{
    test_submaps_and_segments(argc, argv);

    return 0;
}