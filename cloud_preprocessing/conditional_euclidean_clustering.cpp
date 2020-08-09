#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;

bool
enforceIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    if (std::abs (point_a.intensity - point_b.intensity) < 15.0f)
    return (true);
  else
    return (false);
}


// bool
// enforceCurvatureOrIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
// {
//   Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
//   if (std::abs (point_a.intensity - point_b.intensity) < 5.0f)
//     return (true);
//   if (std::abs (point_a_normal.dot (point_b_normal)) < 0.05)
//     return (true);
//   return (false);
// }

bool
customCondition(const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  // std::cout << "a normal z: " << point_a_normal.z() << std::endl;
  // std::cout << "b normal z: " << point_b_normal.z() << std::endl;
  if (squared_distance < 0.25 && (point_a.curvature < 0.05 && point_b.curvature < 0.05)) {
    return (true);
  }
  return (false);
}

bool
enforceCurvatureSimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  // std::cout << "a normal z: " << point_a_normal.z() << std::endl;
  // std::cout << "b normal z: " << point_b_normal.z() << std::endl;
  if (std::abs (point_a.curvature - point_b.curvature) < 0.4 )
    return (true);
  return (false);
}

bool
enforceCurvatureSmall (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  // std::cout << "a normal z: " << point_a_normal.z() << std::endl;
  // std::cout << "b normal z: " << point_b_normal.z() << std::endl;
  if (point_a.curvature < 0.09 && point_b.curvature < 0.09)
    return (true);
  return (false);
}

bool
customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  if (squared_distance < 1)
  {
    if (std::abs (point_a.intensity - point_b.intensity) < 20.0f)
      return (true);
    if (std::abs (point_a_normal.dot (point_b_normal)) > 0.9)
      return (true);
  }
  else
  {
    if (std::abs (point_a.intensity - point_b.intensity) < 10.0f)
      return (true);
  }
  return (false);
}

int
main (int argc, char** argv)
{
  // Data containers used
  pcl::PointCloud<PointTypeIO>::Ptr cloud_in (new pcl::PointCloud<PointTypeIO>), cloud_out (new pcl::PointCloud<PointTypeIO>);
  pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>);
  pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
  pcl::search::KdTree<PointTypeIO>::Ptr search_tree (new pcl::search::KdTree<PointTypeIO>);
  pcl::console::TicToc tt;

  // Load the input point cloud
  std::cerr << "Loading...\n", tt.tic ();
  pcl::io::loadPCDFile (argv[1], *cloud_in);
  std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_in->points.size () << " points\n";

  // Downsample the cloud using a Voxel Grid class
  std::cerr << "Downsampling...\n", tt.tic ();
  pcl::VoxelGrid<PointTypeIO> vg;
  vg.setInputCloud (cloud_in);
  vg.setLeafSize (0.1, 0.1, 0.1);
  vg.setDownsampleAllData (true);
  vg.filter (*cloud_out);
  cloud_out = cloud_in;
  std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_out->points.size () << " points\n";


  // Set up a Normal Estimation class and merge data in cloud_with_normals
  std::cerr << "Computing normals...\n", tt.tic ();
  pcl::copyPointCloud (*cloud_out, *cloud_with_normals);
  pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
  ne.setInputCloud (cloud_out);
  ne.setSearchMethod (search_tree);
  ne.setRadiusSearch (1);
  ne.compute (*cloud_with_normals);
  std::cerr << ">> Done: " << tt.toc () << " ms\n";

  pcl::search::KdTree<PointTypeFull>::Ptr tree (new pcl::search::KdTree<PointTypeFull>);
  // Set up a Conditional Euclidean Clustering class
  std::cerr << "Segmenting to clusters...\n", tt.tic ();
  pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
  cec.setInputCloud (cloud_with_normals);
  cec.setConditionFunction (&customCondition);
  cec.setClusterTolerance(0.5);
  // cec.setSearchMethod (search_tree);
  cec.searcher_ = tree;
  cec.setMinClusterSize (100);
  // cec.setMaxClusterSize (cloud_with_normals->points.size () / 100);
  cec.segment (*clusters);
  cec.getRemovedClusters (small_clusters, large_clusters);
  std::cerr << ">> Done: " << tt.toc () << " ms\n";

  // Using the intensity channel for lazy visualization of the output
  for (uint32_t i = 0; i < small_clusters->size (); ++i)
    for (uint32_t j = 0; j < (*small_clusters)[i].indices.size (); ++j)
      cloud_out->points[(*small_clusters)[i].indices[j]].intensity = -2.0;
  for (uint32_t i = 0; i < large_clusters->size (); ++i)
    for (uint32_t j = 0; j < (*large_clusters)[i].indices.size (); ++j)
      cloud_out->points[(*large_clusters)[i].indices[j]].intensity = +10.0;
  for (uint32_t i = 0; i < clusters->size (); ++i)
  {
    uint32_t label = rand () % 8;
    for (uint32_t j = 0; j < (*clusters)[i].indices.size (); ++j)
      cloud_out->points[(*clusters)[i].indices[j]].intensity = label;
  }

  std::sort(clusters->begin(), clusters->end(), [](auto& cluster_x, auto& cluster_y) {
    return cluster_x.indices.size() < cluster_y.indices.size();
  });

  std::cout << "We got " << small_clusters->size() << " small clusters, " << large_clusters->size () << " large clusters and "
            << clusters->size() << " normal clusters." << std::endl;

  for (size_t i = 0; i < clusters->size(); ++i) {
    std::cout << "cluster " << i << " contains " << (*clusters)[i].indices.size() << " points" << std::endl;
  }

#if 0
  // Remove the largest cluster, ground points, from the point cloud
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  for (size_t i = 0; i < clusters->back().indices.size(); ++i) {
    inliers->indices.emplace_back(clusters->back().indices[i]);
  }
  inliers->header = clusters->back().header;

  pcl::ExtractIndices<pcl::PointXYZI> extract;
  extract.setInputCloud (cloud_out);
  extract.setIndices(inliers);
  // extract.setNegative (false);

  // // Get the points associated with the planar surface
  // extract.filter (*cloud_out);
  // std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

  // Remove the planar inliers, extract the rest
  extract.setNegative (true);
  extract.filter (*cloud_out);
#endif 

  // Save the output point cloud
  std::cerr << "Saving...\n", tt.tic ();
  pcl::io::savePCDFile ("output.pcd", *cloud_out);
  std::cerr << ">> Done: " << tt.toc () << " ms\n";

  return (0);
}