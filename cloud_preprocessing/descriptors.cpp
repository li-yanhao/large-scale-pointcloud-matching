#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>


namespace {

Eigen::Vector3f getRealSortedEigenValues(Eigen::Matrix<std::complex<float>, 3, 1> eigenValues)
{
  Eigen::Vector3f realEigenValues;
  for (std::size_t i = 0; i < 3; i++){
    realEigenValues(i) = std::real(eigenValues(i));
  }
  if (realEigenValues(0) > realEigenValues(1))
    std::swap(realEigenValues(0), realEigenValues(1));
  if (realEigenValues(1) > realEigenValues(2))
    std::swap(realEigenValues(1), realEigenValues(2));
  if (realEigenValues(0) > realEigenValues(1))
    std::swap(realEigenValues(0), realEigenValues(1));
  // std::cout << "realEigenValues: " << realEigenValues << "\n";
  return realEigenValues; 
}

} // namespace


namespace pcl {

// L: linearity, P: planarity, S: sphericality
struct PointXYZLPS
{
	// PCL_ADD_POINT4D;
	union
	{
		float coordinate[3];
		struct
		{
			float x;
			float y;
			float z;
		};
	};
	float linearity;
	float planarity;
	float sphericity;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
} EIGEN_ALIGN16;


template <typename PointInT, typename PointOutT>
class DescriptorEstimation : public Feature<PointInT, PointOutT>
{
    typedef typename DescriptorEstimation<PointInT, PointOutT>::PointCloudOut PointCloudOut;
    using Feature<PointInT, PointOutT>::input_;
    using Feature<PointInT, PointOutT>::search_parameter_;
    using Feature<PointInT, PointOutT>::search_radius_;
    typedef boost::shared_ptr<NormalEstimation<PointInT, PointOutT> > Ptr;
    typedef boost::shared_ptr<const NormalEstimation<PointInT, PointOutT> > ConstPtr;

protected:

    void computeFeature (PointCloudOut &output)
    {
        for (int i = 0; i < input_->size(); i++){
            const PointInT& searchPoint = input_->points[i];
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            // kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            
            this->searchForNeighbors (i, search_parameter_, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            
            if (pointIdxRadiusSearch.size() < 5) {
                continue;
            }
            typename pcl::PointCloud<PointInT>::Ptr neighbours (new pcl::PointCloud<PointInT>);
            for (std::size_t j = 0; j < pointIdxRadiusSearch.size (); ++j){
                neighbours->points.push_back(input_->points[ pointIdxRadiusSearch[j]]);
            }
            
            
            Eigen::Matrix3f covariance_matrix;

            // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
            Eigen::Vector4f xyz_centroid;
            
            pcl::compute3DCentroid (*neighbours, xyz_centroid);
            // Compute the 3x3 covariance matrix
            pcl::computeCovarianceMatrix (*neighbours, xyz_centroid, covariance_matrix);
            // std::cout << "CCCC" << std::endl;
            // std::cout << "covariance_matrix: \n" << covariance_matrix << "\n";
            // std::cout << "xyz_centroid: \n" << xyz_centroid << "\n";

            Eigen::EigenSolver<Eigen::Matrix3f> eig(covariance_matrix);     // [vec val] = eig(A)
            // std::cout << "DDDD" << std::endl;
            Eigen::Matrix<std::complex<float>, 3, 1> eigenComplexValues = eig.eigenvalues();
            // std::cout << "EEEE" << std::endl;
            // Eigen::Matrix3f D = eig.pseudoEigenvalueMatrix();

            // int col_index, row_index;
            // std::cout << D.maxCoeff(&row_index, &col_index) << endl;
            // std::cout << row_index << " " << col_index << endl;
            // std::cout << "eigen value matrix: " << getRealSortedEigenValues(eigenValues) << "\n";
            Eigen::Vector3f eigenValues = getRealSortedEigenValues(eigenComplexValues);
            // std::cout << "FFFF" << std::endl;

            PointOutT pointWithFeature;

            pointWithFeature.x = searchPoint.x;
            pointWithFeature.y = searchPoint.y;
            pointWithFeature.z = searchPoint.z;
            pointWithFeature.linearity = 1 - eigenValues[1] / eigenValues[2];
            pointWithFeature.planarity = (eigenValues[1] - eigenValues[0]) / eigenValues[2];
            pointWithFeature.sphericity = eigenValues[0] / eigenValues[2];
            output.points.push_back(pointWithFeature);


            // float linearity = 1 - eigenValues[1] / eigenValues[2];
            // float planarity = (eigenValues[1] - eigenValues[0]) / eigenValues[2];
            // float sphericity = eigenValues[0] / eigenValues[2];
            // std::cout << "(" << linearity << ", " << planarity << ", " << sphericity << ")" << std::endl;
            // cloud->points[i].linearity = linearity;
            // cloud->points[i].planarity = planarity;
            // cloud->points[i].sphericity = sphericity;

        }
    }

}; // class DescriptorEstimation

} // namespace pcl

 
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZLPS,// 注册点类型宏
	(float, x, x)
	(float, y, y)
	(float, z, z)
	(float, linearity, linearity)
	(float, planarity, planarity)
	(float, sphericity, sphericity)
)


// pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
// ne.setInputCloud (cloud_out);
// ne.setSearchMethod (search_tree);
// ne.setRadiusSearch (1);
// ne.compute (*cloud_with_normals);


typedef pcl::PointXYZ PointIn;
typedef pcl::PointXYZLPS PointOut;

int test_new(int argc, char** argv)
{
    pcl::PointCloud<PointIn>::Ptr cloud(new pcl::PointCloud<PointIn>);
    pcl::PointCloud<PointOut>::Ptr cloud_out(new pcl::PointCloud<PointOut>);

    pcl::io::loadPCDFile<PointIn> (argv[1], *cloud);
    // pcl::io::loadPLYFile<PointT> (argv[1], *cloud);

    pcl::VoxelGrid<PointIn> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.1f, 0.1f, 0.1f);
	sor.filter(*cloud);

    // pcl::KdTreeFLANN<PointIn> kdtree;
    pcl::search::KdTree<PointIn>::Ptr search_tree (new pcl::search::KdTree<PointIn>);
    search_tree->setInputCloud (cloud);

    pcl::DescriptorEstimation<PointIn, PointOut> de;
    de.setInputCloud(cloud);
    de.setSearchMethod(search_tree);
    de.setRadiusSearch(1);
    de.compute (*cloud_out);

    cloud_out->width = 1;
    cloud_out->height = cloud_out->points.size();
    pcl::io::savePCDFile ("descriptors_output.pcd", *cloud_out);

    return 0;

}

int test_gt(int argc, char** argv)
{
    pcl::PointCloud<PointIn>::Ptr cloud (new pcl::PointCloud<PointIn>);
    pcl::PointCloud<PointOut>::Ptr cloudOut (new pcl::PointCloud<PointOut>);

    pcl::io::loadPCDFile<PointIn> (argv[1], *cloud);
    // pcl::io::loadPLYFile<PointT> (argv[1], *cloud);

    pcl::VoxelGrid<PointIn> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.1f, 0.1f, 0.1f);
	sor.filter(*cloud);

    // Placeholder for the 3x3 covariance matrix at each surface patch
    Eigen::Matrix3f covariance_matrix;
    // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
    Eigen::Vector4f xyz_centroid;

    // Estimate the XYZ centroid
    pcl::compute3DCentroid (*cloud, xyz_centroid);

    // Compute the 3x3 covariance matrix
    pcl::computeCovarianceMatrix (*cloud, xyz_centroid, covariance_matrix);

    // std::cout << "covariance_matrix: \n" << covariance_matrix << "\n";
    // std::cout << "xyz_centroid: \n" << xyz_centroid << "\n";

    pcl::KdTreeFLANN<PointIn> kdtree;

    kdtree.setInputCloud (cloud);

    
    float radius = 0.5f;

    std::cout << "Done" << std::endl;

    for (int i = 0; i < cloud->size(); i++){
        PointIn searchPoint = cloud->points[i];
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        
        pcl::PointCloud<PointIn>::Ptr neighbours (new pcl::PointCloud<PointIn>);
        for (std::size_t j = 0; j < pointIdxRadiusSearch.size (); ++j){
        neighbours->points.push_back(cloud->points[ pointIdxRadiusSearch[j]]);

        }
        if (neighbours->points.size() < 5) {
            continue;
        }
        pcl::compute3DCentroid (*neighbours, xyz_centroid);
        // std::cout << "BBBB" << std::endl;
        // Compute the 3x3 covariance matrix
        pcl::computeCovarianceMatrix (*neighbours, xyz_centroid, covariance_matrix);
        // std::cout << "CCCC" << std::endl;
        // std::cout << "covariance_matrix: \n" << covariance_matrix << "\n";
        // std::cout << "xyz_centroid: \n" << xyz_centroid << "\n";

        Eigen::EigenSolver<Eigen::Matrix3f> eig(covariance_matrix);     // [vec val] = eig(A)
        // std::cout << "DDDD" << std::endl;
        Eigen::Matrix<std::complex<float>, 3, 1> eigenComplexValues = eig.eigenvalues();
        // std::cout << "EEEE" << std::endl;
        // Eigen::Matrix3f D = eig.pseudoEigenvalueMatrix();

        // int col_index, row_index;
        // std::cout << D.maxCoeff(&row_index, &col_index) << endl;
        // std::cout << row_index << " " << col_index << endl;
        // std::cout << "eigen value matrix: " << getRealSortedEigenValues(eigenValues) << "\n";
        Eigen::Vector3f eigenValues = getRealSortedEigenValues(eigenComplexValues);
        // std::cout << "FFFF" << std::endl;

        PointOut pointWithFeature;

        pointWithFeature.x = searchPoint.x;
        pointWithFeature.y = searchPoint.y;
        pointWithFeature.z = searchPoint.z;
        pointWithFeature.linearity = 1 - eigenValues[1] / eigenValues[2];
        pointWithFeature.planarity = (eigenValues[1] - eigenValues[0]) / eigenValues[2];
        pointWithFeature.sphericity = eigenValues[0] / eigenValues[2];
        cloudOut->points.push_back(pointWithFeature);


        // float linearity = 1 - eigenValues[1] / eigenValues[2];
        // float planarity = (eigenValues[1] - eigenValues[0]) / eigenValues[2];
        // float sphericity = eigenValues[0] / eigenValues[2];
        // std::cout << "(" << linearity << ", " << planarity << ", " << sphericity << ")" << std::endl;
        // cloud->points[i].linearity = linearity;
        // cloud->points[i].planarity = planarity;
        // cloud->points[i].sphericity = sphericity;

    }

    cloudOut->width = 1;
    cloudOut->height = cloudOut->points.size();
    pcl::io::savePCDFile ("descriptors_output.pcd", *cloudOut);
}

int main(int argc, char** argv)
{
    // test_gt(argc, argv);
    test_new(argc, argv);

    return 0;
}