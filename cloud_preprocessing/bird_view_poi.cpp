#include <iostream>
#include <vector>
#include <algorithm>
#include <pcl/io/pcd_io.h>

// 1. read file
// 2. histogram of xy plane
// 3. select the top k grids with most points

#if 0
template<typename T>
class Grid : public std::vector<std::vector<T>> {
public:
    Grid(std::size_t m, std::size_t n) : m_(m), n_(n), std::vector<std::vector<T>>(m, std::vector<T>(n))
    {}

    Grid(std::size_t m, std::size_t n, T val) : m_(m), n_(n), std::vector<std::vector<T>>(m, std::vector<T>(n, val))
    {}

    T at(int x, int y)
    {
        return this[x][y];
    }
private:
    std::size_t m_;
    std::size_t n_;
};
#endif 


template<typename T>
class Grid : public std::vector<T> {
public:
    Grid(std::size_t m, std::size_t n) : m_(m), n_(n), std::vector<T>(m_ * n_)
    {}

    Grid(std::size_t m, std::size_t n, T val) : m_(m), n_(n)
    {
        this->resize(m_ * n_, val);
    }

    T& at(std::size_t x, std::size_t y)
    {
        // CHECK(x>=0 && x < m_);
        // CHECK(y>=0 && y < n_);
        // std::cout << "x * n_ + y = " << x * n_ + y << std::endl;
        return (*this)[x * n_ + y];
    }

    void sort()
    {
        std::sort(this->begin(), this->end());
    }

private:
    std::size_t m_;
    std::size_t n_;
};

// template<typename T, std::size_t M, std::size_t N>
Grid<int> make_grid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float scale, float resolution)
{
    // 1. calculate the center
    pcl::PointXYZ center(0,0,0);
    for (const auto& point : cloud->points) {
        center.x += point.x;
        center.y += point.y;
        center.z += point.z;
    }
    center.x /= cloud->size();
    center.y /= cloud->size();
    center.z /= cloud->size();

    int grid_size = int(scale / resolution);
    Grid<int> grid(grid_size, grid_size, 0);

    std::cout << "grid initialization success\n";

    for (const auto& point : cloud->points) {
        int x_index = (point.x - center.x + scale / 2) / resolution;
        int y_index = (point.y - center.y + scale / 2) / resolution;
        // std::cout << "(xid, yid) = (" << x_index << "," << y_index << ")\n";
        if (0 <=x_index && x_index < grid_size && 0 <= y_index && y_index < grid_size) {
            // std::cout << grid.at(x_index, y_index) << std::endl;
            ++grid.at(x_index, y_index);
        }
    }

    return grid;
}

int bird_view_poi(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "usage: ./bird_view_poi point_cloud.pcd" << std::endl;
    }

    // 1. read file
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);

    reader.read (argv[1], *cloud_in);
    std::cout << "cloud_in has: " << cloud_in->points.size () << " data points." << std::endl;

    // 2. histogram of xy plane
    const float scale = 100.f;
    const float resolution = 0.1f;
    auto grid = make_grid(cloud_in, scale, resolution);

    // std::sort(grid.begin(), grid.end(), [](int x, int y){
    //     return x < y;
    // });

    grid.sort();

    int k = 5;
    for (auto iter = grid.end(); grid.end() - iter < k;) {
        std::cout << *(--iter) << std::endl;
    }

    return 0;
}


int main(int argc, char** argv)
{
    return bird_view_poi(argc, argv);
}