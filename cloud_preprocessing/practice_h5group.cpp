#include <iostream>
using std::cout;
using std::endl;
#include <string>
#include "H5Cpp.h"
#include <H5DataSet.h>
#include <vector>
#include <array>
#include <string>

using namespace H5;
const H5std_string FILE_NAME( "/home/admini/yanhao/large-scale-pointcloud-matching/tools/carla_scan.h5" );
const H5std_string DATASET_NAME( "IntArray" );
const int    NX_SUB = 3;    // hyperslab dimensions
const int    NY_SUB = 4;
const int    NX = 7;        // output buffer dimensions
const int    NY = 7;
const int    NZ = 3;
const int    RANK_OUT = 3;

// Operator function
extern "C" herr_t file_info(hid_t loc_id, const char *name, const H5L_info_t *linfo,
    void *opdata);

// bool read_file(double *&data, int &row, int &col, const DataSet* dataset, const char *db_name)
bool read_dataset(const DataSet* dataset)
{
    H5T_class_t type_class = dataset->getTypeClass();
    std::cout << type_class << std::endl; // type_class=1 => 浮点数
    FloatType floattype = dataset->getFloatType();
    H5std_string order_string;
    floattype.getOrder(order_string);
    cout << "order_string: " << order_string << endl; // 小端储存
    cout << "Data size is " << floattype.getSize() << endl; // 一个数据占8个字节 => double
    DataSpace dataspace = dataset->getSpace();
    int rank = dataspace.getSimpleExtentNdims(); // 数据有多少个维度
    // hsize_t dims_out[5];
    std::vector<hsize_t> dims_out(rank);
    dataspace.getSimpleExtentDims( dims_out.data(), NULL); // 每个维度分别是多少
    cout << "rank " << rank << ", dimensions " <<
            (unsigned long)(dims_out[0]) << " x " <<
            (unsigned long)(dims_out[1]) << endl;

    // const hsize_t data_size = dataset->getInMemDataSize() / dataset->getFloatType().getSize();
    const hsize_t data_size = dataset->getInMemDataSize() / sizeof(double);

    // double *buf = new double[data_size];s
    std::vector<double> buf(data_size);
    cout << "dataset->getInMemDataSize():" << dataset->getInMemDataSize() << endl;

    // 读出数据到 buf 中
    dataset->read(buf.data(), dataset->getDataType());

    for (size_t i = 0; i < data_size; i++)
    {
        cout << buf[i] << endl;
    }
    
    return true;
}

bool write_dataset()
{
    // 为矩阵matrix进行内存分配，这里采用了技巧分配了连续内存
    int rows = 10;
    int columns = 8;
    double* data_mem = new double[rows*columns];
    double** matrix = new double*[rows];
    for (int i = 0; i < rows; i++)
        matrix[i] = data_mem + i*columns;
    // 为matrix矩阵填入数据，元素个数的整数部分为行号，分数部分为列号
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns; c++) {
            matrix[r][c] = r + 0.01*c;
        }
    }

    // 打开HDF5文件
    hid_t file_id;
    herr_t status;
    file_id = H5Fcreate("/home/admini/yanhao/large-scale-pointcloud-matching/tools/my_file.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // 创建一个组
    // hid_t group_id;
    H5Gcreate(file_id, "/MyGroup1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // 创建数据集的metadata中的dataspace信息项目
    unsigned rank = 2;
    hsize_t dims[2];
    dims[0] = rows;
    dims[1] = columns;
    hid_t dataspace_id;  // 数据集metadata中dataspace的id
    // dataspace_id = H5Screate_simple(int rank, 空间维度
    //              const hsize_t* current_dims, 每个维度元素个数
    //                    - 可以为0，此时无法写入数据
    //                  const hsize_t* max_dims, 每个维度元素个数上限
    //                    - 若为NULL指针，则和current_dim相同，
    //                    - 若为H5S_UNLIMITED，则不舍上限，但dataset一定是分块的(chunked).
    dataspace_id = H5Screate_simple(rank, dims, NULL);

    // 创建数据集中的数据本身
    hid_t dataset_id;    // 数据集本身的id
    // dataset_id = H5Dcreate(loc_id, 位置id
    //              const char *name, 数据集名
    //                hid_t dtype_id, 数据类型
    //                hid_t space_id, dataspace的id
    //             连接(link)创建性质,
    //                 数据集创建性质,
    //                 数据集访问性质)
    dataset_id = H5Dcreate(file_id, "/MyGroup1/dset", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);



    // 将数据写入数据集
    // herr_t 写入状态 = H5Dwrite(写入目标数据集id,
    //                               内存数据格式,
    //                       memory_dataspace_id, 定义内存dataspace和其中的选择
    //                          - H5S_ALL: 文件中dataspace用做内存dataspace，file_dataspace_id中的选择作为内存dataspace的选择
    //                         file_dataspace_id, 定义文件中dataspace的选择
    //                          - H5S_ALL: 文件中datasapce的全部，定义为数据集中dataspace定义的全部维度数据
    //                        本次IO操作的转换性质,
    //                          const void * buf, 内存中数据的位置
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix[0]);

    // 关闭dataset相关对象
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);

    // 关闭文件对象
    status = H5Fclose(file_id);

    cout << "status: " << status << endl;

    // 释放动态分配的内存
    delete[] matrix;
    delete[] data_mem;
    
    return true;
}

int main ()
{
    try{
        /*
        * Now reopen the file and group in the file.
        */
        H5File* file = new H5File(FILE_NAME, H5F_ACC_RDWR);
        Group* group = new Group(file->openGroup("0"));
        DataSet* dataset;
        /*
        * Access "Compressed_Data" dataset in the group.
        */
        try {  // to determine if the dataset exists in the group
            dataset = new DataSet( group->openDataSet( "global_pose" ));

            read_dataset(dataset);
            delete dataset;
        }
        catch( GroupIException not_found_error ) {
            cout << " Dataset is not found." << endl;
        }
        cout << "dataset \"/0/lidar_scan\" is open" << endl;
        /*
        * Close the dataset.
        */


    }  // end of try block
    // catch failure caused by the H5File operations
    catch( FileIException error )
    {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the DataSet operations
    catch( DataSetIException error )
    {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error )
    {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the Attribute operations
    catch( AttributeIException error )
    {
        error.printErrorStack();
        return -1;
    }

    write_dataset();
    return 0;
}

/*
 * Operator function.
 */
herr_t
file_info(hid_t loc_id, const char *name, const H5L_info_t *linfo, void *opdata)
{
    hid_t group;
    /*
     * Open the group using its name.
     */
    group = H5Gopen2(loc_id, name, H5P_DEFAULT);
    /*
     * Display group name.
     */
    cout << "Name : " << name << endl;
    H5Gclose(group);
    return 0;
}