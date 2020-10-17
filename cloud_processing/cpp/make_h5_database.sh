seq=00
build/submap_correspondences \
    -file_prefix=/media/li/LENOVO/dataset/kitti/lidar_odometry/extracted_submaps/${seq}/submap_ \
    -h5filename=/media/li/LENOVO/dataset/kitti/lidar_odometry/submap_segments_${seq}.h5 \
    -correspondence_filename=/media/li/LENOVO/dataset/kitti/lidar_odometry/correspondences_${seq}.txt \
    -id_max=226 \
    -num_thread=7

seq=02
build/submap_correspondences \
    -file_prefix=/media/li/LENOVO/dataset/kitti/lidar_odometry/extracted_submaps/${seq}/submap_ \
    -h5filename=/media/li/LENOVO/dataset/kitti/lidar_odometry/submap_segments_${seq}.h5 \
    -correspondence_filename=/media/li/LENOVO/dataset/kitti/lidar_odometry/correspondences_${seq}.txt \
    -id_max=232 \
    -num_thread=7

seq=05
build/submap_correspondences \
    -file_prefix=/media/li/LENOVO/dataset/kitti/lidar_odometry/extracted_submaps/${seq}/submap_ \
    -h5filename=/media/li/LENOVO/dataset/kitti/lidar_odometry/submap_segments_${seq}.h5 \
    -correspondence_filename=/media/li/LENOVO/dataset/kitti/lidar_odometry/correspondences_${seq}.txt \
    -id_max=137 \
    -num_thread=7

seq=08
build/submap_correspondences \
    -file_prefix=/media/li/LENOVO/dataset/kitti/lidar_odometry/extracted_submaps/${seq}/submap_ \
    -h5filename=/media/li/LENOVO/dataset/kitti/lidar_odometry/submap_segments_${seq}.h5 \
    -correspondence_filename=/media/li/LENOVO/dataset/kitti/lidar_odometry/correspondences_${seq}.txt \
    -id_max=202 \
    -num_thread=7