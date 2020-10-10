seq=08
build/submap_correspondences \
    -file_prefix=/media/admini/LENOVO/dataset/kitti/lidar_odometry/extracted_submaps/${seq}/submap_ \
    -h5filename=/media/admini/LENOVO/dataset/kitti/lidar_odometry/submap_segments_${seq}.h5 \
    -correspondence_filename=/media/admini/LENOVO/dataset/kitti/lidar_odometry/correspondences_${seq}.txt \
    -id_max=551 \
    -num_thread=11
