seq=00
build/bird_view_dataset \
    -file_prefix=/media/admini/LENOVO/dataset/kitti/lidar_odometry/extracted_submaps/${seq}/submap_ \
    -h5filename=/media/admini/LENOVO/dataset/kitti/lidar_odometry/submaps_birdview_${seq}.h5 \
    -correspondence_filename=/media/admini/LENOVO/dataset/kitti/lidar_odometry/correspondences_birdview_${seq}.txt \
    -id_max=20 \
    -num_thread=11

