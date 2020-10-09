#!/usr/bin/env bash

if [[ $# < 2 ]]   # Shell
then
    echo "usage: ./match_pcds.sh cloud_A.pcd cloud_B.pcd"
    exit
else
    pcd_A=$1
    pcd_B=$2
    echo "pcd_A: "${pcd_A}
    echo "pcd_B: "${pcd_B}
fi

cd build/submap_A
rm *.pcd
../euclidean_cluster_extraction ${pcd_A}
cd ../submap_B
rm *.pcd
../euclidean_cluster_extraction ${pcd_B}

cd ../../../SapientNet


python matching_visualization.py

# Self-supervised Large scale point segment match net (LesPmant)