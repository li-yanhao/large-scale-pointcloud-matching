#
# Copyright 2020 Wayz.ai. All Rights Reserved.
#

package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_binary")


cc_library(
    name = "pcl_lib",
    srcs = [],
    hdrs = [],
    deps = [
        "@pcl//:common",
        "@pcl//:search",
        "@pcl//:io",
        "@pcl//:visualization",
        "@pcl//:filters",
        "@pcl//:features",
        "@pcl//:segmentation",
        "@pcl//:sample_consensus",
        
    ],
)

cc_binary(
    name = "region_growing_rgb_segmentation",
    srcs = [
        "region_growing_rgb_segmentation.cpp",
    ],
    deps = [
        "@pcl//:common",
        "@pcl//:search",
        "@pcl//:io",
        "@pcl//:visualization",
        "@pcl//:filters",
        "@pcl//:segmentation",
    ],

    linkopts = [
        "-lboost_system",
        "-lpthread",
    ],
)

cc_binary(
    name = "region_growing_segmentation",
    srcs = [
        "region_growing_segmentation.cpp",
    ],
    deps = [
        "@pcl//:common",
        "@pcl//:search",
        "@pcl//:io",
        "@pcl//:visualization",
        "@pcl//:filters",
        "@pcl//:features",
        "@pcl//:segmentation",
    ],

    linkopts = [
        "-lboost_system",
        "-lpthread",
    ],
)

cc_binary(
    name = "don_segmentation",
    srcs = [
        "don_segmentation.cpp",
    ],
    deps = [
        ":pcl_lib"
    ],
)

cc_binary(
    name = "euclidean_cluster_extraction",
    srcs = [
        "euclidean_cluster_extraction.cpp",
    ],
    deps = [
        ":pcl_lib"
    ],
)

cc_binary(
    name = "conditional_euclidean_clustering",
    srcs = [
        "conditional_euclidean_clustering.cpp",
    ],
    deps = [
        ":pcl_lib"
    ],
)

cc_binary(
    name = "read_hdf5",
    srcs = [
        "read_hdf5.cpp",
    ],
    deps = [
        "@hdf5"
    ],
)

cc_binary(
    name = "write_hdf5",
    srcs = [
        "write_hdf5.cpp",
    ],
    deps = [
        "@hdf5"
    ],
)

cc_binary(
    name = "practice_h5group",
    srcs = [
        "practice_h5group.cpp",
    ],
    deps = [
        "@hdf5"
    ],
)