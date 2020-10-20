
import sys
import os
sys.path.append(os.path.dirname(__file__))
print(sys.path)

import numpy as np
import torch
# from SapientNet.Superglue import SuperGlue
from model.Superglue import SuperGlue

from sapientnet_with_dgcnn import DgcnnModel
import open3d as o3d
import matplotlib.pyplot as plt
import h5py
from scipy.spatial.transform import Rotation as R
import json


DATA_DIR = '/media/admini/My_data/0629'
# DATA_DIR = '/home/li/wayz'
h5_filename = os.path.join(DATA_DIR, "submap_segments_downsampled.h5")
correspondences_filename = os.path.join(DATA_DIR, "correspondences.json")

def load_correspondences(correspondences_filename):
    with open(correspondences_filename) as f:
        correspondences_all = json.load(f)['correspondences']
        correspondences_all = [{
            'submap_pair': correspondence['submap_pair'],
            'segment_pairs': np.array(list(map(int, correspondence['segment_pairs'].split(',')[:-1]))).reshape(-1,
                                                                                                               2).transpose(),
        } for correspondence in correspondences_all]
    return correspondences_all

def make_submap_dict(h5file : h5py.File, submap_id : int):
    submap_name = 'submap_' + str(submap_id)
    submap_dict = {}
    submap_dict['num_segments'] = np.array(h5file[submap_name + '/num_segments'])[0]
    segments = []
    center_submap_xy = torch.Tensor([0., 0.])
    num_points = 0
    translation = np.array([20, 20, 0])
    rotation_matrix = R.from_rotvec((-np.pi / 6 + np.random.ranf() * 2 * np.pi / 6) * np.array([0, 0, 1])).as_matrix()
    for i in range(submap_dict['num_segments']):
        # submap_dict[segment_name] = np.array(h5file[submap_name + '/num_segments'])
        segment_name = submap_name + '/segment_' + str(i)
        segment = np.array(h5file[segment_name]) @ rotation_matrix
        segments.append(segment)
        center_submap_xy += segment.sum(axis=0)[:2]
        num_points += segment.shape[0]
    center_submap_xy /= num_points
    # segments = [np.array(segment - np.hstack([center_submap_xy, 0.])) for segment in segments]
    segment_centers = np.array([segment.mean(axis=0) - np.hstack([center_submap_xy, 0.]) for segment in segments])

    submap_dict['segment_centers'] = torch.Tensor(segment_centers)
    submap_dict['segment_scales'] = torch.Tensor(np.array([np.sqrt(segment.var(axis=0)) for segment in segments]))
    submap_dict['segments'] = [torch.Tensor((segment - segment.mean(axis=0)) / np.sqrt(segment.var(axis=0))) for segment
                               in segments]
    submap_dict['segments_original'] = [segment for segment
                               in segments]
    return submap_dict


def match_pipeline(submap_dict_A : dict, submap_dict_B : dict):
    # h5_filename = os.path.join(DATA_DIR, "submap_segments_downsampled.h5")
    # correspondences_filename = os.path.join(DATA_DIR, "correspondences.json")
    # sapientnet_dataset = SapientNetDataset(h5_filename, correspondences_filename, mode='test')

    # train_loader = DataLoader(sapientnet_dataset, batch_size=1, shuffle=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    descriptor_dim = 256
    # model = DescripNet(k=10, in_dim=3, emb_dims=[64, 128, 128, 512], out_dim=descriptor_dim) # TODO: debug here
    model = DgcnnModel(k=5, feature_dims=[64, 128, 256], emb_dims=[512, 256], output_classes=descriptor_dim)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "model-dgcnn-no-dropout.pth"), map_location=torch.device('cpu')))

    super_glue_config = {
        'descriptor_dim': descriptor_dim,
        'weights': '',
        'keypoint_encoder': [32, 64, 128],
        'GNN_layers': ['self', 'cross'] * 6,
        'sinkhorn_iterations': 150,
        'match_threshold': 0.1,
    }
    superglue = SuperGlue(super_glue_config)
    superglue.load_state_dict(torch.load(os.path.join(DATA_DIR, "Superglue-dgcnn-no-dropout.pth"), map_location=dev))

    model.train()
    superglue.train()
    model = model.to(dev)
    superglue = superglue.to(dev)

    meta_info_A = torch.cat([submap_dict_A['segment_centers'], submap_dict_A['segment_scales']], dim=1)
    meta_info_B = torch.cat([submap_dict_B['segment_centers'], submap_dict_B['segment_scales']], dim=1)
    segments_A = submap_dict_A['segments']
    segments_B = submap_dict_B['segments']

    with torch.no_grad():
        # segments_A = [segment.to(dev) for segment in segments_A]
        # segments_B = [segment.to(dev) for segment in segments_B]
        # descriptors_A = torch.Tensor.new_empty(1, 256, len(segments_A), device=dev)
        # descriptors_B = torch.Tensor.new_empty(1, 256, len(segments_B), device=dev)
        descriptors_A = []
        descriptors_B = []
        # for i in range(len(segments_A)):
        #     descriptors_A[0, :, i] = model(segments_A[i], dev)
        # for i in range(len(segments_B)):
        #     descriptors_B.append(model(segment, dev))
        for segment in segments_A:
            # descriptors_A.append(model(segment.to(dev), dev))
            descriptors_A.append(model(segment.reshape(1, -1, 3).to(dev)))
        for segment in segments_B:
            # descriptors_B.append(model(segment.to(dev), dev))
            descriptors_B.append(model(segment.reshape(1, -1, 3).to(dev)))
        descriptors_A = torch.cat(descriptors_A, dim=0).transpose(0, 1).reshape(1, descriptor_dim, -1)
        descriptors_B = torch.cat(descriptors_B, dim=0).transpose(0, 1).reshape(1, descriptor_dim, -1)
        data = {
            'descriptors0': descriptors_A,
            'descriptors1': descriptors_B,
            'keypoints0': meta_info_A.reshape(1,-1,6).to(dev),
            'keypoints1': meta_info_B.reshape(1,-1,6).to(dev),
        }

        match_output = superglue(data)

        return match_output


def visualize_match_result(submap_dict_A, submap_dict_B, match_result, segment_pairs_ground_truth = np.array([[], []])):

    num_segments_A = submap_dict_A['segment_centers'].shape[0]
    num_segments_B = submap_dict_B['segment_centers'].shape[0]
    translation_offset_for_visualize = np.array([0, 0, 30])
    # draw correspondence lines
    points = np.vstack([np.array([segment_original.mean(axis=0) for segment_original in submap_dict_A["segments_original"]]),
                       np.array([segment_original.mean(axis=0) for segment_original in submap_dict_B["segments_original"]])
                       + translation_offset_for_visualize])
    lines = []
    line_labels = []

    pcd_target = o3d.geometry.PointCloud()
    pcd_source = o3d.geometry.PointCloud()
    label = 0
    labels_A = []
    for segment in submap_dict_A['segments_original']:
        labels_A += [label] * segment.shape[0]
        label += 1
        pcd_target.points.extend(o3d.utility.Vector3dVector(np.array(segment)[:, :3]))
    labels_A = np.array(labels_A)

    label_B_offest = num_segments_A
    label = label_B_offest
    labels_B = []
    for segment in submap_dict_B['segments_original']:
        labels_B += [label] * segment.shape[0]
        label += 1
        pcd_source.points.extend(o3d.utility.Vector3dVector(np.array(segment)[:, :3] + translation_offset_for_visualize))
    labels_B = np.array(labels_B)

    if isinstance(match_result['matches0'], torch.Tensor):
        matches_A_to_B = np.array(match_result['matches0'].cpu()).reshape(-1)
    else:
        matches_A_to_B = match_result['matches0'].reshape(-1)
    for label_A in range(len(matches_A_to_B)):
        label_B = matches_A_to_B[label_A] + label_B_offest

        if label_B >= label_B_offest:
            labels_B[labels_B == label_B] = label_A
            lines.append([label_A, label_B])
            candidate_label_B = segment_pairs_ground_truth[:, np.where(segment_pairs_ground_truth[0]==label_A)[0]][1]
            if (label_B-label_B_offest) in candidate_label_B:
                line_labels.append(True)
            else:
                line_labels.append(False)
        else:
            labels_A[labels_A == label_A] = -1

    max_label = labels_A.max()
    labels_B[labels_B > max_label] = -1

    # colors_source = plt.get_cmap("tab20")(labels_A / (max_label if max_label > 0 else 1))
    # colors_source[labels_A < 0] = 0
    # pcd_target.colors = o3d.utility.Vector3dVector(colors_source[:, :3])

    colors_B = plt.get_cmap("tab20")(labels_B / (max_label if max_label > 0 else 1))
    colors_B[labels_B < 0] = 0
    pcd_source.colors = o3d.utility.Vector3dVector(colors_B[:, :3])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    print("precisions={}".format(np.array(line_labels).mean()))

    color_lines = []
    for line_label in line_labels:
        if line_label==True:
            color_lines.append([0, 1, 0])
        else:
            color_lines.append([1, 0, 0])
    line_set.colors = o3d.utility.Vector3dVector(color_lines)

    SEGMENTS_BG_DIR = "/media/admini/My_data/0721/juxin/segments"


    # segments = [np.array(o3d.io.read_point_cloud(os.path.join(SEGMENTS_BG_DIR, file_name)).points) for file_name in os.listdir(SEGMENTS_BG_DIR)]
    # segments = np.vstack(segments)

    LARGE_SCALE_VISUALIZATION = True
    if LARGE_SCALE_VISUALIZATION:
        pcd_bg = o3d.geometry.PointCloud()
        for file_name in os.listdir(SEGMENTS_BG_DIR):
            pcd_bg.points.extend(o3d.io.read_point_cloud(os.path.join(SEGMENTS_BG_DIR, file_name)).points)
        pcd_bg.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_target.paint_uniform_color([0.5, 0.5, 0.5])
        line_set.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([pcd_bg, pcd_target, pcd_source, line_set])
    else:
        o3d.visualization.draw_geometries([pcd_target, pcd_source, line_set])
    # o3d.io.write_line_set("correspondence_lines.pcd", line_set, write_ascii=True)
    # o3d.io.write_point_cloud("pcd_target.pcd", pcd_target, write_ascii=True)
    # o3d.io.write_point_cloud("pcd_source.pcd", pcd_source, write_ascii=True)


# TODO: RANSAC matching

# TODO: pipeline [pcd_A0, ..., pcd_Am] & [pcd_B0, ..., pcd_Bm] => SapientNet input => SapientNet result => RANSAC result => ICP matching => final pcd

def make_submap_dict_from_pcds(segment_pcds : list, add_random_bias = False):
    submap_dict = {}
    segments = []
    center_submap_xy = torch.Tensor([0., 0.])
    num_points = 0
    translation = np.array([5, 5, 0])
    rotation_matrix = R.from_rotvec((-np.pi / 18 + np.random.ranf() * 2 * np.pi / 18) * np.array([0, 0, 1])).as_matrix()
    for pcd in segment_pcds:
        if add_random_bias:
            segment = np.array(pcd.points) @ rotation_matrix + translation
        else:
            segment = np.array(pcd.points)
        segments.append(segment)
        center_submap_xy += segment.sum(axis=0)[:2]
        num_points += segment.shape[0]
    center_submap_xy /= num_points
    segment_centers = np.array([segment.mean(axis=0) - np.hstack([center_submap_xy, 0.]) for segment in segments])

    submap_dict['segment_centers'] = torch.Tensor(segment_centers)
    submap_dict['segment_scales'] = torch.Tensor(np.array([np.sqrt(segment.var(axis=0)) for segment in segments]))
    submap_dict['segments'] = [torch.Tensor((segment - segment.mean(axis=0)) / np.sqrt(segment.var(axis=0))) for segment
                               in segments]
    submap_dict['segments_original'] = segments
    return submap_dict


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (N * d) matrix where "N" is the number of points and "d" the dimension
         ref = (N * d) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d) translation vector
           Such that R * data + T is aligned on ref
    '''
    barycenter_ref = np.mean(ref, axis=0)
    barycenter_data = np.mean(data, axis=0)

    Q_ref = ref - barycenter_ref
    Q_data = data - barycenter_data
    H = Q_data.T.dot(Q_ref)
    U, S, V = np.linalg.svd(H)
    R = V.T.dot(U.T)
    if np.linalg.det(R) < 0:
        U[:, -1] = -U[:, -1]
        R = V.T.dot(U.T)
    T = barycenter_ref - R.dot(barycenter_data)

    return R, T


def ransac_filter(submap_dict_A : dict, submap_dict_B : dict, match_result):
    matches_A_to_B = np.array(match_result['matches0'].cpu()).reshape(-1)
    correspondences_valid = np.vstack([np.where(matches_A_to_B > -1), matches_A_to_B[matches_A_to_B > -1]])
    centers_A = np.array(submap_dict_A["segment_centers"][correspondences_valid[0]].cpu())
    centers_B = np.array(submap_dict_B["segment_centers"][correspondences_valid[1]].cpu())
    num_matches = correspondences_valid.shape[1]
    n, k = 10000, 4
    selections = np.random.choice(num_matches, (n, k), replace=True)

    score = -99999
    R_best, T_best= None, None
    MAX_DISTANCE = 2
    selection_best = None
    for selection in selections:
        R, T = best_rigid_transform(centers_A[selection, :], centers_B[selection, :])
        # centers_aligned_A = R.dot(centers_A[idx_A, :]) + T
        diff = centers_A @ R.T + T - centers_B
        distances_squared = np.sum(diff[:, :2] * diff[:, :2], axis=1)

        if score < (distances_squared < MAX_DISTANCE**2).sum():
            R_best, T_best = R, T
            # score = np.sum(diff * diff, axis=1).mean()
            score = (distances_squared < MAX_DISTANCE**2).sum()
            selection_best = np.where(distances_squared < MAX_DISTANCE**2)
            # selection_best = selection
    matches0_amended = np.ones(match_result["matches0"].reshape(-1).shape[0]) * (-1)
    matches0_amended[correspondences_valid[0, selection_best]] = correspondences_valid[1, selection_best]
    match_result_amended = {"matches0": matches0_amended}

    return match_result_amended

# each segment may correspond to k segments
# not finished
def ransac_filter_advance(submap_dict_A : dict, submap_dict_B : dict, match_result):
    top_k_matches1 = np.array(match_result['top_k_matches1'].cpu()) # k * N
    k, num_matches = top_k_matches1.shape

    # correspondences_valid = np.vstack([np.where(matches_A_to_B > -1), matches_A_to_B[matches_A_to_B > -1]])
    centers_A = np.array(submap_dict_A["segment_centers"].cpu())
    centers_B = np.array(submap_dict_B["segment_centers"].cpu())
    # num_matches = correspondences_valid.shape[1]

    n, l = 5000, 4
    selections_target = np.random.choice(num_matches, (n, l), replace=True)
    selections_source = np.random.choice(k, (n, l), replace=True)

    score = -99999
    R_best, T_best= None, None
    MAX_DISTANCE = 2
    selection_best = None
    for selection in selections:
        R, T = best_rigid_transform(centers_A[selection, :], centers_B[selection, :])
        # centers_aligned_A = R.dot(centers_A[idx_A, :]) + T
        diff = centers_A @ R.T + T - centers_B
        distances_squared = np.sum(diff[:, :2] * diff[:, :2], axis=1)

        if score < (distances_squared < MAX_DISTANCE**2).sum():
            R_best, T_best = R, T
            # score = np.sum(diff * diff, axis=1).mean()
            score = (distances_squared < MAX_DISTANCE**2).sum()
            selection_best = np.where(distances_squared < MAX_DISTANCE**2)
            # selection_best = selection
    matches0_amended = np.ones(match_result["matches0"].reshape(-1).shape[0]) * (-1)
    matches0_amended[correspondences_valid[0, selection_best]] = correspondences_valid[1, selection_best]
    match_result_amended = {"matches0": matches0_amended}

    return match_result_amended

if __name__ == "__main__":
    if False:
        submap_id_A = 15
        submap_id_B = 295

        correspondences = load_correspondences(correspondences_filename)
        segment_pairs_ground_truth = [correspondence for correspondence in correspondences if
                                      correspondence["submap_pair"] == (str(submap_id_A) + ',' + str(submap_id_B))][0][
            'segment_pairs']

        h5_file = h5py.File(h5_filename, 'r')
        submap_dict_A = make_submap_dict(h5_file, submap_id_A)
        submap_dict_B = make_submap_dict(h5_file, submap_id_B)

        match_result = match_pipeline(submap_dict_A, submap_dict_B)
        visualize_match_result(submap_dict_A, submap_dict_B, match_result, segment_pairs_ground_truth)

    if True:
        # SUBMAP_A_DIR = "/home/li/study/intelligent-vehicles/cooper-AR/large-scale-pointcloud-matching/cloud_preprocessing/build/submap_A"
        # SUBMAP_B_DIR = "/home/li/study/intelligent-vehicles/cooper-AR/large-scale-pointcloud-matching/cloud_preprocessing/build/submap_B"
        # SEGMENTS_TARGET_DIR = "/home/admini/yanhao/large-scale-pointcloud-matching/cloud_preprocessing/build/submap_A"
        # SEGMENTS_SOURCE_DIR = "/home/admini/yanhao/large-scale-pointcloud-matching/cloud_preprocessing/build/submap_B"

        SEGMENTS_TARGET_DIR = "/media/admini/My_data/0721/juxin/tmp/segments_target"
        SEGMENTS_SOURCE_DIR = "/media/admini/My_data/0721/juxin/tmp/segments_source"
        pcds_A = [o3d.io.read_point_cloud(os.path.join(SEGMENTS_TARGET_DIR, file_name)) for file_name in os.listdir(SEGMENTS_TARGET_DIR)]
        pcds_B = [o3d.io.read_point_cloud(os.path.join(SEGMENTS_SOURCE_DIR, file_name)) for file_name in os.listdir(SEGMENTS_SOURCE_DIR)]
        submap_dict_A = make_submap_dict_from_pcds(pcds_A, add_random_bias=False)
        submap_dict_B = make_submap_dict_from_pcds(pcds_B)

        match_result = match_pipeline(submap_dict_A, submap_dict_B)
        match_result_amended = ransac_filter(submap_dict_A, submap_dict_B, match_result)

        # visualize_match_result(submap_dict_A, submap_dict_B, match_result)
        visualize_match_result(submap_dict_A, submap_dict_B, match_result_amended)