import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import open3d as o3d
import argparse
import os
from model.Descriptor.descnet import *
from model.Descriptor.descriptor_dataset import *
import h5py

parser = argparse.ArgumentParser(description='DescriptorVisualize')
parser.add_argument('--dataset_dir', type=str, default="/media/admini/My_data/matcher_database/05", help='dataset_dir')
parser.add_argument('--checkpoints_dir', type=str,
                    default="/media/admini/My_data/matcher_database/checkpoints", help='checkpoints_dir')
parser.add_argument('--add_random_rotation', type=bool, default=True, help='add_random_rotation')
args = parser.parse_args()

def visualize():
    h5_filename = os.path.join(args.dataset_dir, "submap_segments.h5")

    dataset = create_submap_dataset(h5py.File(h5_filename, "r"))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DgcnnModel(k=16, feature_dims=[64, 128, 512], emb_dims=[256, 256], output_classes=32)

    checkpoint_filename = "descriptor-32-dgcnn-kitti00.pth"
    dgcnn_model_checkpoint = torch.load(os.path.join(args.checkpoints_dir, checkpoint_filename),
                                        map_location=lambda storage, loc: storage)
    model.load_state_dict(dgcnn_model_checkpoint)
    print("Loaded model checkpoints from \'{}\'.".format(
        os.path.join(args.checkpoints_dir, checkpoint_filename)))

    model = model.to(dev)

    # model.eval()
    submap_and_descriptors_dict = {}
    i = 0
    with torch.no_grad():
        with tqdm(dataset) as tq:
            for submap_name in tq:
                segments = dataset[submap_name]['segments']
                descriptors = [
                    model(segment.unsqueeze(0).to(dev)).cpu().numpy().reshape(-1) for segment in segments
                ]
                submap_and_descriptors_dict[submap_name] = descriptors
                i += 1
                if i > 200:
                    break

    descriptors_all = np.concatenate(
        [descriptors for descriptors in submap_and_descriptors_dict.values()],
        axis=0)

    np.save("descriptors.npy", descriptors_all)

    pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
    pca.fit(descriptors_all)  # 对样本进行降维
    axis_scale = 5000
    segments_points = []
    segments_labels = []
    # cloud_segment = o3d.geometry.PointCloud()
    max_label = 1
    for submap_name in submap_and_descriptors_dict:
        descriptors = submap_and_descriptors_dict[submap_name]
        segments = dataset[submap_name]['segments']
        segment_scales = dataset[submap_name]['segment_scales']
        reduced_descriptors = pca.transform(descriptors)
        for i in range(dataset[submap_name]['num_segments']):
            segments_points.append(segments[i].numpy() * segment_scales[i].numpy()
                                 + np.array([reduced_descriptors[i][0], reduced_descriptors[i][1], 0]) * axis_scale)
            segments_labels.append(np.ones(segments[i].shape[0]) * max_label)
            max_label += 1
    segments_points = np.concatenate(segments_points)
    segments_labels = np.concatenate(segments_labels)

    submap_colors = plt.get_cmap("tab20")(segments_labels / (max_label if max_label > 0 else 1))
    segments_cloud = o3d.geometry.PointCloud()
    segments_cloud.points = o3d.utility.Vector3dVector(segments_points)
    segments_cloud.colors = o3d.utility.Vector3dVector(submap_colors[:,:3])
    o3d.visualization.draw_geometries([segments_cloud])

    o3d.io.write_point_cloud("segments_distribution.pcd", segments_cloud)


if __name__ == '__main__':
    visualize()