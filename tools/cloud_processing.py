import open3d as o3d
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud("/media/admini/My_data/0721/zhonghuan/point_clouds/Paul_Zhonghuan.bag_segment_136.pcd")
# o3d.visualization.draw_geometries([pcd])
X = np.asarray(pcd.points)





# plane segmentation
plane_model, inliers = pcd.segment_plane(distance_threshold=0.5,
                                         ransac_n=6,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

ground_cloud = pcd.select_down_sample(inliers)
ground_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_down_sample(inliers, invert=True)
outlier_cloud.paint_uniform_color([0, 1, 0])

X = np.asarray(outlier_cloud.points)

# clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=2).fit(X)
clustering = DBSCAN(eps=2, min_samples=50, algorithm='kd_tree', leaf_size=80, n_jobs=6).fit(X)
# clustering = OPTICS(min_samples=100, n_jobs=4).fit(X)
# labels = GaussianMixture(n_components=20).fit_predict(X)
clustering.labels_ = GaussianMixture(n_components=20).fit_predict(X)

np.unique(clustering.labels_)
max_label = clustering.labels_.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(clustering.labels_ / (max_label if max_label > 0 else 1))
colors[clustering.labels_ < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([outlier_cloud])

labels = np.array(pcd.cluster_dbscan(eps=2, min_points=50, print_progress=True))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])