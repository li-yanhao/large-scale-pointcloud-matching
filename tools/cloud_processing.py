import open3d as o3d
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# pcd = o3d.io.read_point_cloud("/media/admini/My_data/0721/zhonghuan/point_clouds/Paul_Zhonghuan.bag_segment_136.pcd")
pcd = o3d.io.read_point_cloud("/home/li/study/intelligent-vehicles/cooper-AR/large-scale-pointcloud-matching/cloud_preprocessing/build/cloud_cluster_1.pcd")
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
labels = clustering.labels_

gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X)
labels = gmm.predict(X)

np.unique(labels)
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([outlier_cloud])

labels = np.array(pcd.cluster_dbscan(eps=2, min_points=50, print_progress=True))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])

def iterative_gmm_clustering(pcd : o3d.geometry.PointCloud):
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    X = np.asarray(pcd.points)

    queue = [X]
    results = []

    max_size = 1000
    min_size = 100

    while (len(queue) > 0):
        X = queue.pop(0)
        gmm.fit(X)
        labels = gmm.predict(X)
        X1 = X[labels == 0]
        X2 = X[labels == 1]
        if X1.shape[0] > max_size:
            queue.append(X1)
        elif X1.shape[0] > min_size:
            results.append(X1)
        if X2.shape[0] > max_size:
            queue.append(X2)
        elif X2.shape[0] > min_size:
            results.append(X2)
    return results

results = iterative_gmm_clustering(pcd)

def draw_list_of_clouds(clouds : list):
    pcd_viewed = o3d.geometry.PointCloud()
    i = 1
    for result in clouds:
        pcd_viewed.points.extend(result)
        color = np.array([[0.2*i % 1, 0.3*i %1 , 0.7*i%1]])
        i = i + 1
        pcd_viewed.colors.extend(np.repeat(color, result.shape[0], axis=0))
    o3d.visualization.draw_geometries([pcd_viewed])
