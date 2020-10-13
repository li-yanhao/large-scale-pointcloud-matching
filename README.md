# large-scale-pointcloud-matching

## Model notes

### DescriptorNet

* descriptor-dgcnn-kitti00.pth

DgcnnModel(k=10, feature_dims=[64, 128, 512], emb_dims=[256, 128], output_classes=16)

* descriptor-32-dgcnn-kitti00.pth

DgcnnModel(k=16, feature_dims=[64, 128, 512], emb_dims=[256, 256], output_classes=32)

* descriptor-256-dgcnn-kitti00.pth (current)

model = DgcnnModel(k=16, feature_dims=[64, 128, 512], emb_dims=[256, 256], output_classes=256)

### Superglue

* current config

````
super_glue_config = {
    'descriptor_dim': descriptor_dim,
    'weights': '',
    'keypoint_encoder': [32, 64, 128, 256],
    'GNN_layers': ['self', 'cross'] * 9,
    'sinkhorn_iterations': 150,
    'match_threshold': 0.2,
}
````
## Some alternative methods

* convex cut for component extracting

* differentiable SVD module for registration transform loss

* sinkhorn (optimal transport) + dustbin + svd = partially overlapped point cloud registration
