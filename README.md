# large-scale-pointcloud-matching

## Model notes

### DescriptorNet

* descriptor-dgcnn-kitti00.pth

DgcnnModel(k=10, feature_dims=[64, 128, 512], emb_dims=[256, 128], output_classes=16)

* descriptor-32-dgcnn-kitti00.pth

DgcnnModel(k=16, feature_dims=[64, 128, 512], emb_dims=[256, 256], output_classes=32)

* descriptor-256-dgcnn-kitti00.pth (current)

model = DgcnnModel(k=16, feature_dims=[64, 128, 512], emb_dims=[256, 256], output_classes=256)

* descriptor-256-dgcnn-e2e.pth

descnet = DgcnnModel(k=5, feature_dims=[64, 128, 256], emb_dims=[512, 256], output_classes=descriptor_dim)
    
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

* ~~convex cut for component extracting~~

* differentiable SVD module for registration transform loss

* ~~sinkhorn (optimal transport) + dustbin + svd = partially overlapped point cloud registration~~

* NetVLAD for birdview image retrieval (Precision~0.7, Optimizer encore le modele pour
 atteindre 85% si possible)

* SuperPoint + SuperGlue for relative pose computation (le reseau pretraine 
 marche tres bien deja, il faudra juste adapter les APIs pour
 localisation globale. S'il nous reste encore du temps, trainer le SuperPoint + SuperGlue)

* Add a model to recognize whether the global localization is good or bad. (Pas necessaire 
si le processus de NetVLAD+SuperPoint+SuperGlue est effectif)


 

## Experiment results

| descriptor dim | pretrained descnet | match_loss | rt_loss | dataset | result |
| ------------- | ------------- | ------------- | ------------- | ---- | ----- | 
|16             |          true |  true | false | ? | ? |
| 256 | false | true | false | juxin | 0.8 |
| 256 | false | false | true | juxin | ? |