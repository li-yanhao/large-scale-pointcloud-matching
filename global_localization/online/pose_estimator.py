from model.Superglue.superglue import SuperGlue
from global_localization.common.compute_pose import compute_relative_pose_with_ransac
from model.Superglue.dataset import pts_from_pixel_to_meter
import torch
import os


class PoseEstimator(object):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {
                # "images_dir": "/media/li/lavie/dataset/birdview_dataset/00",
                'superglue': {
                    'weights': 'outdoor',
                    'sinkhorn_iterations': 100,
                    'match_threshold': 0.2,
                },
                'saved_model_path': '/media/li/lavie/dataset/birdview_dataset/saved_models',
                "meters_per_pixel": 0.25,
                "scale": 100,
            }

        self.meters_per_pixel_ = config["meters_per_pixel"]
        self.superglue_  = SuperGlue(config['superglue']).eval()
        # saved_model_file_superglue = os.path.join(config["saved_model_path"], 'superglue-juxin.pth.tar')
        saved_model_file_superglue = os.path.join(config["saved_model_path"], 'superglue-rotation-invariant.pth.tar')
        model_checkpoint = torch.load(saved_model_file_superglue, map_location=lambda storage, loc: storage)
        self.superglue_.load_state_dict(model_checkpoint)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.superglue_.to(self.device_)

        self.resolution_ = int(config["scale"] / config["meters_per_pixel"])

    def estimate_pose(self, query_image_info, candidate_image_info):
        query_features = query_image_info["features"]
        candidate_features = candidate_image_info["features"]
        keypoints0 = torch.stack(candidate_features["keypoints"])
        keypoints1 = torch.stack(query_features["keypoints"])
        data = {
            "descriptors0": torch.stack(candidate_features["descriptors"]).to(self.device_),
            "keypoints0": keypoints0.to(self.device_),
            "scores0": torch.stack(candidate_features["scores"]).to(self.device_),
            "descriptors1": torch.stack(query_features["descriptors"]).to(self.device_),
            "keypoints1": keypoints1.to(self.device_),
            "scores1": torch.stack(query_features["scores"]).to(self.device_),
            "image_shape": (1, 1, self.resolution_, self.resolution_),
        }
        with torch.no_grad():
            matching_result = self.superglue_(data)


        kpts0 = keypoints0[0].cpu().numpy()
        kpts1 = keypoints1[0].cpu().numpy()
        matches = matching_result['matches0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        target_kpts_in_meters = pts_from_pixel_to_meter(mkpts0, self.meters_per_pixel_)
        source_kpts_in_meters = pts_from_pixel_to_meter(mkpts1, self.meters_per_pixel_)
        T_target_source, score = compute_relative_pose_with_ransac(target_kpts_in_meters, source_kpts_in_meters)

        return T_target_source, score