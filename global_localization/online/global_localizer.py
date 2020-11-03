import cv2
import numpy as np
import faiss
from scipy.spatial.transform import Rotation as R
import os

from global_localization.online.place_recognizer import PlaceRecognizer
from global_localization.online.feature_extractor import FeatureExtractor
from global_localization.online.pose_estimator import PoseEstimator



class GlobalLocalizer(object):
    def __init__(self, config={}):
        super(GlobalLocalizer, self).__init__()
        default_config = {
            "database_image_dir": "/media/li/lavie/dataset/birdview_dataset/juxin_1023_map/",
            "tmp_image_dir": "/media/li/lavie/dataset/birdview_dataset/juxin_1023_map/",
            "netvlad_imgsize": 400,
            "vlad_dim": 256,
            "top_k": 1,
            "meters_per_pixel": 0.25,
            "scale": 100,
            "min_inliers": 20,
            "max_inliers": 40,
        }
        config = {**default_config, **config}
        self.database_image_dir = config["database_image_dir"]
        self.tmp_image_dir = config["tmp_image_dir"]
        self.netvlad_imgsize_ = config["netvlad_imgsize"]
        self.images_info_ = []
        self.index_ = faiss.IndexFlatL2(config['vlad_dim'])
        self.top_k_ = config["top_k"]

        self.feature_imgsize_ = int(config["scale"] / config["meters_per_pixel"])
        self.min_inliers_ = config["min_inliers"]
        self.max_inliers_ = config["max_inliers"]
        self.place_recognizer_ = PlaceRecognizer()
        self.feature_extractor_ = FeatureExtractor()
        self.pose_estimator_ = PoseEstimator()

        self.image_id_ = 0

        pass

    def handle_slam_spi(self, image, pose):

        spinetvlad_image = cv2.resize(image, (self.netvlad_imgsize_, self.netvlad_imgsize_)) # 8 ms
        features_image = cv2.resize(image, (self.feature_imgsize_, self.feature_imgsize_)) # 8 ms

        global_descriptor = self.place_recognizer_.extract_descriptor(spinetvlad_image)  # 1 * D
        local_features = self.feature_extractor_.extract_features(features_image)


        image_file = os.path.join(self.tmp_image_dir, "submap_" + str(self.image_id_) + ".png")
        query_image_info = {
            "image_file": image_file,
            "pose": pose,
            'vlad': global_descriptor.squeeze(),
            "features": local_features,
        }

        best_query_pose, best_score = None, -1
        if len(self.images_info_) >= self.top_k_:


            # search spi in database

            distances, result_indices = self.index_.search(global_descriptor, self.top_k_)
            candidate_images_info = [self.images_info_[index] for index in result_indices[0]]


            for candidate_image_info in candidate_images_info:
                query_pose, score = self.pose_estimator_.estimate_pose(query_image_info, candidate_image_info)
                if score > self.max_inliers_:
                    best_query_pose, best_score = query_pose, score
                    break
                if score < self.min_inliers_:
                    continue
                if score > best_score:
                    best_query_pose, best_score = query_pose, score

        # save image info
        self.images_info_.append(query_image_info)
        self.index_.add(global_descriptor)

        print("Saved SPI ", global_descriptor.shape)


        # print("query done")
        return best_query_pose, best_score

    def handle_localization_spi(self, image):
        pass

    def load_spi_database(self):
        pass
