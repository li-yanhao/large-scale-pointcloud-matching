import cv2
import numpy as np
import faiss
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm
from global_localization.online.place_recognizer import PlaceRecognizer
from global_localization.online.feature_extractor import FeatureExtractor
from global_localization.online.pose_estimator import PoseEstimator
from global_localization.common.image_info import make_images_info



class GlobalLocalizer(object):
    def __init__(self, config={}):
        super(GlobalLocalizer, self).__init__()
        default_config = {
            # "database_struct_file": "/media/li/lavie/dataset/birdview_dataset/struct_file_juxin_1023_map.txt",
            # "database_images_dir": "/media/li/lavie/dataset/birdview_dataset/juxin_1023_map",
            # "tmp_image_dir": "/media/li/lavie/dataset/birdview_dataset/juxin_1023_map/",
            "database_struct_file": "/media/li/lavie/dataset/birdview_dataset/struct_file_juxin_0617.txt",
            "database_images_dir": "/media/li/lavie/dataset/birdview_dataset/juxin_0617",
            "tmp_image_dir": "/media/li/lavie/dataset/birdview_dataset/juxin_0617/",
            "netvlad_imgsize": 400,
            "vlad_dim": 256,
            "top_k": 3,
            "meters_per_pixel": 0.25,
            "scale": 100,
            "min_inliers": 20,
            "max_inliers": 40,
            "loop_detect_threshold": 10,
            "pure_localization": False,
        }
        self.config_ = {**default_config, **config}
        self.database_images_dir_ = self.config_["database_images_dir"]
        self.tmp_image_dir = self.config_["tmp_image_dir"]
        self.netvlad_imgsize_ = self.config_["netvlad_imgsize"]
        self.images_info_ = []
        self.index_ = faiss.IndexFlatL2(self.config_['vlad_dim'])
        self.top_k_ = self.config_["top_k"]
        self.pure_localization_ = self.config_["pure_localization"]

        self.feature_imgsize_ = int(self.config_["scale"] / self.config_["meters_per_pixel"])
        self.min_inliers_ = self.config_["min_inliers"]
        self.max_inliers_ = self.config_["max_inliers"]
        self.place_recognizer_ = PlaceRecognizer()
        self.feature_extractor_ = FeatureExtractor()
        self.pose_estimator_ = PoseEstimator()

        self.image_id_ = 0

        if self.pure_localization_:
            print("Loading SPI database from {} ...".format(self.config_["database_images_dir"]))
            self.load_spi_database()
        pass

    def handle_slam_spi(self, image, pose, seq):

        spinetvlad_image = cv2.resize(image, (self.netvlad_imgsize_, self.netvlad_imgsize_), interpolation=cv2.INTER_LINEAR) # 8 ms
        features_image = cv2.resize(image, (self.feature_imgsize_, self.feature_imgsize_), interpolation=cv2.INTER_LINEAR) # 8 ms

        global_descriptor = self.place_recognizer_.extract_descriptor(spinetvlad_image)  # 1 * D
        local_features = self.feature_extractor_.extract_features(features_image) # dict

        image_file = "submap_" + str(seq) + ".png"
        self.image_id_ += 1
        query_image_info = {
            "image_file": image_file,
            "pose": pose,
            'vlad': global_descriptor.squeeze(),
            "features": local_features,
        }
        # best_T_w_target = None
        best_candidate_image_info = None
        best_T_w_source, best_score = None, -1
        if len(self.images_info_) >= self.top_k_:
            assert(len(self.images_info_) == self.index_.ntotal)

            # search spi in database
            if not self.pure_localization_:
                # Deny some adjacent results
                distances, result_indices = self.index_.search(global_descriptor, 50)
                result_indices = result_indices[0]
                result_indices = result_indices[result_indices >= 0]
                result_indices = result_indices[result_indices < (len(self.images_info_) - self.config_['loop_detect_threshold'])]
                if len(result_indices) >= self.top_k_:
                    result_indices = result_indices[:self.top_k_]
                candidate_images_info = [self.images_info_[index] for index in result_indices]
            else:
                distances, result_indices = self.index_.search(global_descriptor, self.top_k_)
                candidate_images_info = [self.images_info_[index] for index in result_indices[0]]

            for candidate_image_info in candidate_images_info:
                T_target_source, score = self.pose_estimator_.estimate_pose(query_image_info, candidate_image_info)
                if T_target_source is None or score < self.min_inliers_:
                    continue
                if score > best_score:
                    T_target_source = np.array(
                        [[T_target_source[0, 0], T_target_source[0, 1], 0, T_target_source[0, 2]],
                         [T_target_source[1, 0], T_target_source[1, 1], 0, T_target_source[1, 2]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
                    # T_target_source = np.array(
                    #     [[1, 0, 0, 0],
                    #      [0, 1, 0, 0],
                    #      [0, 0, 1, 0],
                    #      [0, 0, 0, 1]])
                    T_w_target = candidate_image_info['pose']
                    best_T_w_source = T_w_target @ T_target_source
                    # best_T_w_target = T_w_target
                    best_candidate_image_info = candidate_image_info
                    best_score = score
                if best_score > self.max_inliers_:
                    break

        # save image info
        if not self.pure_localization_:
            self.images_info_.append(query_image_info)
            self.index_.add(global_descriptor)

        # print("Saved SPI ", global_descriptor.shape)
        if best_candidate_image_info is not None:
            # print("candidate position: {}".format(best_candidate_image_info['pose'][:3, 3]))
            print("candidate submap: {}".format(best_candidate_image_info['image_file']))
            print("query submap: {}".format(query_image_info['image_file']))
        # print("query done")
        return best_T_w_source, best_score

    def handle_localization_spi(self, image):
        pass

    def load_spi_database(self, struct_file=None, images_dir=None):
        """
        :param struct_file: xxx.txt
        :param images_dir: directory of SPI images
        :return:
        """

        if struct_file is None:
            struct_file = self.config_['database_struct_file']
        if images_dir is None:
            images_dir = self.config_['database_images_dir']

        self.images_info_ = make_images_info(struct_file)
        global_descriptors = []
        for image_info in tqdm(self.images_info_):
            image = cv2.imread(os.path.join(images_dir, image_info['image_file']), cv2.IMREAD_GRAYSCALE)
            spinetvlad_image = cv2.resize(image, (self.netvlad_imgsize_, self.netvlad_imgsize_), interpolation=cv2.INTER_LINEAR)  # 8 ms
            features_image = cv2.resize(image, (self.feature_imgsize_, self.feature_imgsize_), interpolation=cv2.INTER_LINEAR)  # 8 ms

            global_descriptor = self.place_recognizer_.extract_descriptor(spinetvlad_image)  # 1 * D
            local_features = self.feature_extractor_.extract_features(features_image)  # dict

            image_info['vlad'] = global_descriptor.squeeze()
            image_info['features'] = local_features

            global_descriptors.append(global_descriptor)
        global_descriptors = np.vstack(global_descriptors)
        self.index_.add(global_descriptors)

        assert(len(self.images_info_) == self.index_.ntotal)

        pass
