#!/usr/bin/env python
import sys
sys.path.append("../../")

import rospy
from std_msgs.msg import String
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import cv2
import threading

from global_localization.online.place_recognizer import PlaceRecognizer
from global_localization.online.feature_extractor import FeatureExtractor
from global_localization.online.pose_estimator import PoseEstimator
from global_localization.online.global_localizer import GlobalLocalizer


"""
This node handles (Submap Projection Image) SPI images for global localization use.
The node is written as a ROS node.
Inputs:
    database SPI images
    global pose of database SPI images
    query SPI image
Outputs:
    global pose of query SPI image
"""

__author__ =  'Yanhao LI <yanhao.li at outlook.com>'
__version__=  '0.1'
__license__ = 'BSD'


database_spi_topic = "database_spi_image"
query_spi_topic = "query_spi_image"


def array2CompressedImage(array):
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "png"
    msg.data = np.array(cv2.imencode('.png', array)[1]).tostring()
    return msg


def CompressedImage2Array(compressed_image):
    np_arr = np.fromstring(compressed_image.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    # msg.format = "png"
    # msg.data = np.array(cv2.imencode('.png', array)[1]).tostring()
    return image


class SpiHandler(object):
    def __init__(self):
        super().__init__()
        rospy.init_node('spi_handler', anonymous=True)
        # self.database_spi_sub_ = rospy.Subscriber("query_spi_image", CompressedImage, self.db_spi_image_callback, queue_size=1)

        ### For Test Use ###
        # self.place_recognizer_ = PlaceRecognizer()
        # self.feature_extractor_ = FeatureExtractor()
        # self.pose_estimator_ = PoseEstimator()
        # self.image_id_ = 0
        # self.query_spi_sub_ = rospy.Subscriber("query_spi_image", CompressedImage, self.query_spi_image_callback, queue_size=1)


        ### For Release Use ###
        self.global_localizer_ = GlobalLocalizer()
        self.query_spi_sub_ = rospy.Subscriber("query_spi_image", CompressedImage, self.slam_spi_image_callback,
                                               queue_size=1)


        self.fake_spi_pub_ = rospy.Publisher('query_spi_image', CompressedImage, queue_size=10)
        sending_thread_ = threading.Thread(target=self.spi_image_player)
        sending_thread_.start()


    def db_spi_image_callback(self, msg):
        image = CompressedImage2Array(msg)
        cv2.imshow("spi_image_callback", image)
        cv2.waitKey(delay=1)
        pose = np.identity(4)
        self.place_recognizer_.save_spi(image)
        print(image.shape)

    def query_spi_image_callback(self, msg):
        image = CompressedImage2Array(msg)
        image_spinetvlad = cv2.resize(image, (600, 600), interpolation=cv2.INTER_LINEAR)
        # print("decoded image msg", image.shape)
        results = self.place_recognizer_.query_spi(image_spinetvlad)

        if results is not None:
            candidate_image_filenames = [result['image_file'] for result in results]
            # cv2.imshow("query_image", image)
            result_filename = "/media/li/lavie/dataset/birdview_dataset/00/" + candidate_image_filenames[0]
            result_image = cv2.imread(result_filename)
            # cv2.imshow("result_image", result_image)
            # cv2.waitKey(delay=1)
            # print("query result:", candidate_image_filenames)

        image_features = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LINEAR)
        features = self.feature_extractor_.extract_features(image_features)
        pose = np.identity(4)
        image_dir = "/media/li/lavie/dataset/birdview_dataset/05/"
        image_file = image_dir + "submap_" + str(self.image_id_) + ".png"
        self.image_id_ += 1
        image_info = {
            "image_file": image_file,
            "position": pose[:3,3],
            "orientation": pose[:3,:3],
            "features": features,
        }

        # print("features:", features)

        match_result = self.pose_estimator_.estimate_pose(image_info, image_info)
        print("match_result:", match_result is not None)
        print("query done")

    def slam_spi_image_callback(self, msg):
        image = CompressedImage2Array(msg)
        fake_pose = np.identity(4)
        result = self.global_localizer_.handle_slam_spi(image, fake_pose, msg.header.seq)
        # print("result:", result)
        pose, score = result
        if pose is not None:
            position = pose[:3, 3]
            print("position: ", position)
        else:
            print("query failed")
        # print("query done")
    
    def spi_image_player(self):
        img_id = 0
        rate = rospy.Rate(2.5)  # 3 Hz
        while not rospy.is_shutdown():
            img_filename = "/media/li/lavie/dataset/birdview_dataset/05/submap_" + str(img_id) + ".png"
            rospy.loginfo(img_filename)
            image = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)

            msg = array2CompressedImage(image)
            msg.header.seq = img_id
            self.fake_spi_pub_.publish(msg)

            img_id += 1

            # cv2.imshow("spi_image", image)
            # cv2.waitKey(delay=1)
            rate.sleep()
            pass


if __name__ == '__main__':
    try:
        sh = SpiHandler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
