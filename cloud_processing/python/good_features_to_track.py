from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255
def cornerHarris_demo(val):
    src_drawn = src.copy()


    corners = cv.goodFeaturesToTrack(src_gray,val,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(src_drawn, (x,y), 4, (0,255,0), 1)



    # thresh = val
    # # Detector parameters
    # blockSize = 2
    # apertureSize = 3
    # k = 0.04
    # # Detecting corners
    # dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
    # # Normalizing
    # dst_norm = np.empty(dst.shape, dtype=np.float32)
    # cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    # dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    
    # count_poi = 0
    # # Drawing a circle around corners
    # for i in range(dst_norm.shape[0]):
    #     for j in range(dst_norm.shape[1]):
    #         if int(dst_norm[i,j]) > thresh:
    #             cv.circle(dst_norm_scaled, (j,i), 4, (0), 1)
    #             cv.circle(src_drawn, (j,i), 4, (0,255,0), 1)
    #             count_poi += 1
    print("count_poi:", len(corners))
    # Showing the result
    cv.namedWindow(corners_window)
    cv.imshow(corners_window, src_drawn)
# Load source image and convert it to gray
parser = argparse.ArgumentParser(description='Code for Harris corner detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='/media/li/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset/00/submap_100.png')
# parser.add_argument('--input', help='Path to input image.', default='/home/li/Pictures/test-1.png')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
src = cv.resize(src, (500,500)) * 5

if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# Create a window and a trackbar
cv.namedWindow(source_window)
thresh = 120 # initial threshold
cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_demo)
cv.imshow(source_window, src)
cornerHarris_demo(thresh)
cv.waitKey()


