import numpy as np
import cv2
# from matplotlib import pyplot as plt

def demo_orb_bfmatcher():
    img1 = cv2.imread('/media/admini/lavie/dataset/birdview_dataset/00/submap_1.png',0)          # queryImage
    img2 = cv2.imread('/media/admini/lavie/dataset/birdview_dataset/00/submap_3.png',0)   # trainImage

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, None, flags=2)

    cv2.imshow('img3', img3)
    cv2.waitKey(0)


def demo_sift_bfmatcher():
    img1 = cv2.imread('/media/admini/lavie/dataset/birdview_dataset/00/submap_1.png',0)          # queryImage
    img2 = cv2.imread('/media/admini/lavie/dataset/birdview_dataset/00/submap_3.png',0)   # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=200, contrastThreshold=0.002, edgeThreshold=15, sigma=1.2)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, None, flags=2)

    cv2.imshow('img3', img3)
    cv2.waitKey(0)


def demo_sift_bfmatcher_knn():
    img1 = cv2.imread('/media/admini/lavie/dataset/birdview_dataset/00/submap_1.png', 0)
    img2 = cv2.imread('/media/admini/lavie/dataset/birdview_dataset/00/submap_3.png', 0)

    # Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=200, contrastThreshold=0.002, edgeThreshold=15, sigma=1.2)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    cv2.imshow('img3', img3)
    cv2.waitKey(0)



def demo_flannmatcher():

    img1 = cv2.imread('/media/admini/lavie/dataset/birdview_dataset/00/submap_103.png', 0)  # queryImage
    img2 = cv2.imread('/media/admini/lavie/dataset/birdview_dataset/00/submap_105.png', 0)  # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=200, contrastThreshold=0.002, edgeThreshold=15, sigma=1.2)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=7)
    search_params = dict(checks=100)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)


if __name__ == '__main__':
    # demo_orb_bfmatcher()
    # demo_sift_bfmatcher()
    demo_flannmatcher()