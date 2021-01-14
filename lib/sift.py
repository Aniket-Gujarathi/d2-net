from PIL import Image
import cv2
import numpy as np
import pandas as pd
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform

data = pd.read_csv('/home/udit/d2-net/lib/ipm/ipm_Pairs.csv', sep=',', header=1, names=['front', 'rear'])
imgFiles = [data['front'], data['rear']]
for i in range(len(imgFiles[0][:])):
    im1 = np.array(Image.open(imgFiles[0][i]))
    im2 = np.array(Image.open(imgFiles[1][i]))
    sift = cv2.xfeatures2d.SIFT_create(100)
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key=lambda x:x.distance)

    src_pts = np.array(np.float32([kp1[m.queryIdx].pt for m in matches]))
    dst_pts = np.array(np.float32([kp2[m.trainIdx].pt for m in matches]))

    model, inlier = ransac((src_pts, dst_pts), AffineTransform, min_samples=4, residual_threshold=8, max_trials=10000)
    n_inliers = np.sum(inlier)
    print('Number of inliers: %d.' % n_inliers)

    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inlier]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inlier]]
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

    image3 = cv2.drawMatches(im1, inlier_keypoints_left, im2, inlier_keypoints_right, placeholder_matches, None)
    cv2.imwrite('/scratch/udit/robotcar/overcast/ipm/sift/pair' + str(i) + '.png',image3)
    print(i)
