import numpy as np
import torch
import os
import cv2
import math
import datetime
import random
import sys
sys.path.append('../')
from lib.save_features import extract
from lib.model_test import D2Net
from lib.model_test import D2Net
from lib.pyramid import process_multiscale

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

np.random.seed(0)

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, nfeatures):

        self.files = []
        self.files += [train_path + f for f in os.listdir(train_path)]

        self.nfeatures = nfeatures
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

        # Creating CNN model
        self.model = D2Net(
        	model_file='/home/udit/d2-net/checkpoints/checkpoint_road_more/d2.15.pth',
        	use_relu=True,
        	use_cuda=use_cuda
        )

        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def __len__(self):
        return len(self.files)

    def randomH(self, img1, min=0, max=360):
        img1 = np.array(img1)
        width, height = img1.shape
        theta = np.random.randint(low=min, high=max) * (np.pi / 180)
        Tx = width / 2
        Ty = height / 2
        sx = random.uniform(-1e-2, 1e-2)
        sy = random.uniform(-1e-2, 1e-2)
        p1 = random.uniform(-1e-4, 1e-4)
        p2 = random.uniform(-1e-4, 1e-4)

        alpha = np.cos(theta)
        beta = np.sin(theta)

        He = np.matrix([[alpha, beta, Tx * (1 - alpha) - Ty * beta], [-beta, alpha, beta * Tx + (1 - alpha) * Ty], [0, 0, 1]])
        Ha = np.matrix([[1, sy, 0], [sx, 1, 0], [0, 0, 1]])
        Hp = np.matrix([[1, 0, 0], [0, 1, 0], [p1, p2, 1]])

        H = He @ Ha @ Hp

        img2 = cv2.warpPerspective(img1, H, dsize=(width, height))

		#cv2.imshow("Image", img2)
		#cv2.waitKey(0)

        return img2, H

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img1 = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        img2, M = self.randomH(img1, min=0, max=360)

        feat1 = extract(img1, self.model, self.device)
        feat2 = extract(img2, self.model, self.device)
        kp1, descs1, scores1 = feat1['keypoints'], feat1['descriptors'], feat1['scores']
        kp2, descs2, scores2 = feat2['keypoints'], feat2['descriptors'], feat2['scores']
        kp1 = np.delete(kp1, 2, 1)
        kp2 = np.delete(kp2, 2, 1)

        # limit the number of keypoints
        kp1_num = min(self.nfeatures, len(kp1))

        kp2_num = min(self.nfeatures, len(kp2))
        kp1 = kp1[:kp1_num]
        kp2 = kp2[:kp2_num]

        kp1_np = np.array([(kp[0], kp[1]) for kp in kp1])
        kp2_np = np.array([(kp[0], kp[1]) for kp in kp2])

        # skip this image pair if no keypoints detected in image
        if len(kp1) < 1 or len(kp2) < 1:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': img1,
                'image1': img2,
                'file_name': file_name
            }

        # confidence of each key point
        # scores1_np = np.array([kp.response for kp in kp1])
        # scores2_np = np.array([kp.response for kp in kp2])

        kp1_np = kp1_np[:kp1_num, :]
        kp2_np = kp2_np[:kp2_num, :]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]
        scores1_np = scores1[:kp1_num]
        scores2_np = scores2[:kp2_num]

        # obtain the matching matrix of the image pair
        matched = self.matcher.match(descs1, descs2)
        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :]
        dists = cdist(kp1_projected, kp2_np)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.)
        descs2 = np.transpose(descs2 / 256.)

        img1 = torch.from_numpy(img1/255.).double()[None].cuda()
        img2 = torch.from_numpy(img2/255.).double()[None].cuda()
        print(kp1_np)
        return{
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': img1,
            'image1': img2,
            'all_matches': list(all_matches),
            'file_name': file_name
        }
