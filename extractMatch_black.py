import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model_testGCN import D2Net
#from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

import cv2
import matplotlib.pyplot as plt
import os
from sys import exit
from PIL import Image
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform


parser = argparse.ArgumentParser(description='Feature extraction script')
parser.add_argument('imgs', type=str, nargs=2)
parser.add_argument(
	'--preprocessing', type=str, default='caffe',
	help='image preprocessing (caffe or torch)'
)
parser.add_argument(
	'--model_file', type=str, default='models/d2_tf.pth',
	help='path to the full model'
)
# parser.add_argument(
# 	'--model_file', type=str, default='checkpoints/d2.08.pth',
# 	help='path to the full model'
# )
parser.add_argument(
	'--max_edge', type=int, default=1600,
	help='maximum image size at network input'
)
parser.add_argument(
	'--max_sum_edges', type=int, default=2800,
	help='maximum sum of image sizes at network input'
)

parser.add_argument(
	'--output_extension', type=str, default='.d2-net',
	help='extension for the output'
)
parser.add_argument(
	'--output_type', type=str, default='npz',
	help='output file type (npz or mat)'
)
parser.add_argument(
	'--multiscale', dest='multiscale', action='store_true',
	help='extract multiscale features'
)
parser.set_defaults(multiscale=False)
parser.add_argument(
	'--no-relu', dest='use_relu', action='store_false',
	help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)


def extract(file, args, model, device):
	image = imageio.imread(file)
	if len(image.shape) == 2:
		image = image[:, :, np.newaxis]
		image = np.repeat(image, 3, -1)

	resized_image = image
	if max(resized_image.shape) > args.max_edge:
		resized_image = scipy.misc.imresize(
			resized_image,
			args.max_edge / max(resized_image.shape)
		).astype('float')
	if sum(resized_image.shape[: 2]) > args.max_sum_edges:
		resized_image = scipy.misc.imresize(
			resized_image,
			args.max_sum_edges / sum(resized_image.shape[: 2])
		).astype('float')

	fact_i = image.shape[0] / resized_image.shape[0]
	fact_j = image.shape[1] / resized_image.shape[1]

	input_image = preprocess_image(
		resized_image,
		preprocessing=args.preprocessing
	)
	with torch.no_grad():
		if args.multiscale:
			keypoints, scores, descriptors = process_multiscale(
				torch.tensor(
					input_image[np.newaxis, :, :, :].astype(np.float32),
					device=device
				),
				model
			)
		else:
			keypoints, scores, descriptors = process_multiscale(
				torch.tensor(
					input_image[np.newaxis, :, :, :].astype(np.float32),
					device=device
				),
				model,
				scales=[1]
			)

	keypoints[:, 0] *= fact_i
	keypoints[:, 1] *= fact_j
	keypoints = keypoints[:, [1, 0, 2]]

	feat = {}
	feat['keypoints'] = keypoints
	feat['scores'] = scores
	feat['descriptors'] = descriptors

	return feat


def	drawMatches(file1, file2, feat1, feat2):
	image1 = np.array(Image.open(file1).convert('RGB'))
	image2 = np.array(Image.open(file2).convert('RGB'))

	matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
	print('Number of raw matches: %d.' % matches.shape[0])

	keypoints_left = feat1['keypoints'][matches[:, 0], : 2]
	keypoints_right = feat2['keypoints'][matches[:, 1], : 2]
	keypoints_left_new = []
	keypoints_right_new = []

	for i in range(keypoints_left.shape[0]):
		if np.all(image1[int(keypoints_left[i, 1]), int(keypoints_left[i, 0])]) == 0:
			continue
		keypoints_left_new.append(keypoints_left[i])
	keypoints_left_new = np.array(keypoints_left_new)[:]

	for i in range(0, keypoints_right.shape[0]):
		if np.all(image2[int(keypoints_right[i, 1]), int(keypoints_right[i, 0])]) == 0:
			continue
		keypoints_right_new.append(keypoints_right[i])
	keypoints_right_new = np.array(keypoints_right_new)[:]

	print(keypoints_left_new.shape, keypoints_right_new.shape)

	np.random.seed(0)
	model, inliers = ransac(
		(keypoints_left_new, keypoints_right_new),
		ProjectiveTransform, min_samples=4,
		residual_threshold=8, max_trials=10000
	)
	n_inliers = np.sum(inliers)
	print('Number of inliers: %d.' % n_inliers)

	inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left_new[inliers]]
	inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right_new[inliers]]
	placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

	image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None)
	#image3 = Image.fromarray(image3)
	#image3.save('/home/udit/d2-net/media/rcar_Pairs_overcast/9_extract.jpg')
	cv2.imwrite('/home/udit/d2-net/extract.jpg', image3)
	plt.figure(figsize=(20, 20))
	plt.imshow(image3)
	plt.axis('off')
	plt.show()


def	drawMatches2(file1, file2, feat1, feat2):
	image1 = np.array(Image.open(file1).convert('RGB'))
	image2 = np.array(Image.open(file2).convert('RGB'))
	des1 = feat1['descriptors']
	des2 = feat2['descriptors']

	matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
	keypoints_left = feat1['keypoints'][matches[:, 0], : 2].T
	keypoints_right = feat2['keypoints'][matches[:, 1], : 2].T
	keypoints_left_new = []
	keypoints_right_new = []

	#print(keypoints_left.shape)
	for i in range(keypoints_left.shape[1]):
		if np.all(image1[int(keypoints_left[1, i]), int(keypoints_left[0, i])]) == 0:
			continue
		keypoints_left_new.append(keypoints_left[:, i])
	keypoints_left_new = np.array(keypoints_left_new)[:38].T

	for i in range(keypoints_right.shape[1]):
		if np.all(image2[int(keypoints_right[1, i]), int(keypoints_right[0, i])]) == 0:
			continue
		keypoints_right_new.append(keypoints_right[:, i])
	keypoints_right_new = np.array(keypoints_right_new)[:38].T
	print(keypoints_right_new.shape)
	for i in range(keypoints_left_new.shape[1]):
		image1 = cv2.circle(image1, (int(keypoints_left_new[0, i]), int(keypoints_left_new[1, i])), 2, (0, 0, 255), 4)
	for i in range(keypoints_right_new.shape[1]):
		image2 = cv2.circle(image2, (int(keypoints_right_new[0, i]), int(keypoints_right_new[1, i])), 2, (0, 0, 255), 4)

	im4 = cv2.hconcat([image1, image2])
	print(keypoints_left_new.shape)
	for i in range(keypoints_left_new.shape[1]):
		im4 = cv2.line(im4, (int(keypoints_left_new[0, i]), int(keypoints_left_new[1, i])), (int(keypoints_right_new[0, i]) +  image1.shape[1], int(keypoints_right_new[1, i])), (0, 255, 0), 1)
	im5 = Image.fromarray(im4)
	im5.save('/home/udit/d2-net/extract_noRansac.jpg')
	cv2.imshow("Image_lines", im4)
	cv2.waitKey(0)


if __name__ == '__main__':
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	args = parser.parse_args()

	model = D2Net(
		config = {},
		model_file=args.model_file,
		use_relu=args.use_relu,
		use_cuda=use_cuda,
	)

	feat1 = extract(args.imgs[0], args, model, device)
	feat2 = extract(args.imgs[1], args, model, device)
	#print("Features extracted.")

	drawMatches(args.imgs[0], args.imgs[1], feat1, feat2)

	#drawMatches2(args.imgs[0], args.imgs[1], feat1, feat2)
