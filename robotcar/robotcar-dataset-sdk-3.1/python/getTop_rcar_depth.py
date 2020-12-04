import open3d as o3d
from sys import argv, exit
from PIL import Image
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from project_laser_into_camera import get_uvd
import argparse
import os
from image import load_image

def plotPts(trgPts):
	ax = plt.subplot(111)
	ax.plot(trgPts[:, 1], trgPts[:, 0], 'ro')
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
	parser.add_argument('--image_dir', type=str, help='Directory containing images')
	parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans')
	parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses')
	parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
	parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
	parser.add_argument('--image_idx', type=int, help='Index of image to display')
	args = parser.parse_args()
	#depthFile = argv[2]

	srcPts = []
	trgPts = []

	#depth = np.load(depthFile)
	#img = Image.open(rgbFile)
	uv, depth, timestamp, model = get_uvd(args.image_dir, args.laser_dir, args.poses_file, args.models_dir, args.extrinsics_dir, args.image_idx)
	#uv = uv.astype(int)

	image_path = os.path.join(args.image_dir, str(timestamp) + '.png')
	img = load_image(image_path, model)
	# bottom left -> bottom right -> top right -> top left
	pts = [[253.15053852855385, 283.0046263264299], [422.34675570493152, 419.66893979729889], [421.74815046405035, 427.32752221557985], [253.82674781369053, 270.16157887964357]]

	rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
	# for i in range(0, len(pts)):
	# 	rgb = cv2.circle(rgb, (pts[i][0], pts[i][1]), 1, (0, 0, 255), 2)
	# cv2.imshow("Image", rgb)
	# cv2.waitKey(0)

	scalingFactor = 1000.0
	focalLength = 964.828979
	centerX = 643.788025
	centerY = 484.407990

	print(uv.shape)
	uv_new = []
	for i in range(uv.shape[1]):
		uv_new.append((uv[0, i], uv[1, i]))
	print(uv_new)

	for u1, v1 in pts:
		index = uv_new.index((u1, v1))
		print(index)
		Z = depth[index]/scalingFactor
		X = (u1 - centerX) * Z / focalLength
		Y = (v1 - centerY) * Z / focalLength

		trgPts.append((X, Z))

	trgPts = np.array(trgPts)
	srcPts = np.array(pts)

	# 3D point adjustment to opencv plane
	# trgPts[2, 1], trgPts[3, 1] = -trgPts[2, 1], -trgPts[3, 1]
	# trgPts[:, 1] = -trgPts[:, 1]

	# Making coordinates positive
	minX = np.min(trgPts[:, 0])
	minY = np.min(trgPts[:, 1])

	if(minX < 0):
		trgPts[:, 0] += (np.abs(minX) + 2)
	if(minY < 0):
		trgPts[:, 1] += (np.abs(minY) + 2)

	# Scaling coordinates
	maxX = np.max(trgPts[:, 0])
	maxY = np.max(trgPts[:, 1])

	trgSize = 400
	ratioX = trgSize/maxX
	ratioY = trgSize/maxY

	trgPts[:, 0] *= ratioX
	trgPts[:, 1] *= ratioY

	# print(trgPts)
	# plotPts(trgPts)

	for i in range(0, trgPts.shape[0]):
		rgb = cv2.circle(rgb, (int(trgPts[i, 0]), int(trgPts[i, 1])), 1, (50, 255, 50), 2)
	cv2.imshow("Image", rgb)
	cv2.waitKey(0)

	homographyMat, status = cv2.findHomography(srcPts, trgPts)
	orgImg = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
	warpImg = cv2.warpPerspective(orgImg, homographyMat, (trgSize, trgSize))

	cv2.imshow("Warped", warpImg)
	cv2.waitKey(0)
	cv2.imwrite("/home/udit/d2-net/media/get_TOP/gazebo_r.png", warpImg)
	cv2.imwrite("/home/udit/d2-net/media/get_TOP/gazebo_r_persp.png", rgb)
