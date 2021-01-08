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
import re
from image import load_image
from shapely.geometry import Point, Polygon

def plotPts(trgPts):
	ax = plt.subplot(111)
	ax.plot(trgPts[:, 1], trgPts[:, 0], 'ro')
	plt.show()

def distance(co1, co2):
    return np.sqrt(pow(abs(co1[0] - co2[0]), 2) + pow(abs(co1[1] - co2[1]), 2))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
	parser.add_argument('--image_dir', type=str, help='Directory containing images', default='/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/mono_rear_rgb/')
	parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans', default='/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/lms_rear')
	parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses', default='/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/gps/ins.csv')
	parser.add_argument('--models_dir', type=str, help='Directory containing camera models', default='/scratch/udit/robotcar/robotcar-dataset-sdk-3.1/models')
	parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics', default='/scratch/udit/robotcar/robotcar-dataset-sdk-3.1/extrinsics/')
	parser.add_argument('--image_idx', type=int, help='Index of image to display')
	args = parser.parse_args()
	#depthFile = argv[2]

	srcPts = []
	trgPts = []

	uv, depth, timestamp, model = get_uvd(args.image_dir, args.laser_dir, args.poses_file, args.models_dir, args.extrinsics_dir, args.image_idx)

	image_path = os.path.join(args.image_dir, str(timestamp) + '.png')
	img = load_image(image_path, model)
	rgb = cv2.resize(img, (1280, 960))
	camera = re.search('(stereo|mono_(left|right|rear))', args.image_dir).group(0)

	if camera == 'mono_rear':
		## Rear points
		# bottom left -> bottom right -> top right -> top left

		pts = [(9, 787), (1179, 823), (809,556), (388,578)]
		# pts = [(9, 787), (1265,800), (911,536), (415,531)]
		## Rear camera intrinsics
		scalingFactor = 0.1
		focalLength = 400.000000
		centerX = 508.222931
		centerY = 498.187378
	elif camera == 'stereo':
		## Front points
		# bottom left -> bottom right -> top right -> top left

		#pts = [(279, 757), (1066,741), (868,601), (439,598)]
		pts = [(128,783), (1122,771), (899,651), (344,649)]

		## Front camera intrinsics
		scalingFactor = 1
		focalLength = 964.828979
		centerX = 643.788025
		centerY = 484.407990

	#rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

	for i in range(0, len(pts)):
		rgb = cv2.circle(rgb, (int(pts[i][0]), int(pts[i][1])), 1, (255, 0, 255), 2)

	cv2.imshow("Image", rgb)
	cv2.waitKey(0)

	poly = Polygon(pts)

	uv_new = []
	## 4 Pts
	# for i in range(uv.shape[1]):
	# 	uv_new.append((uv[0, i], uv[1, i]))
	#
	#
	# for coord in pts:
	# 	u_cl, v_cl = min(uv_new, key=lambda x:distance(x, coord))
	# 	index = uv_new.index((u_cl, v_cl))
	#
	# 	Z = depth[index]
	# 	X = (coord[0] - centerX) * Z / focalLength
	# 	Y = (coord[1] - centerY) * Z / focalLength
	#
	# 	trgPts.append((X, Z))
	#
	# trgPts = np.array(trgPts)
	# srcPts = np.array(pts)

	## All pts on road
	# for i in range(uv.shape[1]):
	# 	x, y = uv[0, i], uv[1, i]
	# 	if(pts[3][0]<x<pts[2][0] and pts[2][1]<y<pts[0][1]):
	# 		uv_new.append((x, y, i))

	for i in range(uv.shape[1]):
		p = Point(uv[0, i], uv[1, i])
		x, y = uv[0, i], uv[1, i]
		if p.within(poly):
			uv_new.append((x, y, i))

	for u1, v1, idx in uv_new:
		Z = depth[idx] / scalingFactor
		X = (u1 - centerX) * Z / focalLength
		Y = (v1 - centerY) * Z / focalLength
		trgPts.append((X, Z))

	trgPts = np.array(trgPts)
	srcPts = np.array(uv_new)[:,0:2]

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
	for i in range(0, len(srcPts)):
		rgb = cv2.circle(rgb, (int(srcPts[i, 0]), int(srcPts[i, 1])), 1, (0, 0, 255), 2)

	cv2.imshow("Image", rgb)
	cv2.waitKey(0)

	for i in range(0, trgPts.shape[0]):
		rgb = cv2.circle(rgb, (int(trgPts[i, 0]), int(trgPts[i, 1])), 1, (50, 255, 50), 2)
	cv2.imshow("Image", rgb)
	cv2.waitKey(0)


	homographyMat, status = cv2.findHomography(srcPts, trgPts)
	orgImg = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
	warpImg = cv2.warpPerspective(orgImg, homographyMat, (trgSize, trgSize))

	cv2.imshow("Warped", warpImg)
	cv2.waitKey(0)

	#cv2.imwrite("/home/udit/d2-net/media/rcar_samples/4f_trial.png", img)
	#warpImg = Image.fromarray(warpImg)
	if camera == 'mono_rear':
		cv2.imwrite("/home/udit/d2-net/media/rcar_Pairs_overcast/10_rtop.png", warpImg)
		cv2.imwrite("/home/udit/d2-net/media/rcar_Pairs_overcast/10_r.png", img)
	elif camera == 'stereo':
		cv2.imwrite("/home/udit/d2-net/media/rcar_Pairs_overcast/10_ftop.png", warpImg)
		cv2.imwrite("/home/udit/d2-net/media/rcar_Pairs_overcast/10_f.png", img)
