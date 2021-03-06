import open3d as o3d
from sys import argv, exit
from PIL import Image
import math
import numpy as np
import copy
import cv2
import time
import matplotlib.pyplot as plt


def display(pcd, T=np.identity(4)):
	axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
	axis.transform(T)

	o3d.visualization.draw_geometries([pcd, axis])


def getPointCloud(rgbFile, depthFile, segmFile):
	#max_thresh = 3
	max_thresh = 3
	#min_thresh = 1.5
	min_thresh = 1.5
	depth = np.load(depthFile)
	rgb = Image.open(rgbFile)
	segm = Image.open(segmFile)
	road = rgb

	print(rgb.size)
	print(segm.size)
	print(np.unique(depth))

	points = []
	colors = []
	srcPxs = []
	#road_img = np.zeros((depth.shape[1], depth.shape[0], 3))
	for v in range(depth.shape[0]):
		for u in range(depth.shape[1]):
			#print(segm[v, u], v, u)
			if segm.getpixel((u, v))[0] == 255:
				#road_img[u, v] = rgb.getpixel((u, v))
				Z = depth[v, u] / scalingFactor
				if Z==0: continue
				if (Z > max_thresh or Z < min_thresh): continue

				X = (u - centerX) * Z / focalLength
				Y = (v - centerY) * Z / focalLength

				srcPxs.append((u, v))
				points.append((X, Y, Z))
				colors.append(rgb.getpixel((u, v)))
			else:
				print('afd')
				road.putpixel((u, v), (0, 0, 0))

	road.show()

	srcPxs = np.asarray(srcPxs).T
	points = np.asarray(points)
	colors = np.asarray(colors)

	pcd = o3d.geometry.PointCloud()

	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)

	downpcd = pcd.voxel_down_sample(voxel_size=0.03)

	return pcd, srcPxs, road


def rotationMatrixFromVectors(vec1, vec2):
	a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	rotationMatrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
	return rotationMatrix


def getNormals(pcd):
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	pcd.orient_normals_towards_camera_location()

	normals = np.asarray(pcd.normals)
	surfaceNormal = np.mean(normals, axis=0)

	return surfaceNormal


def getPointsInCamera(pcd, T):
	pcd.transform(np.linalg.inv(T))

	mean = np.mean(np.asarray(pcd.points), axis=0)
	TCent = np.identity(4)
	TCent[0, 3] = mean[0]
	#TCent[2, 3] -= 0.01
 	## front
	TCent[1, 3] = mean[1]
	## rear
	#TCent[1, 3] = mean[1] - 0.8
	display(pcd, TCent)
	pcd.transform(np.linalg.inv(TCent))

	return pcd


def extractPCD(pcd):
	pcdPoints = np.asarray(pcd.points)
	pcdColor = np.asarray(pcd.colors).T

	return pcdPoints, pcdColor


def getPixels(pcdPoints):
	# focalLength = 400.000000
	# centerX = 508.222931
	# centerY = 498.187378
	# scalingFactor = 0.1
	K = np.array([[focalLength, 0, centerX], [0, focalLength, centerY], [0, 0, 1]])
	pxh = K @ pcdPoints.T

	pxh[0, :] = pxh[0, :]/pxh[2, :]
	pxh[1, :] = pxh[1, :]/pxh[2, :]
	pxh[2, :] = pxh[2, :]/pxh[2, :]

	return pxh[0:2, :]


def resizePxs(pxs, imgSize):
	minX = np.min(pxs[0, :])
	minY = np.min(pxs[1, :])

	if(minX < 0):
		pxs[0, :] += (np.abs(minX) + 2)
	if(minY < 0):
		pxs[1, :] += (np.abs(minY) + 2)

	maxX = np.max(pxs[0, :])
	maxY = np.max(pxs[1, :])

	ratioX = imgSize/maxX
	ratioY = imgSize/maxY

	pxs[0, :] *= ratioX
	pxs[1, :] *= ratioY

	return pxs


def pxsToImg(pxs, pcdColor, imgSize):
	height = imgSize; width = imgSize

	img = np.zeros((height, width, 3), np.uint8)

	for i in range(pxs.shape[1]):
		r = int(pxs[1, i]); c = int(pxs[0, i])
		if(r<height and c<width and r>0 and c>0):
			red = 255*pcdColor[0, i]; green = 255*pcdColor[1, i]; blue = 255*pcdColor[2, i]
			img[r, c] = (blue, green, red)

	return img


def getImg(pcd, T):
	pcd = getPointsInCamera(pcd, T)

	pcdPoints, pcdColor = extractPCD(pcd)

	# print(np.mean(pcdPoints, axis=0))

	pxs = getPixels(pcdPoints)

	imgSize = 800
	pxs = resizePxs(pxs, imgSize)

	img = pxsToImg(pxs, pcdColor, imgSize)

	cv2.imshow("image", img)
	cv2.waitKey(0)


def getImgHomo(pcd, T, srcPxs, rgbFile, road_img):
	pcd = getPointsInCamera(pcd, T)

	pcdPoints, pcdColor = extractPCD(pcd)
	# print(np.mean(pcdPoints, axis=0))

	trgPxs = getPixels(pcdPoints)

	imgSize = 800
	trgPxs = resizePxs(trgPxs, imgSize)

	img = pxsToImg(trgPxs, pcdColor, imgSize)

	cv2.imshow("image", img)
	cv2.waitKey(0)

	homographyMat, status = cv2.findHomography(srcPxs.T, trgPxs.T)
	orgImg = cv2.cvtColor(np.array(road_img), cv2.COLOR_BGR2RGB)
	warpImg = cv2.warpPerspective(orgImg, homographyMat, (imgSize, imgSize))


	cv2.imshow("Image", warpImg)
	cv2.waitKey(0)


def getPlane(pcd):
	planeModel, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
	[a, b, c, d] = planeModel
	surfaceNormal = [a, b, c]
	planeDis = d
	# print("Plane equation: {}x + {}y + {}z + {} = 0".format(a, b, c, d))
	return surfaceNormal, planeDis


if __name__ == '__main__':
	rgbFile = argv[1]
	depthFile = argv[2]
	segmFile = argv[3]

	focalLength = 964.828979
	centerX = 643.788025
	centerY = 484.407990
	scalingFactor = 0.1

	# focalLength = 400.000000
	# centerX = 508.222931
	# centerY = 498.187378
	# scalingFactor = 0.1

	pcd, srcPxs, road = getPointCloud(rgbFile, depthFile, segmFile)

	# surfaceNormal = -getNormals(pcd)
	surfaceNormal, planeDis = getPlane(pcd)

	zAxis = np.array([0, 0, 1])
	rotationMatrix = rotationMatrixFromVectors(zAxis, surfaceNormal)
	T = np.identity(4)
	T[0:3, 0:3] = rotationMatrix

	display(pcd)
	display(pcd, T)

	#getImg(pcd, T)
	getImgHomo(pcd, T, srcPxs, rgbFile, road)
	surfaceNormal, planeDis = getPlane(pcd)
	print(planeDis)
