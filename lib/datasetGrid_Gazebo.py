import numpy as np
from PIL import Image
import os
from sys import exit, argv
import csv
import torch
import torchgeometry as tgm
from torch.utils.data import Dataset
from lib.utils import preprocess_image, grid_positions, upscale_positions
import cv2
from tqdm import tqdm

np.random.seed(0)


class PhotoTourism(Dataset):
	def __init__(self, rootDir, imgData, preprocessing):
		self.rootDir = rootDir
		self.imgData = imgData
		self.preprocessing = preprocessing
		self.dataset = []

		# points_src = torch.FloatTensor([[
		# 	[190,210],[455,210],[633,475],[0,475],
		# ]]).cuda()
		points_src = torch.FloatTensor([[
			[149, 157],[447, 166],[311, 151],[322, 265],
		]]).cuda()
		points_dst = torch.FloatTensor([[
			[0, 0], [399, 0], [399, 399], [0, 399],
		]]).cuda()
		cropH = tgm.get_perspective_transform(points_src, points_dst)

		points_src = torch.FloatTensor([[
			[0, 0], [400, 0], [400, 400], [0, 400]
			]]).cuda()
		points_dst = torch.FloatTensor([[
			[400, 400], [0, 400], [0, 0], [400, 0]
			]]).cuda()
		flipH = tgm.get_perspective_transform(points_src, points_dst)

		self.H1 = cropH
		self.H2 = cropH

	def getImageFiles(self):
		imgFiles = os.listdir(self.rootDir)
		imgFiles = [os.path.join(self.rootDir, img) for img in imgFiles]

		return imgFiles

	def getimgPair(self):
		imgFiles = []

		with open(self.imgData) as csvFile:
			csvReader = csv.reader(csvFile, delimiter=',')

			for i, row in enumerate(csvReader):
				if(i == 0):
					continue
				else:
					imgFiles.append(row)
				#print(imgFiles)
		return imgFiles


	def imgRot(self, img1):
		img2 = img1.rotate(np.random.randint(low=90, high=270))
		# img2 = img1.rotate(np.random.randint(low=0, high=60))

		return img2

	def imgCrop(self, img1, cropSize=400):
		w, h = img1.size
		left = np.random.randint(low = 0, high = w - (cropSize + 10))
		upper = np.random.randint(low = 0, high = h - (cropSize + 10))

		cropImg = img1.crop((left, upper, left+cropSize, upper+cropSize))

		# cropImg = cv2.cvtColor(np.array(cropImg), cv2.COLOR_BGR2RGB)
		# cv2.imshow("Image", cropImg)
		# cv2.waitKey(0)

		return cropImg

	def getFloor(self, img1, img2):
		img_warp1 = tgm.warp_perspective(img1, self.H1, dsize=(400, 400))
		img_warp2 = tgm.warp_perspective(img2, self.H2, dsize=(400, 400))
		return img_warp1, img_warp2

	def getGrid(self, img1, img2, minCorr=10, scaling_steps=3, matcher="FLANN"):
		im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
		im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

		# surf = cv2.xfeatures2d.SURF_create(100)
		surf = cv2.xfeatures2d.SIFT_create()

		kp1, des1 = surf.detectAndCompute(img1,None)
		kp2, des2 = surf.detectAndCompute(img2,None)

		if(len(kp1) < minCorr or len(kp2) < minCorr):
			print('ha', len(kp1))
			return [], []

		if(matcher == "BF"):

			bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
			matches = bf.match(des1,des2)
			matches = sorted(matches, key=lambda x:x.distance)

		elif(matcher == "FLANN"):

			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks = 50)
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			matches = flann.knnMatch(des1,des2,k=2)
			good = []
			for m, n in matches:
				if m.distance < 0.7*n.distance:
					good.append(m)
			matches = good

		if(len(matches) > 800):
			matches = matches[0:800]
		elif(len(matches) < minCorr):
			return [], []

		# im4 = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=2)
		# cv2.imshow('Image4', im4)
		# cv2.waitKey(0)

		src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

		H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		if H is None:
			return [], []
		h1, w1 = int(im1.shape[0]/(2**scaling_steps)), int(im1.shape[1]/(2**scaling_steps))
		device = torch.device("cpu")

		fmap_pos1 = grid_positions(h1, w1, device)
		pos1 = upscale_positions(fmap_pos1, scaling_steps=scaling_steps).data.cpu().numpy()

		pos1[[0, 1]] = pos1[[1, 0]]

		ones = np.ones((1, pos1.shape[1]))

		pos1Homo = np.vstack((pos1, ones))
		pos2Homo = np.dot(H, pos1Homo)
		pos2Homo = pos2Homo/pos2Homo[2, :]
		pos2 = pos2Homo[0:2, :]

		pos1[[0, 1]] = pos1[[1, 0]]
		pos2[[0, 1]] = pos2[[1, 0]]
		pos1 = pos1.astype(np.float32)
		pos2 = pos2.astype(np.float32)

		ids = []
		for i in range(pos2.shape[1]):
			x, y = pos2[:, i]
			if(2 < x < (im1.shape[0]-2) and 2 < y < (im1.shape[1]-2)):
				ids.append(i)
		pos1 = pos1[:, ids]
		pos2 = pos2[:, ids]

		# for i in range(0, pos1.shape[1], 20):
		# 	im1 = cv2.circle(im1, (pos1[1, i], pos1[0, i]), 1, (0, 0, 255), 2)
		# for i in range(0, pos2.shape[1], 20):
		# 	im2 = cv2.circle(im2, (pos2[1, i], pos2[0, i]), 1, (0, 0, 255), 2)

		# im3 = cv2.hconcat([im1, im2])

		# for i in range(0, pos1.shape[1], 20):
		# 	im3 = cv2.line(im3, (int(pos1[1, i]), int(pos1[0, i])), (int(pos2[1, i]) +  im1.shape[1], int(pos2[0, i])), (0, 255, 0), 1)

		# cv2.imshow('Image', im1)
		# cv2.imshow('Image2', im2)
		# cv2.imshow('Image3', im3)
		# cv2.waitKey(0)

		return pos1, pos2

	def build_dataset(self, cropSize=400):
		print("Building Dataset.")
		#device = torch.device("cuda")
		imgFiles = self.getimgPair()

		for i in range(len(imgFiles)):
			rgbFile1, depthFile1, rgbFile2, depthFile2 = imgFiles[i]
			rgbFile1 = os.path.join(self.rootDir, rgbFile1)
			rgbFile2 = os.path.join(self.rootDir, rgbFile2)
			img1 = Image.open(rgbFile1)
			img2 = Image.open(rgbFile2)


			if(img1.mode != 'RGB'):
				img1 = img1.convert('RGB')
			# elif(img1.size[0] < cropSize or img1.size[1] < cropSize):
			# 	continue

			if(img2.mode != 'RGB'):
				img2 = img2.convert('RGB')
			# elif(img2.size[0] < cropSize or img2.size[1] < cropSize):
			# 	continue

			#img1 = self.imgCrop(img1, cropSize)
			#img2 = self.imgCrop(img2, cropSize)
			#img2 = self.imgRot(img1)

			img1 = np.array(img1)
			img2 = np.array(img2)

			img1 = torch.from_numpy(img1.astype(np.float32)).cuda().unsqueeze(0).permute(0, 3, 1, 2)
			img2 = torch.from_numpy(img2.astype(np.float32)).cuda().unsqueeze(0).permute(0, 3, 1, 2)

			img1, img2 = self.getFloor(img1, img2)

			img1 = img1.cpu().squeeze(0).permute(1, 2, 0)
			img2 = img2.cpu().squeeze(0).permute(1, 2, 0)

			img1 = np.array(img1).astype('uint8')
			img2 = np.array(img2).astype('uint8')
			#print('img1', img1.size)

			cv2.imwrite('/home/dhagash/d2-net/d2-net_udit/media/img1.jpg', img1)
			cv2.imwrite('/home/dhagash/d2-net/d2-net_udit/media/img2.jpg', img2)

			pos1, pos2 =  self.getGrid(img1, img2)

			if(len(pos1) == 0 or len(pos2) == 0):
				continue
			print('pos1', pos1)
			self.dataset.append((img1, img2, pos1, pos2))


	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		image1, image2, pos1, pos2 = self.dataset[idx]

		image1 = preprocess_image(image1, preprocessing=self.preprocessing)
		image2 = preprocess_image(image2, preprocessing=self.preprocessing)

		return {
			'image1': torch.from_numpy(image1.astype(np.float32)),
			'image2': torch.from_numpy(image2.astype(np.float32)),
			'pos1': torch.from_numpy(pos1.astype(np.float32)),
			'pos2': torch.from_numpy(pos2.astype(np.float32))
		}


if __name__ == '__main__':
	rootDir = argv[1]
	imgData = argv[2]

	training_dataset = PhotoTourism(rootDir, imgData, 'caffe')
	training_dataset.build_dataset()

	data = training_dataset[0]
	print(data['image1'].shape, data['image2'].shape, data['pos1'].shape, data['pos2'].shape)
