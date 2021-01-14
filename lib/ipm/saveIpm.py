import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import perspective, Plane, load_camera_params, bilinear_sampler
from sys import argv, exit
from tqdm import tqdm
import os


def ipm_from_parameters(image, xyz, K, RT):
	P = K @ RT
	pixel_coords = perspective(xyz, P, TARGET_H, TARGET_W)
	image2 = bilinear_sampler(image, pixel_coords)
	return image2.astype(np.uint8)


def getImageFiles(rootDir):
	imgFiles = os.listdir(rootDir)
	imgFiles = [os.path.join(rootDir, img) for img in imgFiles]

	return imgFiles


if __name__ == '__main__':
	rootDir = argv[1]
	cfgFile = argv[2]

	imgFiles = getImageFiles(rootDir)

	TARGET_H, TARGET_W = 1024, 1024
	extrinsic, intrinsic = load_camera_params(cfgFile)
	plane = Plane(20, -25, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.035)

	i = 1
	for imgFile in tqdm(imgFiles, total=len(imgFiles)):
		image = cv2.cvtColor(cv2.imread(imgFile), cv2.COLOR_BGR2RGB)
		
		warped = ipm_from_parameters(image, plane.xyz, intrinsic, extrinsic)

		cv2.imwrite(imgFile, cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

		# outFile = os.path.join(rootDir, str(i) + ".png")
		# cv2.imwrite(outFile, cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
		# i += 1


		# fig, ax = plt.subplots(1, 2)
		# ax[0].imshow(image)
		# ax[0].set_title('Front View')
		# ax[1].imshow(warped)
		# ax[1].set_title('IPM')
		# plt.show()

		# exit(1)
