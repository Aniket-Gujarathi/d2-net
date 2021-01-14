import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import perspective, Plane, load_camera_params, bilinear_sampler
from sys import argv, exit


def ipm_from_parameters(image, xyz, K, RT, TARGET_H, TARGET_W):
	P = K @ RT
	pixel_coords = perspective(xyz, P, TARGET_H, TARGET_W)
	image2 = bilinear_sampler(image, pixel_coords)
	return image2.astype(np.uint8)

def ipm(imgFile, cfgFile):
	image = cv2.cvtColor(cv2.imread(imgFile), cv2.COLOR_BGR2RGB)
	TARGET_H, TARGET_W = 1024, 1024
	# TARGET_H, TARGET_W = 800, 800

	plane = Plane(20, -25, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.035)

	extrinsic, intrinsic = load_camera_params(cfgFile)

	warped1 = ipm_from_parameters(image, plane.xyz, intrinsic, extrinsic, TARGET_H, TARGET_W)

	return warped1

if __name__ == '__main__':
	imgFile = argv[1]
	cfgFile = argv[2]

	image = cv2.cvtColor(cv2.imread(imgFile), cv2.COLOR_BGR2RGB)
	TARGET_H, TARGET_W = 1024, 1024
	# TARGET_H, TARGET_W = 800, 800

	plane = Plane(20, -25, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.035)

	extrinsic, intrinsic = load_camera_params(cfgFile)

	warped1 = ipm_from_parameters(image, plane.xyz, intrinsic, extrinsic)
	print(warped1.shape)
	cv2.imwrite('/home/udit/d2-net/media/ipm/pair15b.png', cv2.cvtColor(warped1, cv2.COLOR_BGR2RGB))


	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(image)
	ax[0].set_title('Front View')
	ax[1].imshow(warped1)
	ax[1].set_title('IPM')
	plt.show()
