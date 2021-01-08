################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import argparse
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime as dt
from image import load_image
from camera_model import CameraModel
import cv2


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized


parser = argparse.ArgumentParser(description='Play back images from a given directory')

parser.add_argument('dir', type=str, help='Directory containing images.')
parser.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
parser.add_argument('--scale', type=float, default=1.0, help='(optional) factor by which to scale images before display')

args = parser.parse_args()

camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0)

timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps'))
if not os.path.isfile(timestamps_path):
  timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps')
  if not os.path.isfile(timestamps_path):
	  raise IOError("Could not find timestamps file")

model = None
if args.models_dir:
	model = CameraModel(args.models_dir, args.dir)

current_chunk = 0
timestamps_file = open(timestamps_path)
for i, line in enumerate(timestamps_file):
	#if(i < 1732):
	# if(i < 825):
	#     continue
	# print("{}th image.".format(i))

	tokens = line.split()
	datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
	chunk = int(tokens[1])

	filename = os.path.join(args.dir, tokens[0] + '.png')
	if not os.path.isfile(filename):
		if chunk != current_chunk:
			print("Chunk " + str(chunk) + " not found")
			current_chunk = chunk
		continue

	current_chunk = chunk

	# print(filename)

	img = load_image(filename, model)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	img = image_resize(img, height=300)
	cv2.imshow('Image', img)
	cv2.waitKey(1)

	# cv2.imwrite(filename, img)
	
	# plt.imshow(img)
	# plt.xlabel(datetime)
	# plt.xticks([])
	# plt.yticks([])
	# plt.pause(0.01)
