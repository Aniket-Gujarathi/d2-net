import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import cv2
from sys import exit

import torch
import torch.nn.functional as F

from lib.utils import (
	grid_positions,
	upscale_positions,
	downscale_positions,
	savefig,
	imshow_image
)
from lib.exceptions import NoGradientError, EmptyTensorError
import torchgeometry as tgm


matplotlib.use('Agg')

def loss_function_PT(model, batch, device, margin=1, safe_radius=4, scaling_steps=3, plot=False):
	output = model({
		'image1': batch['image1'].to(device),
		'image2': batch['image2'].to(device)
	})

	loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
	has_grad = False

	n_valid_samples = 0
	for idx_in_batch in range(batch['image1'].size(0)):
		# Network output
		dense_features1 = output['dense_features1'][idx_in_batch]
		c, h1, w1 = dense_features1.size()
		scores1 = output['scores1'][idx_in_batch].view(-1)

		dense_features2 = output['dense_features2'][idx_in_batch]
		_, h2, w2 = dense_features2.size()
		scores2 = output['scores2'][idx_in_batch]


		all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
		descriptors1 = all_descriptors1

		all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)

		fmap_pos1 = grid_positions(h1, w1, device)

		# hOrig, wOrig = int(batch['image1'].shape[2]/8), int(batch['image1'].shape[3]/8)
		# fmap_pos1Orig = grid_positions(hOrig, wOrig, device)
		#pos1 = upscale_positions(fmap_pos1Orig, scaling_steps=scaling_steps)

		# get correspondences
		img1 = imshow_image(
						batch['image1'][idx_in_batch].cpu().numpy(),
						preprocessing=batch['preprocessing']
							)
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
		img2 = imshow_image(
						batch['image2'][idx_in_batch].cpu().numpy(),
						preprocessing=batch['preprocessing']
							)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

		pos1, pos2 = getCorr(img1, img2)
		if(len(pos1) == 0 or len(pos2) == 0):
			continue

		pos1 = torch.from_numpy(pos1.astype(np.float32)).to(device)
		pos2 = torch.from_numpy(pos2.astype(np.float32)).to(device)
		img1 = torch.from_numpy(img1.astype(np.float32)).to(device)
		img2 = torch.from_numpy(img2.astype(np.float32)).to(device)

		# print('p1', pos1.size())
		# print('p2',pos2.size())
		ids = idsAlign(pos1, device, h1, w1)
		#print('ids', ids)

		fmap_pos1 = fmap_pos1[:, ids]
		descriptors1 = descriptors1[:, ids]
		scores1 = scores1[ids]

		# Skip the pair if not enough GT correspondences are available

		if ids.size(0) < 128:
			print('hi', ids.size(0))
			continue

		# Descriptors at the corresponding positions
		fmap_pos2 = torch.round(
			downscale_positions(pos2, scaling_steps=scaling_steps)
		).long()

		descriptors2 = F.normalize(
			dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
			dim=0
		)

		positive_distance = 2 - 2 * (
			descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)
		).squeeze()

		#positive_distance = getPositiveDistance(descriptors1, descriptors2)

		all_fmap_pos2 = grid_positions(h2, w2, device)
		position_distance = torch.max(
			torch.abs(
				fmap_pos2.unsqueeze(2).float() -
				all_fmap_pos2.unsqueeze(1)
			),
			dim=0
		)[0]
		is_out_of_safe_radius = position_distance > safe_radius

		distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)
		#distance_matrix = getDistanceMatrix(descriptors1, all_descriptors2)

		negative_distance2 = torch.min(
			distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
			dim=1
		)[0]

		#negative_distance2 = semiHardMine(distance_matrix, is_out_of_safe_radius, positive_distance, margin)

		all_fmap_pos1 = grid_positions(h1, w1, device)
		position_distance = torch.max(
			torch.abs(
				fmap_pos1.unsqueeze(2).float() -
				all_fmap_pos1.unsqueeze(1)
			),
			dim=0
		)[0]
		is_out_of_safe_radius = position_distance > safe_radius

		distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
		#distance_matrix = getDistanceMatrix(descriptors2, all_descriptors1)

		negative_distance1 = torch.min(
			distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
			dim=1
		)[0]

		#negative_distance1 = semiHardMine(distance_matrix, is_out_of_safe_radius, positive_distance, margin)

		diff = positive_distance - torch.min(
			negative_distance1, negative_distance2
		)
		print('positive_distance_min', torch.min(positive_distance))
		print('negative_distance1_min', torch.min(negative_distance1))
		print('positive_distance_max', torch.max(positive_distance))
		print('negative_distance1_max', torch.max(negative_distance1))
		print('positive_distance_mean', torch.mean(positive_distance))
		print('negative_distance1_mean', torch.mean(negative_distance1))

		scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]]

		loss = loss + (
			torch.sum(scores1 * scores2 * F.relu(margin + diff)) /
			(torch.sum(scores1 * scores2) )
		)

		print('scores1_min', torch.min(scores1))
		print('scores1_max', torch.max(scores1))
		print('scores1_mean', torch.mean(scores1))

		has_grad = True
		n_valid_samples += 1

		if plot and batch['batch_idx'] % batch['log_interval'] == 0:
			drawTraining(batch['image1'], batch['image2'], pos1, pos2, batch, idx_in_batch, output, save=True)
			#drawTraining(img_warp1, img_warp2, pos1, pos2, batch, idx_in_batch, output, save=True)

	if not has_grad:
		raise NoGradientError
	print('scores1', scores1)
	print('scores2', scores2)
	loss = loss / (n_valid_samples )

	return loss
	
def getCorr(img1, img2):
	im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
	im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

	surf = cv2.xfeatures2d.SURF_create(100)
	# surf = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = surf.detectAndCompute(im1,None)
	kp2, des2 = surf.detectAndCompute(im2,None)

	if(len(kp1) < 128 or len(kp2) < 128):
		return [], []

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key=lambda x:x.distance)

	if(len(matches) > 800):
		matches = matches[0:800]
	elif(len(matches) < 128):
		return [], []

	pos1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T
	pos2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T

	pos1[[0, 1]] = pos1[[1, 0]]
	pos2[[0, 1]] = pos2[[1, 0]]

	# for i in range(0, pos1.shape[1], 1):
	# 	im1 = cv2.circle(im1, (pos1[1, i], pos1[0, i]), 1, (0, 0, 255), 2)
	# for i in range(0, pos2.shape[1], 1):
	# 	im2 = cv2.circle(im2, (pos2[1, i], pos2[0, i]), 1, (0, 0, 255), 2)

	# im3 = cv2.hconcat([im1, im2])

	# for i in range(0, pos1.shape[1], 1):
	# 	im3 = cv2.line(im3, (int(pos1[1, i]), int(pos1[0, i])), (int(pos2[1, i]) +  im1.shape[1], int(pos2[0, i])), (0, 255, 0), 1)

	# im4 = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=2)
	# cv2.imshow('Image', im1)
	# cv2.imshow('Image2', im2)
	# cv2.imshow('Image3', im3)
	# cv2.imshow('Image4', im4)
	# cv2.waitKey(0)

	return pos1, pos2





def drawTraining(image1, image2, pos1, pos2, batch, idx_in_batch, output, save=False):
	pos1_aux = pos1.cpu().numpy()
	pos2_aux = pos2.cpu().numpy()

	k = pos1_aux.shape[1]
	col = np.random.rand(k, 3)
	n_sp = 4
	plt.figure()
	plt.subplot(1, n_sp, 1)
	im1 = imshow_image(
		image1[0].cpu().numpy(),
		preprocessing=batch['preprocessing']
	)
	plt.imshow(im1)
	plt.scatter(
		pos1_aux[1, :], pos1_aux[0, :],
		s=0.25**2, c=col, marker=',', alpha=0.5
	)
	plt.axis('off')
	plt.subplot(1, n_sp, 2)
	plt.imshow(
		output['scores1'][idx_in_batch].data.cpu().numpy(),
		cmap='Reds'
	)
	plt.axis('off')
	plt.subplot(1, n_sp, 3)
	im2 = imshow_image(
		image2[0].cpu().numpy(),
		preprocessing=batch['preprocessing']
	)
	plt.imshow(im2)
	plt.scatter(
		pos2_aux[1, :], pos2_aux[0, :],
		s=0.25**2, c=col, marker=',', alpha=0.5
	)
	plt.axis('off')
	plt.subplot(1, n_sp, 4)
	plt.imshow(
		output['scores2'][idx_in_batch].data.cpu().numpy(),
		cmap='Reds'
	)
	plt.axis('off')

	if(save == True):
		savefig('train_vis/%s.%02d.%02d.%d.png' % (
			'train' if batch['train'] else 'valid',
			batch['epoch_idx'],
			batch['batch_idx'] // batch['log_interval'],
			idx_in_batch
		), dpi=300)
	else:
		plt.show()

	plt.close()

	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

	for i in range(0, pos1_aux.shape[1], 5):
		im1 = cv2.circle(im1, (pos1_aux[1, i], pos1_aux[0, i]), 1, (0, 0, 255), 2)
	for i in range(0, pos2_aux.shape[1], 5):
		im2 = cv2.circle(im2, (pos2_aux[1, i], pos2_aux[0, i]), 1, (0, 0, 255), 2)

	im3 = cv2.hconcat([im1, im2])

	for i in range(0, pos1_aux.shape[1], 5):
		im3 = cv2.line(im3, (int(pos1_aux[1, i]), int(pos1_aux[0, i])), (int(pos2_aux[1, i]) +  im1.shape[1], int(pos2_aux[0, i])), (0, 255, 0), 1)

	if(save == True):
		cv2.imwrite('train_vis/%s.%02d.%02d.%d.png' % (
			'train_corr' if batch['train'] else 'valid',
			batch['epoch_idx'],
			batch['batch_idx'] // batch['log_interval'],
			idx_in_batch
		), im3)
	else:
		cv2.imshow('Image', im3)
		cv2.waitKey(0)



def idsAlign(pos1, device, h1, w1):
	pos1D = downscale_positions(pos1, scaling_steps=3)
	row = pos1D[0, :]
	col = pos1D[1, :]

	ids = []

	for i in range(row.shape[0]):

		index = ((w1) * (row[i])) + (col[i])
		ids.append(index)

	ids = torch.round(torch.Tensor(ids)).long().to(device)

	return ids


def semiHardMine(distance_matrix, is_out_of_safe_radius, positive_distance, margin):
	negative_distances = distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.

	negDist = []

	for i, row in enumerate(negative_distances):
		posDist = positive_distance[i]

		row = row[(posDist + margin > row) & (row > posDist)]

		if(row.size(0) == 0):
			negDist.append(negative_distances[i, 0])
		else:
			perm = torch.randperm(row.size(0))
			negDist.append(row[perm[0]])

	negDist = torch.Tensor(negDist).to(positive_distance.device)

	return negDist


def getPositiveDistance(descriptors1, descriptors2):
	positive_distance = torch.norm(descriptors1 - descriptors2, dim=0)

	return positive_distance


def getDistanceMatrix(descriptors1, all_descriptors2):
	d1 = descriptors1.t().unsqueeze(0)
	all_d2 = all_descriptors2.t().unsqueeze(0)
	distance_matrix = torch.cdist(d1, all_d2, p=2).squeeze()

	return distance_matrix

