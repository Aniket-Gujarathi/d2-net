import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
#from torchsummary import summary

import matplotlib.pyplot as plt
from lib.utils import imshow_image

#import kornia.geometry.transform as tgm

from copy import deepcopy
from pathlib import Path

class DenseFeatureExtractionModule(nn.Module):
	def __init__(self, finetune_feature_extraction=False, use_cuda=False):
		super(DenseFeatureExtractionModule, self).__init__()

		model = models.vgg16()
		vgg16_layers = [
			'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
			'pool1',
			'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
			'pool2',
			'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
			'pool3',
			'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
			'pool4',
			'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
			'pool5'
		]
		conv4_3_idx = vgg16_layers.index('conv4_3')

		self.model = nn.Sequential(
			*list(model.features.children())[: conv4_3_idx + 1]
		)

		self.num_channels = 512

		# Fix forward parameters
		for param in self.model.parameters():
			param.requires_grad = False
		if finetune_feature_extraction:
			# Unlock conv4_3
			for param in list(self.model.parameters())[-2 :]:
				param.requires_grad = True

#		if use_cuda:
#			self.model = self.model.cuda()

	def return_mod(self):
		return self.model

	def forward(self, batch):
		output = self.model(batch)
		return output


class SoftDetectionModule(nn.Module):
	def __init__(self, soft_local_max_size=3):
		super(SoftDetectionModule, self).__init__()

		self.soft_local_max_size = soft_local_max_size

		self.pad = self.soft_local_max_size // 2

	def forward(self, batch):
		b = batch.size(0)

		batch = F.relu(batch)

		max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
		exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
		sum_exp = (
			self.soft_local_max_size ** 2 *
			F.avg_pool2d(
				F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
				self.soft_local_max_size, stride=1
			)
		)
		local_max_score = exp / (sum_exp)

		depth_wise_max = torch.max(batch, dim=1)[0]
		depth_wise_max_score = batch / (depth_wise_max.unsqueeze(1))

		all_scores = local_max_score * depth_wise_max_score
		score = torch.max(all_scores, dim=1)[0]

		score = score / (torch.sum(score.view(b, -1), dim=1).view(b, 1, 1))

		return score

def MLP(mod, channels: list, do_bn=True):
	""" Multi-layer perceptron """
	n = len(channels)
	layers = []
	for i in range(1, n):
		layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=1, bias=True))
		if i < (n-1):
			if do_bn:
				layers.append(nn.BatchNorm2d(channels[i]))
			layers.append(nn.ReLU())
	mod = nn.Sequential(mod, *layers)

	return nn.Sequential(*self.layers), mod


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, mod, feature_dim, layers):
        super().__init__()
		self.mod = mod
        self.encoder, self.mod = MLP(self.mod, [3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))

	def return_mod(self):
		return self.mod


def attention(query, key, value):
	dim = query.shape[1]
	scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
	prob = torch.nn.functional.softmax(scores, dim=-1)
	return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
	""" Multi-head attention to increase model expressivitiy """
	def __init__(self, num_heads: int, d_model: int, mod):
		super().__init__()
		self.mod = mod
		assert d_model % num_heads == 0
		self.dim = d_model // num_heads
		self.num_heads = num_heads
		self.merge = nn.Conv2d(d_model, d_model, kernel_size=1)
		self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
		self.mod = nn.Sequential(self.mod, self.proj, self.merge)

	def return_mod(self):
		return self.mod

	def forward(self, query, key, value):
		batch_dim = query.size(0)
		query, key, value = [l(x) for l, x in zip(self.proj, (query, key, value))]
		#.view(batch_dim, self.dim, self.num_heads, -1)
		x, _ = attention(query, key, value)
		return self.merge(x.contiguous())
		#.view(batch_dim, self.dim*self.num_heads, -1)

class AttentionalPropagation(nn.Module):
	def __init__(self, feature_dim: int, num_heads: int, mod):
		super().__init__()
		self.mod = mod
		self.attn = MultiHeadedAttention(num_heads, feature_dim, self.mod)
		self.mod = self.attn.return_mod()
		self.mlp, self.mod = MLP(self.mod, [feature_dim*2, feature_dim*2, feature_dim])

		nn.init.constant_(self.mlp[-1].bias, 0.0)

	def return_mod(self):
		return self.mod

	def forward(self, x, source):
		message = self.attn(x, source, source)
		return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
	def __init__(self, mod, feature_dim: int, layer_names: list):
		super().__init__()
		self.mod = mod
		self.attn = AttentionalPropagation(feature_dim, 4, self.mod)
		self.layers = nn.ModuleList([
            self.attn
            for _ in range(len(layer_names))])
		self.names = layer_names
		self.mod = self.attn.return_mod()

	def return_mod(self):
		return self.mod

	def forward(self, desc0, desc1):
		for layer, name in zip(self.layers, self.names):
			if name == 'cross':
				src0, src1 = desc1, desc0
				#print('cross')
			else:  # if name == 'self':
				#print('self')
				src0, src1 = desc0, desc1
			delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
			desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
		return desc0, desc1

class Align(nn.Module):
	def __init__(self):
		super(Align, self).__init__()

		# Resnet-18 features
		# Predict 8 points (4 pixels) per homography matrix

		points_src = torch.FloatTensor([[
			[190,210],[455,210],[633,475],[0,475],
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
		self.H2 = flipH @ cropH


	def forward(self, img1, img2):
		img_warp1 = tgm.warp_perspective(img1, self.H1, dsize=(400, 400))
		img_warp2 = tgm.warp_perspective(img2, self.H2, dsize=(400, 400))

		return img_warp1, img_warp2, self.H1, self.H2

class D2Net(nn.Module):

	default_config = {
        'descriptor_dim': 512,
        'GNN_layers': ['self', 'cross'] * 9,
	}

	def __init__(self, config, model_file=None, use_cuda=False):
		super(D2Net, self).__init__()

		self.config = {**self.default_config, **config}
		self.dense_feature_extraction = DenseFeatureExtractionModule(
			finetune_feature_extraction=True,
			use_cuda=use_cuda
		)
		self.mod = self.dense_feature_extraction.return_mod()
		self.gnn = AttentionalGNN(self.mod,
            self.config['descriptor_dim'], self.config['GNN_layers'])

		self.mod = self.gnn.return_mod()

		self.final_proj = nn.Conv2d(
        	self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

		self.mod = nn.Sequential(self.mod, self.final_proj)

		self.detection = SoftDetectionModule()


		if model_file is not None:
			if use_cuda:
				self.load_state_dict(torch.load(model_file)['model'], strict=False)
			else:
				self.load_state_dict(torch.load(model_file, map_location='cpu')['model'], strict=False)

		if use_cuda:
			self.mod = self.mod.cuda()
			print(self.mod, (3, 640, 480))


	def forward(self, batch):
		b = batch['image1'].size(0)
		dense_features = self.dense_feature_extraction(
			torch.cat([batch['image1'], batch['image2']], dim=0)
		)
		dense_features1 = dense_features[: b, :, :, :]
		dense_features2 = dense_features[b :, :, :, :]

		#dense_features1 = dense_features1.reshape([1, 512, -1])
		#dense_features2 = dense_features2.reshape([1, 512, -1])

		desc0, desc1 = self.gnn(dense_features1, dense_features2)

		mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
		#print(mdesc0.size())
		dense_features = torch.cat([mdesc0, mdesc1], dim=0)

		dense_features1 = dense_features[: b, :, :, :]
		dense_features2 = dense_features[b :, :, :, :]

		scores = self.detection(dense_features)
		#print(mdesc0.size())
		scores1 = scores[: b, :, :]
		scores2 = scores[b :, :, :]



		return {
			'dense_features1': dense_features1,
			#'dense_features1' : mdesc0,
			'scores1': scores1,
			'dense_features2': dense_features2,
			#'dense_features2': mdesc1,
			'scores2': scores2
		}

class D2NetAlign(nn.Module):
	default_config = {
        'descriptor_dim': 512,
        'GNN_layers': ['self', 'cross'] * 9,
		'keypoint_encoder': [32, 64, 128, 256],
	}

	def __init__(self, config, model_file=None, use_cuda=False):
		super(D2NetAlign, self).__init__()

		self.config = {**self.default_config, **config}
		self.alignment = Align()

		self.dense_feature_extraction = DenseFeatureExtractionModule(
			finetune_feature_extraction=True,
			use_cuda=use_cuda
		)

		self.mod = self.dense_feature_extraction.return_mod()

		self.kenc = KeypointEncoder(self.mod,
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

		self.gnn = AttentionalGNN(self.mod,
            self.config['descriptor_dim'], self.config['GNN_layers'])

		self.mod = self.gnn.return_mod()

		self.final_proj = nn.Conv2d(
        	self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

		self.mod = nn.Sequential(self.mod, self.final_proj)

		self.detection = SoftDetectionModule()

		if model_file is not None:
			if use_cuda:
				self.load_state_dict(torch.load(model_file)['model'], strict=False)
			else:
				self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

		if use_cuda:
			self.mod = self.mod.cuda()
			print(self.mod, (3, 640, 480))


	def display(self, img1, img2):
		plt.figure()

		im1 = imshow_image(
			img1[0].cpu().numpy(),
			preprocessing='caffe'
		)

		im2 = imshow_image(
			img2[0].cpu().numpy(),
			preprocessing='caffe'
		)

		plt.subplot(1, 2, 1)
		plt.imshow(im1)
		plt.axis('off')

		plt.subplot(1, 2, 2)
		plt.imshow(im2)
		plt.axis('off')

		plt.show()

		exit(1)


	def forward(self, batch):
		b = batch['image1'].size(0)

		img_warp1, img_warp2, H1, H2 = self.alignment(batch['image1'], batch['image2'])

		dense_features = self.dense_feature_extraction(
			torch.cat([img_warp1, img_warp2], dim=0)
		)
		dense_features1 = dense_features[: b, :, :, :]
		dense_features2 = dense_features[b :, :, :, :]
		print(dense_features1.size())
		# self.display(img_warp1, img_warp2)

		# dense_features = self.dense_feature_extraction(
		# 	torch.cat([batch['image1'], batch['image2']], dim=0)
		# )
		desc0, desc1 = self.gnn(dense_features1, dense_features2)

		mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
		#print(mdesc0.size())
		dense_features = torch.cat([mdesc0, mdesc1], dim=0)

		scores = self.detection(dense_features)

		dense_features1 = dense_features[: b, :, :, :]
		dense_features2 = dense_features[b :, :, :, :]

		scores1 = scores[: b, :, :]
		scores2 = scores[b :, :, :]

		return {
			'dense_features1': dense_features1,
			'scores1': scores1,
			'dense_features2': dense_features2,
			'scores2': scores2,
			'H1': H1,
			'H2': H2
		}
