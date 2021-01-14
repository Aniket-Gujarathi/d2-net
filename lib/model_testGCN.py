import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import matplotlib.pyplot as plt
from lib.utils import imshow_image

from copy import deepcopy
from pathlib import Path

class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=1),
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        )
        self.num_channels = 512

        self.use_relu = use_relu

    def return_mod(self):
        return self.model

        # if use_cuda:
        #     self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        if self.use_relu:
            output = F.relu(output)
        return output

class MLP():

	def __init__(self, mod, channels: list, do_bn=True):
		""" Multi-layer perceptron """
		self.mod = mod
		n = len(channels)
		self.layers = []
		for i in range(1, n):
			self.layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=1, bias=True))
			if i < (n-1):
				if do_bn:
					self.layers.append(nn.BatchNorm2d(channels[i]))
				self.layers.append(nn.ReLU())
		self.mod = nn.Sequential(self.mod, *self.layers)

	def forward(self):
		return nn.Sequential(*self.layers)

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
		self.perc = MLP(self.mod, [feature_dim*2, feature_dim*2, feature_dim])
		self.mlp = self.perc.forward()
		nn.init.constant_(self.mlp[-1].bias, 0.0)
		self.mod = self.perc.return_mod()

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

class D2Net(nn.Module):
    default_config = {
        'descriptor_dim': 512,
        'GNN_layers': ['self', 'cross'] * 9,
	}
    def __init__(self, config, model_file=None, use_relu=True, use_cuda=True):
        super(D2Net, self).__init__()

        self.config = {**self.default_config, **config}
        self.dense_feature_extraction = DenseFeatureExtractionModule(
            use_relu=use_relu, use_cuda=use_cuda
        )

        self.mod = self.dense_feature_extraction.return_mod()
        self.gnn = AttentionalGNN(self.mod, self.config['descriptor_dim'], self.config['GNN_layers'])

        self.mod = self.gnn.return_mod()

        self.final_proj = nn.Conv2d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        self.mod = nn.Sequential(self.mod, self.final_proj)

        self.detection = HardDetectionModule()

        self.localization = HandcraftedLocalizationModule()

        if model_file is not None:
            if use_cuda:
                self.load_state_dict(torch.load(model_file)['model'], strict=False)
            else:
                self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

        if use_cuda:
            self.mod = self.mod.cuda()
            print(self.mod, (3, 640, 480))

    def forward(self, batch):
        _, _, h, w = batch.size()
        dense_features = self.dense_feature_extraction(batch)

        dense_features1 = dense_features[: b, :, :, :]
        dense_features2 = dense_features[b :, :, :, :]

        desc0, desc1 = self.gnn(dense_features1, dense_features2)

        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        dense_features = torch.cat([mdesc0, mdesc1], dim=0)

        detections = self.detection(dense_features)

        displacements = self.localization(dense_features)

        return {
            'dense_features': dense_features,
            'detections': detections,
            'displacements': displacements
        }


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)
