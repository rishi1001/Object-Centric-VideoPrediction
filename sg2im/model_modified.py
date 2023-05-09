
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sg2im.box_utils as box_utils
from sg2im.graph import GraphTripleConv, GraphTripleConvNet
from sg2im.crn import RefinementNetwork
from sg2im.layout import boxes_to_layout, masks_to_layout
from sg2im.layers import build_mlp


class Sg2ImModel(nn.Module):
	def __init__(self,image_size=(224, 224), embedding_dim=1004,
							 gconv_dim=128, gconv_hidden_dim=512,
							 gconv_pooling='avg', gconv_num_layers=5,
							 refinement_dims=(1024, 512, 256, 256, 224),
							 normalization='batch', activation='leakyrelu-0.2',
							 mask_size=None, mlp_normalization='none', layout_noise_dim=0,
							 **kwargs):
		super(Sg2ImModel, self).__init__()

		# We used to have some additional arguments: 
		# vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
		if len(kwargs) > 0:
			print('WARNING: Model got unexpected kwargs ', kwargs)

		# self.vocab = vocab
		self.image_size = image_size
		self.layout_noise_dim = layout_noise_dim

		# num_objs = len(vocab['object_idx_to_name'])
		# num_preds = len(vocab['pred_idx_to_name'])
		# self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
		# self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

		if gconv_num_layers == 0:
			self.gconv = nn.Linear(embedding_dim, gconv_dim)
		elif gconv_num_layers > 0:
			gconv_kwargs = {
				'input_dim': embedding_dim,
				'output_dim': gconv_dim,
				'hidden_dim': gconv_hidden_dim,
				'pooling': gconv_pooling,
				'mlp_normalization': mlp_normalization,
			}
			self.gconv = GraphTripleConv(**gconv_kwargs)

		self.gconv_net = None
		if gconv_num_layers > 1:
			gconv_kwargs = {
				'input_dim': gconv_dim,
				'hidden_dim': gconv_hidden_dim,
				'pooling': gconv_pooling,
				'num_layers': gconv_num_layers - 1,
				'mlp_normalization': mlp_normalization,
			}
			self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

		box_net_dim = 4
		box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
		self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

		self.mask_net = None
		if mask_size is not None and mask_size > 0:
			self.mask_net = self._build_mask_net(num_objs, gconv_dim, mask_size)

		rel_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, 64]
		self.rel_aux_net = build_mlp(rel_aux_layers, batch_norm=mlp_normalization)

		refinement_kwargs = {
			'dims': (gconv_dim + layout_noise_dim,) + refinement_dims,
			'normalization': normalization,
			'activation': activation,
		}
		self.refinement_net = RefinementNetwork(**refinement_kwargs)

		#VIDGNN
		self.edge_embedding = nn.Sequential(
			nn.Linear(8,256),
			nn.ReLU(),
			nn.Linear(256,256),
			nn.ReLU(),
			nn.Linear(256,1004)
		)


	def _build_mask_net(self, num_objs, dim, mask_size):
		output_dim = 1
		layers, cur_size = [], 1
		while cur_size < mask_size:
			layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
			layers.append(nn.BatchNorm2d(dim))
			layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
			layers.append(nn.ReLU())
			cur_size *= 2
		if cur_size != mask_size:
			raise ValueError('Mask size must be a power of 2')
		layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
		return nn.Sequential(*layers)

	
	def get_edge_info(self,obj_vecs,obj_to_img):
		#assume fully connected graphs
		edges = []
		in_ = []
		s = []
		o = []
		for i in range(obj_vecs.size(0)):
			for j in range(obj_vecs.size(0)):
				if i!=j and obj_to_img[i] == obj_to_img[j]:
					s.append(i)
					o.append(j)
					edges.append([i,j])
					in_.append(torch.cat([obj_vecs[i][-4:],obj_vecs[i][-4:]]))
					
		edges = torch.tensor(edges).to(obj_vecs.device).long()
		in_ = torch.stack(in_)
		edge_vecs = self.edge_embedding(in_)

		return edges,edge_vecs,o,s

	def forward(self,obj_vecs,obj_to_img):
		
		edges,pred_vecs,o,s =  self.get_edge_info(obj_vecs,obj_to_img)
		obj_vecs_orig = obj_vecs 
		if isinstance(self.gconv, nn.Linear):
			obj_vecs = self.gconv(obj_vecs)
		else:
			obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
		
		if self.gconv_net is not None:
			obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)
		
		boxes_pred = self.box_net(obj_vecs)


		H, W = self.image_size
		layout_boxes = boxes_pred # if boxes_gt is None else boxes_gt

		layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
		
		img = self.refinement_net(layout)
		return img, boxes_pred