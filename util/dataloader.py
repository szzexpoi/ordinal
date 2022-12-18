import numpy as np
import random
import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import json
from PIL import Image
from torchvision import transforms
import pandas as pd

class LSP_generator(data.Dataset):
	""" Dataloader for processing LSP data
	"""
	def __init__(self, img_dir, anno_dir, split='train', bbox_size=32,  
				patch_size=16, mask_size=32, mask_sigma=0.2, use_softmax=False):
		""" Initialization function

		Inputs:
			img_dir: Directory storing image files
			anno_dir: Directory storing annotation files
			mode: Specify data split, 'train', 'val', or 'test'
			bbox_size: Spatial size of the bounding box
			patch_size: Spatial size of the patch for each point
			mask_size: spatial size of gaussian mask for each point
			mask_sigma: variance of gaussian kernel for 
					generating the gaussian mask
			use_softmax: whether or not using softmax for classification layer
		"""
		self.img_dir = img_dir
		self.split = split
		self.patch_size = patch_size
		self.mask_size = mask_size
		self.mask_sigma = mask_sigma
		self.normalizer = transforms.Normalize([0.460, 0.432, 0.378], [0.264, 0.246, 0.251])
		self.bbox_rescaler = transforms.Resize((bbox_size, bbox_size))
		self.mask_rescaler = transforms.Resize((mask_size, mask_size))
		self.use_softmax = use_softmax
		anno_file = pd.read_csv(os.path.join(anno_dir, 
						split+'_lsp.csv'))
		self.init_data(anno_file)


	def init_data(self, anno_file):
		""" Function for reorganizing the data.

			Input:
				anno_file: Annotation data.
		"""
		self.img = []
		self.points = []
		self.label = []

		for i in range(len(anno_file['img_name'])):
			self.img.append(anno_file['img_name'][i])
			y1, x1 = anno_file['y_A'][i], anno_file['x_A'][i]
			y2, x2 = anno_file['y_B'][i], anno_file['x_B'][i],
			self.points.append([y1-1, x1-1, y2-1, x2-1]) # the raw data uses 1-indexing
			self.label.append(0 if anno_file['ordinal relationship'][i] == '<' else 1)


	def crop_patch(self, img, points):
		""" Function for cropping patches around the selecte points.

			Inputs:
				img: Torch tensor storing the image data
				points: A list storing the location of points

			Returns:
				patch: A list containing tensors for the two patches
		"""

		height, width = img.shape[1], img.shape[2]
		y1, x1, y2, x2 = points
		
		# croping patches centered at the given points
		patch = []
		for y, x in [[y1, x1], [y2, x2]]:
			y_min = max(0, y-self.patch_size/2)
			y_max = min(y+self.patch_size/2, height)
			x_min = max(0, x-self.patch_size/2)
			x_max = min(x+self.patch_size/2, width)
			cur_patch = img[:, int(y_min):int(y_max), int(x_min):int(x_max)]

			# zero-padding (before normalization)
			patch_height, patch_width = cur_patch.shape[1], cur_patch.shape[2]
			pad_interval = [0, 0, 0, 0]
			if patch_height < self.patch_size:
				residual = self.patch_size-patch_height
				if residual%2 == 0:
					pad_interval[2] = int(residual/2)
				else:
					pad_interval[2] = int(residual/2)+1
				pad_interval[3] = int(residual/2)
			if patch_width < self.patch_size:
				residual = self.patch_size-patch_width
				if residual%2 == 0:
					pad_interval[0] = int(residual/2)
				else:
					pad_interval[0] = int(residual/2)+1
				pad_interval[1] = int(residual/2)				

			cur_patch = nn.ZeroPad2d(tuple(pad_interval))(cur_patch)
			patch.append(cur_patch)

		return patch


	def generate_gaussian_mask(self, img, points):
		""" Function for generating gaussian masks for the two points.

			Inputs:
				img: Torch tensor storing the image data
				points: A list storing the location of points

			Returns:
				mask: A list containing tensors for the two patches
		"""

		height, width = img.shape[1], img.shape[2]
		y1, x1, y2, x2 = points

		# convert sigma into pixel expression
		# see: https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
		sigma = 2*int(4*self.mask_sigma+0.5)+1

		# generate gaussian masks
		mask = []
		for y, x in [[y1, x1], [y2, x2]]:
			cur_mask = np.zeros([height, width]).astype('float32')
			for i in range(height):
				for j in range(width):
					cur_mask[i, j] = (((i-y)**2+(j-x)**2)/(2.0*sigma**2))

			cur_mask = np.exp(-cur_mask)/(2*np.pi*sigma**2)
			cur_mask /= np.sum(cur_mask)
			mask.append(torch.from_numpy(cur_mask).unsqueeze(0))

		return mask


	def __getitem__(self, index):
		# converting raw image data into tensor
		img_name = self.img[index]
		img =  Image.open(os.path.join(self.img_dir, 
						img_name)).convert('RGB')
		img = transforms.ToTensor()(img)

		# supply model with additional information about relative positions
		delta_y = self.points[index][0] - self.points[index][2]
		delta_x = self.points[index][1] - self.points[index][3]
		relative_pos = torch.FloatTensor([delta_y, delta_x])

		# cropping the patches around the two points
		patch_1, patch_2 = self.crop_patch(img, self.points[index])

		# generate gaussian masks for the two points
		mask_1, mask_2 = self.generate_gaussian_mask(img, self.points[index])

		# normalize and rescale the image and patches
		img = self.normalizer(img)
		img = self.bbox_rescaler(img)
		patch_1 = self.normalizer(patch_1)
		patch_2 = self.normalizer(patch_2)
		mask_1 = self.mask_rescaler(mask_1)
		mask_1 /= mask_1.max() # previously max
		mask_2 = self.mask_rescaler(mask_2)
		mask_2 /= mask_2.max() # previously max


		if not self.use_softmax:
			# get binary label
			label = torch.FloatTensor([self.label[index]])
		else:
			label = torch.zeros(2)
			label[int(self.label[index])] = 1

		return img, patch_1, patch_2, mask_1, mask_2, relative_pos, label


	def __len__(self, ):
		return len(self.img)

