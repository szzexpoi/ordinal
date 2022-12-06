import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalNet(nn.Module):
	""" Re-implementation of the ordinal prediction model
	introduced in paper (hyperparameters are fixed):
	https://people.csail.mit.edu/danielzoran/ordinal.pdf
	"""

	def __init__(self, ):
		super(OrdinalNet.self).__init__()
		# Use the same LRN normalization as it has no trainable parameter
		self.lrn = nn.LocalResponseNorm(5, 1e-4, 0.75)

		# layers for downscaled image
		self.conv1_I = nn.Conv2d(3, 16, kernel_size=5, 
							padding=2, bias=True)
		self.pool1_I = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2_I = nn.Conv2d(16, 48, kernel_size=5, 
							padding=2, bias=True)
		self.pool2_I = nn.MaxPool2d(kernel_size=2, stride=2)

		# layer for image+ROI
		self.fc_ROI_I = nn.Linear(16384, 300)

		# layers for Bounding box
		self.conv1_BB = nn.Conv2d(3, 16, kernel_size=5, 
							padding=2, bias=True)
		self.pool1_BB = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2_BB = nn.Conv2d(16, 48, kernel_size=3, 
							padding=1, bias=True)
		self.pool2_BB = nn.MaxPool2d(kernel_size=2, stride=2)
		self.fc_BB = nn.Linear(3072, 100)

		# layers for Patches (two in total)
		self.conv_P1 = nn.Conv2d(3, 16, kernel_size=5, 
							padding=2, bias=True)
		self.pool_P1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv_P2 = nn.Conv2d(3, 16, kernel_size=5, 
							padding=2, bias=True)
		self.pool_P2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# layers for Masks (for the patches)
		self.pool_M1 = nn.AvgPool2d(kernel_size=2, stride=2)
		self.pool_M2 = nn.AvgPool2d(kernel_size=2, stride=2)

		# classification layers
		self.fc1_cls = nn.Linear(2960, 300)
		self.dp = nn.Dropout(0.5)
		self.fc2_cls = nn.Linear(300, 150)
		self.fc3_cls = nn.Linear(150, 3)


	def forward(self, I, ROI, Bbox, P1, P2, M1, M2):
		""" Inference process for predicting the ordinal relationship
		between two points.

		Inputs:
			I: Downscaled images with size (batch, 3, 64, 64)
			ROI: Binary masks indicating the locations of ROIs with 
				size (batch, 1, 64, 64)
			Bbox: Cropped bounding boxes covering the two points, 
				with size (batch, 3, 32, 32)
			P1: Patches for the first points, size (batch, 3, 16, 16)
			P2: Patches for the second points, size (batch, 3, 16, 16)
			M1: Gaussian masks for the first points, size (batch, 1, 32, 32)
			M1: Gaussian masks for the second points, size (batch, 1, 32, 32)

		Return:
			pred: Predicted ordinal relationships between the selected 
				two points. Predictions are formulated as probabilities 
				of three classes: equal, greater than, and less than.  
		"""
		batch = len(I)

		# learning features from the downscaled image
		I_conv_feat = torch.relu(self.conv1_I(I))
		I_conv_feat = self.pool1_I(I_conv_feat)
		I_conv_feat = self.lrn(I_conv_feat)

		I_conv_feat = torch.relu(self.conv2_I(I_conv_feat))
		I_conv_feat = self.pool2_I(I_conv_feat)
		I_conv_feat = self.lrn(I_conv_feat)
		I_feat = I_conv_feat.view(batch, -1)

		# merging features from the image and ROI
		ROI_feat = ROI.view(batch, -1)
		I_ROI_feat = torch.cat([I_feat, ROI_feat], dim=-1)
		I_ROI_feat = torch.relu(self.fc_ROI_I(I_ROI_feat))

		# learning features from bounding boxes
		BB_conv_feat = torch.relu(self.conv1_BB(Bbox))
		BB_conv_feat = self.pool1_BB(BB_conv_feat)
		BB_conv_feat = self.lrn(BB_conv_feat)

		BB_conv_feat = torch.relu(self.conv2_BB(BB_conv_feat))
		BB_conv_feat = self.pool2_BB(BB_conv_feat)
		BB_conv_feat = self.lrn(BB_conv_feat)

		BB_feat = BB_conv_feat.view(batch, -1)
		BB_feat = torch.relu(self.fc_BB(BB_feat))

		# learning features from the two patches
		P1_feat = torch.relu(self.conv_P1(P1))
		P1_feat = self.pool_P1(P1_feat)
		P1_feat = self.lrn(P1_feat)
		P1_feat = P1_feat.view(batch, -1)

		P2_feat = torch.relu(self.conv_P2(P2))
		P2_feat = self.pool_P2(P2_feat)
		P2_feat = self.lrn(P2_feat)
		P2_feat = P2_feat.view(batch, -1)

		# resizing (and reversing) binary masks for patches
		M1_feat = self.pool_M1(M1)
		M1_feat = M1_feat.view(batch, -1) 
		M2_feat = -self.pool_M2(M2)  # need to time -1
		M2_feat = M2_feat.view(batch, -1) 

		# concatenating all features and proceed for classification
		cls_feat = torch.cat([I_ROI_feat, BB_feat, P1_feat, 
							P2_feat, M1_feat, M2_feat], dim=-1)
		cls_feat = torch.relu(self.fc1_cls(cls_feat))
		cls_feat = torch.relu(self.fc2_cls(self.dp(cls_feat)))
		pred = F.softmax(self.fc3_cls(cls_feat), dim=-1)

		return pred


class OrdinalNet_slim(nn.Module):
	""" A customized version of the ordinal prediction model
	introduced in paper (removing global context):
	https://people.csail.mit.edu/danielzoran/ordinal.pdf
	"""

	def __init__(self, ):
		super(OrdinalNet_slim, self).__init__()
		# Use the same LRN normalization as it has no trainable parameter
		self.lrn = nn.LocalResponseNorm(5, 1e-4, 0.75)

		# layers for Bounding box
		self.conv1_BB = nn.Conv2d(3, 16, kernel_size=5, 
							padding=2, bias=True)
		self.pool1_BB = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2_BB = nn.Conv2d(16, 48, kernel_size=3, 
							padding=1, bias=True)
		self.pool2_BB = nn.MaxPool2d(kernel_size=2, stride=2)
		self.fc_BB = nn.Linear(3072, 100)

		# layers for Patches (two in total)
		self.conv_P1 = nn.Conv2d(3, 16, kernel_size=5, 
							padding=2, bias=True)
		self.pool_P1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv_P2 = nn.Conv2d(3, 16, kernel_size=5, 
							padding=2, bias=True)
		self.pool_P2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# layers for Masks (for the patches)
		self.pool_M1 = nn.AvgPool2d(kernel_size=2, stride=2)
		self.pool_M2 = nn.AvgPool2d(kernel_size=2, stride=2)

		# classification layers
		self.fc1_cls = nn.Linear(2660, 300)
		self.dp = nn.Dropout(0.1)
		self.fc2_cls = nn.Linear(300, 150)
		self.fc3_cls = nn.Linear(150, 1)


	def forward(self, Bbox, P1, P2, M1, M2):
		""" Inference process for predicting the ordinal relationship
		between two points.

		Inputs:
			Bbox: Cropped bounding boxes covering the two points, 
				with size (batch, 3, 32, 32)
			P1: Patches for the first points, size (batch, 3, 16, 16)
			P2: Patches for the second points, size (batch, 3, 16, 16)
			M1: Gaussian masks for the first points, size (batch, 1, 32, 32)
			M1: Gaussian masks for the second points, size (batch, 1, 32, 32)

		Return:
			pred: Predicted ordinal relationships between the selected 
				two points. Predictions are formulated as binary probabilities 
				point A is greater than point B.   
		"""
		batch = len(Bbox)

		# learning features from bounding boxes
		BB_conv_feat = torch.relu(self.conv1_BB(Bbox))
		BB_conv_feat = self.pool1_BB(BB_conv_feat)
		BB_conv_feat = self.lrn(BB_conv_feat)

		BB_conv_feat = torch.relu(self.conv2_BB(BB_conv_feat))
		BB_conv_feat = self.pool2_BB(BB_conv_feat)
		BB_conv_feat = self.lrn(BB_conv_feat)

		BB_feat = BB_conv_feat.view(batch, -1)
		BB_feat = torch.relu(self.fc_BB(BB_feat))

		# learning features from the two patches
		P1_feat = torch.relu(self.conv_P1(P1))
		P1_feat = self.pool_P1(P1_feat)
		P1_feat = self.lrn(P1_feat)
		P1_feat = P1_feat.view(batch, -1)

		P2_feat = torch.relu(self.conv_P2(P2))
		P2_feat = self.pool_P2(P2_feat)
		P2_feat = self.lrn(P2_feat)
		P2_feat = P2_feat.view(batch, -1)

		# resizing (and reversing) binary masks for patches
		M1_feat = self.pool_M1(M1)
		M1_feat = M1_feat.view(batch, -1) 
		M2_feat = -self.pool_M2(M2)  # need to time -1
		M2_feat = M2_feat.view(batch, -1) 

		# concatenating all features and proceed for classification
		cls_feat = torch.cat([BB_feat, P1_feat, P2_feat, 
							M1_feat, M2_feat], dim=-1)
		cls_feat = torch.relu(self.fc1_cls(cls_feat))
		cls_feat = torch.relu(self.fc2_cls(self.dp(cls_feat)))
		pred = torch.sigmoid(self.fc3_cls(cls_feat))

		return pred

class OrdinalNet_att(nn.Module):
	""" A customized model for ordinal prediction. The model leverages
		attention mechanism to replace the explicit extraction of patches.
	"""

	def __init__(self, use_softmax):
		super(OrdinalNet_att, self).__init__()

		self.use_softmax = use_softmax

		# user a global encoder to extract features from the whole image
		self.encoder = nn.Sequential(
				nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
				nn.ReLU(),
				# nn.BatchNorm2d(16),
				nn.MaxPool2d(kernel_size=2, stride=2),
				nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True),
				nn.ReLU(),
				# nn.BatchNorm2d(32),
				nn.MaxPool2d(kernel_size=2, stride=2),
				nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=True),
				nn.ReLU(),
				# nn.BatchNorm2d(48),
			)

		# learn independent decoders for the two patches and adaptive context
		self.P1_decoder = nn.Linear(48, 48)
		self.P2_decoder = nn.Linear(48, 48)

		# use an attention mechanism to capture contextual information
		self.context_att = nn.Conv2d(48, 1, kernel_size=3, 
						padding=1, bias=True)
		self.context_decoder = nn.Linear(48, 32)

		# classification layer
		self.dp = nn.Dropout(0.2)
		if not self.use_softmax:
			self.cls_layer = nn.Linear(128, 1)
		else:
			self.cls_layer = nn.Linear(128, 2)

	def forward(self, Bbox, M1, M2):
		""" Inference process for predicting the ordinal relationship
		between two points.

		Inputs:
			Bbox: Cropped bounding boxes covering the two points, 
				with size (batch, 3, 32, 32)
			M1: Gaussian masks for the first points, size (batch, 1, 32, 32)
			M1: Gaussian masks for the second points, size (batch, 1, 32, 32)

		Return:
			pred: Predicted ordinal relationships between the selected 
				two points. Predictions are formulated as binary probabilities 
				point A is greater than point B.   
		"""
		
		# extract visual features from the whole image
		v_feat = self.encoder(Bbox)

		# use the two gaussian masks to locate features at the proximity of points
		P1_feat = (v_feat*M1).sum([2, 3])
		P1_feat = torch.relu(self.P1_decoder(P1_feat))
		P2_feat = (v_feat*M2).sum([2, 3])
		P2_feat = torch.relu(self.P1_decoder(P2_feat))

		# extract contextual information with a single attention map
		att_map = self.context_att(v_feat)
		batch, _, h, w = att_map.shape
		att_map = F.softmax(att_map.view(batch, h*w), dim=-1)
		att_map = att_map.view(batch, 1, h, w)
		context_feat = (v_feat*att_map).sum([2, 3])
		context_feat = self.context_decoder(context_feat)

		# classification
		cls_feat = torch.cat([context_feat, P1_feat, P2_feat], dim=-1)
		if not self.use_softmax:
			pred = torch.sigmoid(self.cls_layer(self.dp(cls_feat)))
		else:
			pred = F.softmax(self.cls_layer(self.dp(cls_feat)), dim=-1)

		return pred
