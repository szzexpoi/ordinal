import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

"""
Script for computing the means and std of training images
"""

transform = transforms.Compose([
		transforms.ToTensor(), 
	])

img_dir = './image'
img_pool = pd.read_csv('./annotation/train_lsp.csv')['img_name']

# for memory efficiency, we compute the means and std separately without storage
mu = torch.zeros(3,)
sigma = torch.zeros(3,)
N = 0

# compute means
for img_name in img_pool:
	img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
	img = transform(img)
	N += img.shape[1]*img.shape[2]
	mu = mu + img.sum([1, 2])
mu /= N

# compute std
tmp_mu = mu.view(3, 1, 1)
for img_name in img_pool:
	img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
	img = transform(img)
	sigma = sigma + ((img - tmp_mu)**2).sum([1, 2])
sigma = torch.sqrt(sigma/N)

print('Means:')
print(mu)
print('Standard deviation:')
print(sigma)




