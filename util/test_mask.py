import numpy as np
import torch
import cv2

def generate_gaussian_mask(img, points, mask_sigma):
	""" Function for generating gaussian masks for the two points.

		Inputs:
			img: Torch tensor storing the image data
			points: A list storing the location of points

		Returns:
			mask: A list containing tensors for the two patches
	"""

	height, width = img.shape[1], img.shape[2]
	y1, x1, y2, x2 = points

	# convert sigma into pixel size
	sigma = 2*int(4*mask_sigma+0.5)+1

	# generate gaussian masks
	mask = []
	for y, x in [[y1, x1], [y2, x2]]:
		cur_mask = np.zeros([height, width]).astype('float32')
		for i in range(height):
			for j in range(width):
				cur_mask[i, j] = (((i-y)**2+(j-x)**2)/(2.0*sigma**2))

		cur_mask = np.exp(-cur_mask)/(2*np.pi*sigma**2)
		cur_mask /= np.max(cur_mask)
		mask.append(cur_mask)

	return mask

img = torch.randn(3, 95, 95)
points = [30, 20, 10, 50]
mask_1, mask_2 = generate_gaussian_mask(img, points, 0.2)

cv2.imwrite('test_1.png', mask_1*255)
cv2.imwrite('test_2.png', mask_2*255)