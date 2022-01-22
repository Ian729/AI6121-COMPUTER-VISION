import matplotlib.pyplot as plt
import cv2, os
import numpy as np

def load_images(path, img1, img2):
	img_1 = cv2.imread(os.path.join(path,img1),cv2.IMREAD_GRAYSCALE)
	img_2 = cv2.imread(os.path.join(path,img2),cv2.IMREAD_GRAYSCALE)
	return np.array(img_1), np.array(img_2)

def calculate_depth(left, right, D: int, S: int):
	"""
	left: left image
	right: right image
	D: depth in disparity map
	S: window size

	"""
	assert(left.shape == right.shape)

	H, W = left.shape
	cost = np.zeros((H, W, D))
	half_window = S//2

	# loop over image
	for y in range(half_window, H - half_window):
		for x in range(half_window, W - half_window):
			# loop over window:
			for v in range(-half_window, half_window + 1):
				for u in range(-half_window, half_window + 1):
					# loop over all depths
					for d in range(D):
						cost[y,x,d] += (int(left[y+v, x+u]) - int(right[y+v, x+u-d])) ** 2
	return cost

def calculate_disparity(cost):
	return np.argmin(cost, axis=2)

if __name__ == '__main__':

	left, right = load_images("/Users/zwx/Desktop/NTUMaster/AI6121 COMPUTER VISION/CV Assignment 2","corridorl.jpeg","corridorr.jpeg")
	
	cost = calculate_depth(left, right, 16, 7)
	d_map = calculate_disparity(cost)
	
	fig, (ax1, ax2, ax3) = plt.subplots(1,3)
	ax1.imshow(left,cmap='gray')
	ax2.imshow(right,cmap='gray')
	ax3.imshow(d_map, cmap='gray')
	plt.show()

