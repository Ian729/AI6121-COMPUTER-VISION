import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def calculate_cdf(channel):
	hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
	cdf = np.cumsum(hist)
	return hist, cdf

def equalize(cdf):
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	return cdf

def HE(path):
	# read image
	img = cv.imread(path)
	# split into three channels
	b, g, r = cv.split(img)

	hist_b, cdf_b = calculate_cdf(b)
	hist_g, cdf_g = calculate_cdf(g)
	hist_r, cdf_r = calculate_cdf(r)

	equalized_cdf_b = equalize(cdf_b)
	equalized_cdf_g = equalize(cdf_g)
	equalized_cdf_r = equalize(cdf_r)

	img2_b = equalized_cdf_b[b]
	img2_g = equalized_cdf_g[g]
	img2_r = equalized_cdf_r[r]

	img2 = cv.merge((img2_b, img2_g, img2_r))

	return img, img2



if __name__ == "__main__":
	for i in range(8):
		img, img2 = HE('sample0' + str(i+1) + '.jpeg')
		res = np.hstack((img,img2))
		cv.imwrite("result"+str(i+1)+".jpeg", res)





