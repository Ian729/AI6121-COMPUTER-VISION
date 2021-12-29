import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def draw_image_and_cdf(image, name):
	# get histogram and bins
	hist,bins = np.histogram(image.flatten(), 256, [0,256])
	# get cdf
	cdf = hist.cumsum()
	# normalize to fit in the range  (0,256)
	cdf_normalized = cdf * float(hist.max()) / cdf.max()
	cv.imshow("image",image)
	plt.plot(cdf_normalized, color = 'b')
	plt.hist(image.flatten(),256,[0,256], color = 'r')
	plt.legend(('cdf','histogram'), loc = 'upper left')
	plt.savefig(name)
	plt.show()
	return cdf

def equalize(cdf):
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	return cdf

def HE(path):
	img = cv.imread(path, 0)
	name = path.split('.')[0]
	cv.imwrite(name+"_1.jpeg", img)
	cdf = draw_image_and_cdf(img, name + "_1_cdf.jpeg")
	equalized_cdf = equalize(cdf)
	img2 = equalized_cdf[img]
	cdf = draw_image_and_cdf(img2,name + "_2_cdf.jpeg")
	cv.imwrite(name+"_2.jpeg", img2)
	return


if __name__ == "__main__":
	for i in range(1,9):
		HE('sample0' + str(i) + '.jpeg')