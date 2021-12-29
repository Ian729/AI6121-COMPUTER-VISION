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

def HE(channel):
    hist, cdf = calculate_cdf(channel)
    eq_cdf = equalize(cdf)
    return eq_cdf[channel]

def HSV(path):
    img = cv.imread(path)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = HE(img_hsv[:, :, 2])
    img_rgb = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
    return img, img_rgb

def YCrCb(path):
    rgb_img = cv.imread(path)
    ycrcb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = HE(ycrcb_img[:, :, 0])
    equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)
    return rgb_img, equalized_img

if __name__ == "__main__":
    for i in range(1,9):
        img, img_hsv = HSV('sample0' + str(i) + '.jpeg')
        img, img_ycrcb = YCrCb('sample0' + str(i) + '.jpeg')
        res = np.hstack((img,img_hsv, img_ycrcb))
        cv.imwrite("result"+str(i+1)+".jpeg", res)