#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
from cv2.ximgproc import guidedFilter
import matplotlib.pyplot as plt

def guidFilterTest(guide_image,src_image,radius,eps):#guide , src, radius of Guided Filter, regularization term of Guided Filter
	'''
	if guide_image is the same as src_image, this filter will change to edge-perserving
	filter.
	'''
	return guidedFilter(guide_image,src_image,radius,eps) 
	
def bilateralFilterTest(source_img,d,sigmaColor,sigmaSpace):#高斯分布直径, 颜色相似度，空间相似度
	return cv2.bilateralFilter(source_img,d,sigmaColor,sigmaSpace)



if __name__ == '__main__':
	img = plt.imread("test.jpg")
	img_guid_filter = guidFilterTest(img,img,9,175)
	img_bilateral_filter = bilateralFilterTest(img,9,175,175)
	plt.figure(figsize = (14,8))
	plt.subplot(211)
	plt.imshow(img_guid_filter)
	plt.subplot(212)
	plt.imshow(img_bilateral_filter)
	plt.show()