#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import time
from queue import PriorityQueue


#with CV: median filter	
def medianFilterOPenCV(source_img,ksize):	
	filtered_img = cv2.medianBlur(source_img, ksize)
	return filtered_img
	
#in order to keep the size of image after filter as the same as original image, 
#first we extend the image, in the way of repeating the edge of the original image
def extendImage(source_img,ksize): #ksize must be odd
	m = source_img.shape[0]	#row
	n = source_img.shape[1]	#colum
	
	extend_img = np.copy(source_img)
	first_row = extend_img[0,:]
	last_row = extend_img[-1,:]

	for i in range(int((ksize-1)/2)):
		extend_img = np.row_stack((first_row,extend_img))
		extend_img = np.row_stack((extend_img,last_row))

	first_colum = extend_img[:,0]
	last_colum = extend_img[:,-1]

	for i in range(int((ksize-1)/2)):
		extend_img = np.column_stack((first_colum,extend_img))
		extend_img = np.column_stack((extend_img,last_colum))
	return extend_img
	
#in oder to make the code more concise, we create a template for different filter,
#to apply filters with the same process
def templateFilter(source_img,ksize,function):
	extended_img = extendImage(source_img,ksize)
	new_m = extended_img.shape[0]
	new_n = extended_img.shape[1]
	value_array = []
	for i in range(new_m-ksize+1):
		for j in range(new_n-ksize+1):
			kernel_window = extended_img[i:i+ksize,j:j+ksize]
			value_array.append(function(kernel_window.flatten()))
	value_array = np.array(value_array)
	return value_array.reshape((source_img.shape[0],source_img.shape[1]))

#naive way to find minimum value	
def naiveMinFilter_1D(pixel_array):
	min = pixel_array[0]
	for element in pixel_array:
		if element<=min:
			min = element
	return min

#find minimum value with PriorityQueue
def queueMinFilter_1D(pixel_array):	
	pq = PriorityQueue()
	for element in pixel_array:
			pq.put(element)
	return pq.get()	#最小值

#my implement:median filter
def efficentMedienFilter_1D(pixel_array): 
	if len(pixel_array)<70:	#ps.when the amount of pixel n>70, nlog(n)>128
		i = (np.sort(pixel_array))[int(len(pixel_array)/2)+1]
		return i
	if len(pixel_array)>=70:
		hist = np.bincount(pixel_array)	
		for i in range(1000):	
			i = 0
			while(1):
				if(sum(hist[0:i])>=(sum(hist[:])/2)):
					break
				i = i+1
		return i

#with CV: bilateral filter
def bilateralFilterOPenCV(source_img,d,sigmaColor,sigmaSpace):#高斯分布直径, 颜色相似度，空间相似度
	return cv2.bilateralFilter(source_img,d,sigmaColor,sigmaSpace)

#================ my implement:bilateral filter ================#
def bilateralFilterMy(source_img,d,sigmaColor,sigmaSpace):
	extended_img = extendImage(source_img,d)
	new_m = extended_img.shape[0]
	new_n = extended_img.shape[1]
	value_array = []
	for i in range(new_m-d+1):
		for j in range(new_n-d+1):
			kernel_window = extended_img[i:i+d,j:j+d]
			new_center_pixel = bilateralFilterWindow(kernel_window,sigmaColor,sigmaSpace)
			value_array.append(new_center_pixel)
	value_array = np.array(value_array)
	return value_array.reshape((source_img.shape[0],source_img.shape[1]))

# auxiliary function for bilateral filter : calculate domain kernel
def calculateDomainKernel(x,y,i,j,sigmaSpace):
	return np.exp(-((x-i)**2+(y-j)**2)/(2*sigmaSpace**2))

# auxiliary function for bilateral filter : calculate range kernel
def calculateRangeKernel(pixel_center,pixel,sigmaColor):
	return np.exp(-((pixel_center-pixel)**2/(2*sigmaColor**2)))

# auxiliary function for bilateral filter : apply a bilateral filter in a window
def bilateralFilterWindow(window,sigmaColor,sigmaSpace):
	d = window.shape[0]
	center_pixel = window[d/2,d/2]
	w_sum = 0
	numerator = 0
	for i in range(d):
		for j in range(d):
			domain_kernel = calculateDomainKernel(d/2,d/2,i,j,sigmaSpace)
			range_kernel = calculateRangeKernel(center_pixel,window[i,j])
			w = domain_kernel*range_kernel
			w_sum = w_sum + w
			numerator = numerator + window[i,j]*w
	return int(numerator/w_sum)
#==============================================================#

# test the runing time, and in order to apply it with different function,
# we set the param as variable
def calculateRunTime(function,*param):
	time_start=time.time()
	filtered_img = function(*param)
	time_end=time.time()
	print('totally cost',time_end-time_start)
	return filtered_img
	
if __name__ == '__main__':
	input_image_name = 'test.jpg'
	
	source_image = (cv2.imread(input_image_name,cv2.IMREAD_GRAYSCALE))# read in grey image
	
	print("the time of opencv->median:")
	filtered_img_cv = calculateRunTime(medianFilterOPenCV,source_image,3)
	print('\n')
	
	print("the time of my->median:")
	filtered_img_my = calculateRunTime(templateFilter,source_image,3,efficentMedienFilter_1D)
	print('\n')
	
	print("the time of naive->min:")
	filtered_img_my = calculateRunTime(templateFilter,source_image,3,naiveMinFilter_1D)
	print('\n')
	
	print("the time of queue->min:")
	filtered_img_my = calculateRunTime(templateFilter,source_image,3,queueMinFilter_1D)
	print('\n')
	
	print("the time of opencv->bilateral:")
	filtered_img_cv = calculateRunTime(bilateralFilterOPenCV,source_image,9,75,75)
	print('\n')
	
	print("the time of my->bilateral:")
	filtered_img_my = calculateRunTime(bilateralFilterOPenCV,source_image,9,75,75)
	print('\n')
