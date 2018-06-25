import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from pylab import ginput,imshow
import scipy.optimize as opt


def convertToGrey(input_image_name):
    grey = cv2.imread(input_image_name,cv2.IMREAD_GRAYSCALE)
    height, width = grey.shape[:2]
    print(height,width)
    size = (int(width*0.0625), int(height*0.0625))
    resize_grey = cv2.resize(grey,size)
    '''
    plt.imshow(resize_grey, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    '''
    return resize_grey
    
def findKPixels(img_list):
    center_image = img_list[len(img_list)//2]
    print(center_image.shape)
    imshow(center_image,cmap='gray')
    
    click_num = 6
    
    print('Please click %f points' %click_num)
    coordinate = ginput(click_num)
    new_coordinate = []
    for xy in coordinate :
        new_coordinate.append((int(xy[0]),int(xy[1])))
        
    print('you clicked:',new_coordinate)
    
    K_pixels = np.zeros((len(img_list),click_num))

    for i,xy in enumerate(new_coordinate):
        new_xy = [xy[0],xy[1]]
        
        if xy[0] - 5 < 0:
            new_xy[0] = 5
        if xy[0] + 5 > center_image.shape[0]:
            new_xy[0] = center_image.shape[0]-5
        if xy[1] - 5 < 0:
            new_xy[1] = 5
        if xy[1] + 5 > center_image.shape[1]:
            new_xy[1] = center_image.shape[1]-5

        for index,img in enumerate(img_list):
            average_pixel_value = int(np.mean(img[new_xy[0]-5:new_xy[0]+5,new_xy[1]-5:new_xy[1]+5]))
            K_pixels[index,i] = average_pixel_value

    
    return K_pixels

    
    
def estimateCameraResponseFunction(K_pixels,exposure_times):#Mitsunagaand NayarTechnique

    c_m_num  = 10
    
    R = 2
    
    print(K_pixels)
    def func(c_m):
        cost = 0.0
        for k in range(K_pixels.shape[1]):#6
            for i in range(K_pixels.shape[0]-1):#8
                t1 = 0
                t2 = 0
                for m in range(c_m_num):
                    t1 += c_m[m] * K_pixels[i,k] ** m 
                    t2 += c_m[m] * K_pixels[i+1,k] ** m
                cost +=  (t1 - R * t2)**2
        return cost

    c_m_solved = opt.fmin(func , np.random.rand(10))
    print(c_m_solved)
    
    x = np.linspace(0,1,255)
    y = []
    for every_pixel in x:
        t = 0
        for index,c_m in enumerate(c_m_solved):
            t += c_m * every_pixel ** index
        y.append(t)
    y = np.array(y)

    plt.figure(figsize=(8,4))
    plt.plot(x,y,color="red",linewidth=2)
    plt.xlabel("pixel")
    plt.ylabel("charge")
    plt.title("f -1 maps pixel values to charge")
    plt.show()
    return y
    
    
def radianceMapEstimation_oneImage(img, exposure_time,f_inverse):
    L_ = np.zeros_like(img)
    w = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            L_[i,j] = f_inverse[img[i,j]-1]/exposure_time
            w[i,j] = weightFunction(img[i,j])
    return L_,w

    
    
def radianceMapEstimation(img_list,exposure_times,f_inverse):
    t1 = np.zeros_like(img_list[0])
    t2 = np.ones_like(img_list[0])*0.01
    for index,img in enumerate(img_list):
        L_,w = radianceMapEstimation_oneImage(img, exposure_times[index],f_inverse)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                t1[i,j] += float(L_[i,j]) * float(w[i,j])

        t2 += w
        
    return t1/t2
                
def weightFunction(pixel_value):
    """ Linear weighting function based on pixel intensity that reduces the
    weight of pixel values that are near saturation.
    reference : https://github.com/vivianhylee/high-dynamic-range-image/blob/master/hdr.py
    """
    z_min, z_max = 0., 255.
    if pixel_value <= (z_min + z_max) / 2:
        return pixel_value - z_min
    return z_max - pixel_value

def cumulativeRadianceHistogram(L,bin):
    h = {}
    
    x = np.linspace(L.min(),L.max(), (L.max()-L.min())/bin)
    for index in range(len(x)-1):
        h[x[index]] = 0
        for i in L.flatten():
            if i> x[index] and i<x[index+1]:
                h[x[index]] += 1
    
    H = {}
    for i in range(len(x)-1):
        H[x[i]] = 0
        for j in range(len(x)-1):
            H[x[i]] += h[x[j]]
            if x[j] >= x[i]:
                break
                
                
        
    R_img = np.zeros_like(L)
    
    #H = {0.1:5.0, 0.3:1.0, 0.2:5.0}
    #sorted_H = {}
    sorted_keys = sorted(H.keys())
    
    for key in sorted_keys:
        H[key] = H[key]/H[max(sorted_keys)]
    '''
    for key in sorted_keys:
        sorted_H[key] = H[key]
    print(sorted_H)
    '''
    for row in range(L.shape[0]):
        for colum in range(L.shape[1]):
            for i in range(len(H.keys())-1):
                '''
                print('----------')
                print(sorted_keys[i])
                print(sorted_keys[i+1])
                print(L[row,colum])
                print('++++++++++')
                '''
                if sorted_keys[i] < L[row,colum] and sorted_keys[i+1] >= L[row,colum]:
                    R_img[row,colum] = H[sorted(H.keys())[i]]
                    print(R_img[row,colum])
                    break
    ''''''
    return R_img*255
    
if __name__ == '__main__':
    input_image_name_list = ['.\image\IMG_0015.ppm','.\image\IMG_0030.ppm','.\image\IMG_0060.ppm','.\image\IMG_0125.ppm','.\image\IMG_0250.ppm','.\image\IMG_0500.ppm','.\image\IMG_1000.ppm','.\image\IMG_2000.ppm']
    exposure_times = np.array([1/15,1/30,1/60,1/125,1/250,1/500,1/1000,1/2000])

    img_list = []
    for img_name in input_image_name_list:
        img_list.append(convertToGrey(img_name))
    K_pixels = findKPixels(img_list)
    f_inverse = estimateCameraResponseFunction(K_pixels, exposure_times)
    
    L = radianceMapEstimation(img_list,exposure_times,f_inverse)
    R_img = cumulativeRadianceHistogram(L,0.01)
    cv2.imwrite("result.jpg", R_img)
    cv2.waitKey(0)
    
    
    