import numpy as np
import cv2
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from skimage import transform
from skimage import filters
import cv2

'''I look up this energy-map-calculation algorithm from other anothur, because my implemention seems dosen't work ( for more details see readme document in github)'''
#=======================================================================================================#
# return difference of pixel with neighbors
def simpleEnergy(x0, x1, y0, y1):
    return sum(abs(x0-x1) + abs(y0-y1))

#returns 2-D numpy array with the same height and width as img Each energy[x][y] is an int specifying the energy of that pixel
# but the anothur didn't use theta = image.size()/20, he set theta as a content number : 2
def energyMap(img):
    x0 = np.roll(img, -1, axis=1).T
    x1 = np.roll(img, 1, axis=1).T
    y0 = np.roll(img, -1, axis=0).T
    y1 = np.roll(img, 1, axis=0).T
    return simpleEnergy(x0, x1, y0, y1).T
    
# reference : https://github.com/margaret/seam-carver/blob/master/energy_functions.py
#=======================================================================================================#

    
    
if __name__ == '__main__':
    input_image_name = 'tram.png'
    source_image = cv2.imread(input_image_name)
    
    #contrast_map = calcContrastMap(extend_image,theta)
    contrast_map = energyMap(source_image)
    

    carved = transform.seam_carve(source_image, contrast_map, 'vertical',
		50)
    cv2.imshow('skt_seam_carving_result',carved)
    new_carved = np.zeros_like(carved)
    new_carved = cv2.normalize(carved,new_carved, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite("skt_seam_carving_result.jpg", new_carved)
    cv2.waitKey(0)
    
    
    
        