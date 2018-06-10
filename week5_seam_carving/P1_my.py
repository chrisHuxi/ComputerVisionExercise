import numpy as np
import cv2
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

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

    
#when we got the energy map, we can calculate the path of the low energy, with dynamic-planning algorithm
#reference : https://www.youtube.com/watch?v=mPWVG6mCu80&t=636s
def dynamicPlanning(energy_map):
    S = np.zeros_like(energy_map)
    S[0,:] = energy_map[0,:]
    path = []
    for m in range(energy_map.shape[1]):
        path.append([])
    
    for i in range(1,energy_map.shape[0]):
        for j in range(0,energy_map.shape[1]):
            t = [float("inf") ,S[i-1,j],float("inf")]
            if j-1 >= 0:
                t[0] = S[i-1,j-1]
            if j+1 <= energy_map.shape[1]-1:
                t[2] = S[i-1,j+1]
            j_choosed = t.index(min(t)) + j-1
            path[j].append(j_choosed)
            S[i,j] = float(min(t))+float(energy_map[i,j])
    min_S_index = np.where(S[energy_map.shape[0]-1,:]==(min(S[energy_map.shape[0]-1,:])))
    pixel_pos = [(0,min_S_index[0][0])]
    
    for element in enumerate(path[min_S_index[0][0]]):
        pixel_pos.append((element[0]+1,element[1]))
    return pixel_pos        

#for the place where we detected line, increase the energy value
def increaseEnergyValue(em,radius,position,value):
    for i in range(radius):
        if (position[0]+i < em.shape[0]-1) and (position[1]+i < em.shape[1]-1):
            em[position[0]+i,position[1]+i] += value
        if (position[0]+i < em.shape[0]-1) and (position[1]-i >= 0):
            em[position[0]+i,position[1]-i] += value
        if (position[0]-i >= 0) and (position[1]+i < em.shape[1]-1):
            em[position[0]-i,position[1]+i] += value
        if (position[0]-i >= 0) and (position[1]-i >= 0):
            em[position[0]-i,position[1]-i] += value
    
#when we find the path of the low energy, remove pixels in this path
def reducePixel(source_image,itreative_num,line_cut_set):
    t = source_image[:]
    intersection_flag = 0
    intersection_pos = []
    for i in range(itreative_num):
        if intersection_flag == 0:
            contrast_map = energyMap(t)
        if intersection_flag == 1:  
            contrast_map = energyMap(t)
            for everypos in intersection_pos:   #对于每一个相交点
                increaseEnergyValue(contrast_map,5,everypos,50)
                intersection_flag = 0
                
        pixel_pos = dynamicPlanning(contrast_map)
        if set(pixel_pos)&set(line_cut_set) != {}:#有相交的情况
            intersection_flag = 1
            intersection_pos = list(set(pixel_pos)&set(line_cut_set))
            print(intersection_pos)
        for xy in pixel_pos:
            t[xy[0],xy[1]:-2,:] = t[xy[0],xy[1]+1:-1,:]
        t = t[:,0:-1,:]
        print(i)
    return t


#to find line using hough transformation
def findStraightLine(img):
    img = cv2.GaussianBlur(img,(3,3),0)  
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10  
    lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)  
    line_cut_set = []
    for startx,starty,endx,endy in lines[:,0,:]:
        line_cut_set.extend(calcuLine(startx,starty,endx,endy))
    return line_cut_set
    
#with hough transformation we can get the start and end point's coordinate
# but we should calculate all the points' coordinate in this line
def calcuLine(startx,starty,endx,endy):
    t = []
    xstep=1
    if startx > endx:
        xstep=-1
    for x in range(startx,endx,xstep):
        y=int((x-startx)*(endy-starty)/(endx-startx))+starty
        t.append((x,y))
    return t
    
#workflow of seam-carvering
if __name__ == '__main__':
    input_image_name = 'tram.png'
    source_image = (cv2.imread(input_image_name))
    line_cut_set = findStraightLine(source_image)
    
    contrast_map = energyMap(source_image)
    
    t = reducePixel(source_image,50,line_cut_set)

    cv2.imshow('my_seam_carving_result',t)
    cv2.imwrite("my_seam_carving_result.jpg", t)
    cv2.waitKey(0)
    
    
    
    
        