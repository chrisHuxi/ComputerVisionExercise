import numpy as np
from scipy import signal
import time

def calculateRunTime(function,*param):
	time_start=time.time()
	result = function(*param)
	time_end=time.time()
	print('totally cost',time_end-time_start)
	return result
    
def fourierTransform(img,filter):
    f = img
    g = filter
    
    s1 = np.array(f.shape)
    s2 = np.array(g.shape)
    size = s1 + s2 - 1
    
    fsize = 2 ** (np.ceil(np.log2(size))).astype(int)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    
    new_f = np.fft.fft2(f , fsize)
    new_g = np.fft.fft2(g , fsize)
    result = np.fft.ifft2(new_f*new_g)[fslice]
    return result.real

def convolution(img, filter):#fast convolution, implemeted also by FFT
    return signal.convolve2d(img , filter , 'full')
    
if __name__ == '__main__':
    img = np.random.randint(0,255,size=[1024,1024])
    filter = np.random.randint(0,10,size=[5,5])
    print("the time of fourier transform method:")
    filtered_img_fourierTransform = calculateRunTime(fourierTransform, img, filter)
    print('\n')
    print("the time of convolution method:")
    filtered_img_convolution = calculateRunTime(convolution, img, filter)
    print('\n')
    
    #print(filtered_img_fourierTransform)
    #print(filtered_img_convolution)
    print(filtered_img_fourierTransform.all() == filtered_img_convolution.all())