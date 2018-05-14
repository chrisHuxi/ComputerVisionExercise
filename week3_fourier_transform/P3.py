import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import cv2
import PIL.Image as Image

img  = cv2.imread("./P3_img/txt[1].png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
gray = cv2.bitwise_not(gray)

thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
fimg = np.log(np.abs(fshift))

max_peak = np.max(np.abs(f))

fshift[fshift < (max_peak * 0.25)] = 0

# Log-scale the data
abs_data = 1 + np.abs(fshift)
c = 255.0 / np.log(1 + max_peak)
log_data = c * np.log(abs_data)


# Find two points within 90% of the max peak of the scaled image
max_scaled_peak = np.max(log_data)

# Determine the angle of two high-peak points in the image
rows, cols = np.where(log_data > (max_scaled_peak * 0.90))

each_other_distance = np.sqrt((rows[2]-rows[1])**2 + (cols[2]-cols[1])**2)


min_col, max_col = np.min(cols), np.max(cols)
min_row, max_row = np.min(rows), np.max(rows)
dy, dx = max_col - min_col, max_row - min_row
theta = -np.arctan(dy / float(dx))
print(theta)


# Translate and scale the image by the value we found
width, height = gray.shape
cx, cy = width / 2, height / 2
new_image = np.zeros(gray.shape)
for x, row in enumerate(gray):
    for y, value in enumerate(row):
        xp = int(cx + (x - cx) * np.cos(theta) - (y - cy) * np.sin(theta))
        yp = int(cy + (x - cx) * np.sin(theta) + (y - cy) * np.cos(theta))
        if xp < 0 or yp < 0 or xp >= width or yp >= height:
            continue
        new_image[xp, yp] = gray[x, y]


hist = cv2.reduce(new_image,1, cv2.REDUCE_AVG).reshape(-1)

th = 12
H,W = gray.shape[:2]
print(H,W)
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th] 
print(uppers)
print(lowers)


for y in uppers:
    cv2.line(new_image, (0,y), (W, y), (255,0,0), 1)

for y in lowers:
    cv2.line(new_image, (0,y), (W, y), (0,0,255), 1)   
    
plt.subplot(121), plt.imshow(gray, 'gray'), plt.title('Original')  
plt.subplot(122), plt.imshow(new_image, 'gray'), plt.title('Result')  
plt.show()
