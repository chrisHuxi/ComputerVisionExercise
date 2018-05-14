from scipy import signal
import cv2
import matplotlib.pyplot as plt
import numpy as np



img  = cv2.imread("./P2_img/AE[1].png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

template = cv2.imread("./P2_img/E[1].png")
template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

il = []
jl = []
for i in range(template_gray.shape[0]):
    for j in range(template_gray.shape[1]):
        if template_gray[i,j] != 0:
            il.append(i)
            jl.append(j)
            
i_max = (max(il))
i_min = (min(il))
j_max = (max(jl))
j_min = (min(jl))

print(i_max,j_max,i_min,j_min)
            


template_gray = template_gray[i_min:i_max,j_min:j_max]

w,h = template_gray.shape[::-1] 
print(w)
print(h)


res = cv2.matchTemplate(gray,template_gray,cv2.TM_CCOEFF_NORMED) 
threshold = 0.8
loc = np.where(res >= threshold)
print(len(loc[0]))
for pt in zip(*loc[::-1]): 
    right_bottom = (pt[0] + w, pt[1] + h) 
    cv2.rectangle(img, pt, right_bottom, (0, 0, 255), 2)

plt.imshow(img)
plt.show()
