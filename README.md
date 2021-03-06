# Computer Vision Exercise
exercise of CV course in TUD


## week 10_12 : Neural Network


### including :   
  * [P1.py](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week9_CNN/P1.py)
  * [P1_edited.py](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week9_CNN/P1_edited.py)
  
### The result as following : 
  * using W of shape[2,2], and discribe this problem as a classification problem, i.e. using softmax_cross_entropy as loss function,( PS: how to draw a line : y = W * x + b with W of shape[2,2], I still don't know. with [reference](http://learningtensorflow.com/classifying/ ), I got a result as following, but can't explain why there are somewhere are not linear.)
  
  ![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week9_CNN/P1_result.png)


  * using W of shape[2,1], using mean square as loss function
 
  ![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week9_CNN/P1_result2.png)


## week 7-9 : HDR


### including :   
  * [HDR_convert.py](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week7_HDR/HDR_convert.py)
  * [opencv implemention](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week7_HDR/solution/hdr_opencv.cpp)
### The result as following :
  * inversed camera response function: (my implementation used Mitsunagaand Nayar Technique)
  
  ![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week7_HDR/response_function_reverse.png)


  * my result: (I don't know which part got a problem, but I guess it happens during the calculating inversed camera response function. because before I visualize the result of Radiance Map, there are also many black and white dot noise.)
 
  ![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week7_HDR/result.jpg)
  
  
  
  
## week 5-6 : seam carving


### including :   
  * [my implemention](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week5_seam_carving/P1_my.py)
  * [opencv implemention](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week5_seam_carving/seam_carving_skt.py)
### The result as following : reduce 50 pixel in horizontal direction
  * my implemention
  
  ![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week5_seam_carving/my_seam_carving_result.jpg)


  * opencv implemention
 
  ![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week5_seam_carving/skt_seam_carving_result.jpg)
  
  

## week 3-4 : fourier transform

### including :   
  * [exercise description](http://cvl.inf.tu-dresden.de/HTML/teaching/courses/cv2/ss18/Ex/2/CV2_Ex2_Fourier.pdf)
  * [P1.py](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week3_fourier_transform/P1.py)
  * [P2.py](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week3_fourier_transform/P2.py)
  * [P3.py](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week3_fourier_transform/P3.py)
  * `TODO : P4.py`
### The result as following:
  * P1
  
  ![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week3_fourier_transform/result/P1_result.PNG)


  * P2
 
  ![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week3_fourier_transform/result/figure_P2_result.png)
  
  
  * P3
  
  ![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week3_fourier_transform/result/figure_P3_result.png)
  


## week 1-2 : nonlinear filter

### edit:
  * I took a mistake in original code(bilateral filter part) : I actually ran the opencv function when I wanted to test my function. So the time of calculation belongs to opencv, rather then mine. Thanks a lot for suggestion of professor Heidrich.
  
  
### including :   
  * median filter (with opencv; with my implement)
  * min filter(naive; with PriorityQueue)
  * bilateral filter(with opencv; with my implement)
  * addtional work : [guid filter](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week1_nonlinear_filter/addtional_guidFilter.py)
  
 [main source code click here](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week1_nonlinear_filter/nonlinear_filter.py)

  

We compare different implement of those filter, and test their time. The result as following:


![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week1_nonlinear_filter/result.PNG)


### key point:
  * my implement of median filter cost much more time then opencv, if anyone has an idea about how to improve it, please 
    contact with me.
  * when the kernel size of median filter < 70, it is faster to use quick sorting rather then histogram. because the average steps of 1       time calculate with histogram method ist 128, while the quick sorting is n\*log(n), so when n is 
    small, sorting method is the better choice.
  * my implemet of min filter with PriorityQueue is much slower then naive. Any idea about it?
  * my implemet of bilateral filter is a little bit faster then opencv, i also don't konw why.
   
