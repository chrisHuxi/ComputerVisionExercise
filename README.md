# Computer Vision Exercise
exercise of CV course in TUD

## week1 : nonlinear filter
### including :   
  * median filter (with opencv; with my implement)
  * min filter(naive; with PriorityQueue)
  * bilateral filter(with opencv; with my implement)
  
  
[source code click here](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week1_nonlinear_filter/nonlinear_filter.py)


We compare different implement of those filter, and test their time. The result as following:


![](https://github.com/chrisHuxi/ComputerVisionExercise/blob/master/week1_nonlinear_filter/result.PNG)


### key point:
  * my implement of median filter cost much more time then opencv, if anyone has an idea about how to improve it, please 
    contact with me.
  * when the kernel size of median filter < 70, it is faster to use quick sorting rather then histogram. because the average steps of 1       time calculate with histogram method ist 128, while the quick sorting is n\*log(n), so when n is 
    small, sorting method is the better choice.
  * my implemet of min filter with PriorityQueue is much slower then naive. Any idea about it?
  * my implemet of bilateral filter is a little bit faster then opencv, i also don't konw why.
   
