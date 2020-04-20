# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

In this project, my goal was to write a software pipeline to identify the lane boundaries in a video

---

### The project

### 1. The pipeline. 

I approached the problem in a systematic way, creating a pipeline divided in different steps and that is easy to follow:

1. **Camera calibration:** 
	Created a camera calibration function to calibrate the camera and get the distortion coefficients using the chessboard images located in the camera_cal directory. I didn't use the pickle file to store the coefficients, I stored the values in memory.
	
	![alternate text](/output_images/camera_calibration.png "Camera Calibration")


* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

In order to draw a single line on the left and right lanes, I didn't make any mayor modification to the *draw_lines()* function, instead of adding the lines to the image I returned an image with both lines. I added a pre-process function called **average-slope-intercept()**. This function is used to find the the x and y coordinates for both left and right lanes.

To show the lanes I combine the original image and the image return by the *draw_lines()* function using the *weighted_img()* function.




### 2. Potential shortcomings in the current pipeline


One potential shortcoming would be what would happen when there is not enough data to draw a line. 

Another shortcoming could be when lanes are not properly marked on the road.


### 3. Possible improvements to the pipeline

A possible improvement would be to find a way to draw a line when data is missing. For example if the data from the right lane is better used this data to draw the left lane.

Another potential improvement could be to deal with curves. 