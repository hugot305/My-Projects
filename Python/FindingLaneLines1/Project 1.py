#In[0]
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 07:37:59 2020

@author: Hugo Trinidad
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

prev_left_fit=[]
prev_right_fit=[]
prev_x1 = 0
prev_y1 = 0
prev_x2 = 0
prev_y2 = 0


def get_xy_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
        
    #print('image shape:', image.shape)
    #print('slope:', slope)
    #print('intercept:', intercept)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    global prev_left_fit
    global prev_right_fit    
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
            
    if len(right_fit) > 0:        
        prev_right_fit = right_fit
    else:
        right_fit = prev_right_fit
        
    if len(left_fit) > 0:
        prev_left_fit = left_fit
    else:
        left_fit = prev_left_fit            
            
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = get_xy_coordinates(image, left_fit_average)
    right_line = get_xy_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (250,) * channel_count
    else:
        ignore_mask_color = 250
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color, thickness):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    global prev_x1
    global prev_y1
    global prev_x2
    global prev_y2
    
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            
            if y1 > image.shape[0] or y2 > image.shape[0]:
                print('y:', line)
                x1 = prev_x1
                x2 = prev_x2
                y1 = prev_y1
                y2 = prev_y2
            else:    
                if y1 < 0 or  y2 < 0:
                    print('y:', line)
                    x1 = prev_x1
                    x2 = prev_x2
                    y1 = prev_y1
                    y2 = prev_y2   
                
            if x1 > image.shape[1] or x2 > image.shape[1]:
                print('x:', line)
                x1 = prev_x1
                x2 = prev_x2
                y1 = prev_y1
                y2 = prev_y2
            else:   
                if x1 < 0 or  x2 < 0:
                    print('x:', line)
                    x1 = prev_x1
                    x2 = prev_x2
                    y1 = prev_y1
                    y2 = prev_y2
                                             
            prev_x1 = x1
            prev_x2 = x2
            prev_y1 = y1
            prev_y2 = y2    
            
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
                    
    return line_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    return lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

# In[1]:
#reading in an image
images = []
input_directory = 'test_images/'

images.append('solidWhiteCurve.jpg')
images.append('solidWhiteRight.jpg')
images.append('solidYellowCurve.jpg')
images.append('solidYellowCurve2.jpg')
images.append('solidYellowLeft.jpg')
images.append('whiteCarLaneSwitch.jpg')

selected_image = images[1]
image = mpimg.imread(input_directory + selected_image)
original_image = np.copy(image)

gray_image = grayscale(original_image)
kernel_size = 5
blur_image = gaussian_blur(gray_image, kernel_size)

low_threshold = 50
high_threshold = 150
canny_image = canny(blur_image, low_threshold, high_threshold)

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 325), (490, 315), (imshape[1],imshape[0])]])
cropped_image = region_of_interest(canny_image, vertices)

rho = 2
theta = np.pi/180
threshold = 15
min_line_lenght = 30
max_line_gap = 20

lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, min_line_lenght, max_line_gap )

averaged_line = average_slope_intercept(original_image, lines)

color=[255,0,0]
thickness = 15

image_with_lines = draw_lines(original_image, averaged_line, color, thickness)

combine_image = weighted_img(image_with_lines, original_image, 0.8, 1., 0.)

plt.imshow(combine_image)

# In[2]:
output_directory = 'images_output/'
image_to_write = cv2.cvtColor(combine_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_directory + selected_image, image_to_write)
    
def process_image(image):
    lane_image = np.copy(image)

    gray_image = grayscale(lane_image)
    kernel_size = 5
    blur_image = gaussian_blur(gray_image, kernel_size)

    low_threshold = 50
    high_threshold = 150
    canny_image = canny(blur_image, low_threshold, high_threshold)

    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 325), (490, 315), (imshape[1],imshape[0])]], dtype=np.int32)
    cropped_image = region_of_interest(canny_image, vertices)

    rho = 0.75
    theta = np.pi/180
    threshold = 15
    min_line_lenght = 30
    max_line_gap = 20

    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, min_line_lenght, max_line_gap )

    averaged_line = average_slope_intercept(lane_image, lines)

    color=[255,0,0]
    thickness = 15

    image_with_lines = draw_lines(lane_image, averaged_line, color, thickness)

    result = cv2.addWeighted(lane_image, 0.8, image_with_lines, 1, 1 )

    return result


# In[3]
video = 'test_videos/solidYellowLeft.mp4'
#video = 'test_videos/solidWhiteRight.mp4'
cap = cv2.VideoCapture(video)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        image_with_lines = process_image(frame)
        cv2.imshow('result', image_with_lines)
        if cv2.waitKey(20) == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()