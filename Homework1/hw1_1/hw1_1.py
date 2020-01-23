from matplotlib import pyplot as plt
from Gaussian_Smooth import gaussian_smooth
from Sobel_Edge_Detection import sobel_edge_detection
from Structure_Tensor import structure_tensor
from NMS import nms
from scipy import ndimage
import numpy as np
import cv2

#a. gaussian_smooth()
img = plt.imread('original.jpg')
img_rotate = ndimage.rotate(img,30)
img_scale = cv2.resize(img,None,fx=0.5,fy=0.5)

height = img.shape[0]
width = img.shape[1]

gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray_rotate = cv2.cvtColor(img_rotate,cv2.COLOR_RGB2GRAY)
gray_scale = cv2.cvtColor(img_scale,cv2.COLOR_RGB2GRAY)

after_img5 = gaussian_smooth(gray_img,size=5,show=1)
after_img10 = gaussian_smooth(gray_img,size=10,show=1)
after_rotate5 = gaussian_smooth(gray_rotate,size=5,show=0)
after_rotate10 = gaussian_smooth(gray_rotate,size=10,show=0)
after_scale5 = gaussian_smooth(gray_scale,size=5,show=0)
after_scale10 = gaussian_smooth(gray_scale,size=10,show=0)

#b. sobel_edge_detection()
gradient10_x, gradient10_y = sobel_edge_detection(after_img5,after_img10,show=1)
rotate10_x, rotate10_y = sobel_edge_detection(after_rotate5,after_rotate10,show=0)
scale10_x, scale10_y = sobel_edge_detection(after_scale5,after_scale10,show=0)

#c. structure_tensor()
harris_3 = structure_tensor(gradient10_x,gradient10_y,3,height,width)
harris_30 = structure_tensor(gradient10_x,gradient10_y,30,height,width)
harris_rotate_3 = structure_tensor(rotate10_x,rotate10_y,3,img_rotate.shape[0],img_rotate.shape[1])
harris_scale_3 = structure_tensor(scale10_x,scale10_y,3,img_scale.shape[0],img_scale.shape[1])

#d. nms()
nms(img,harris_3,window=3,type=None)
nms(img,harris_30,window=30,type=None)
nms(img_rotate,harris_rotate_3,window=3,type="rotate")
nms(img_scale,harris_scale_3,window=3,type="scale")