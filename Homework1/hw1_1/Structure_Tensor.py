from scipy.signal import convolve2d
from Gaussian_Smooth import gaussian_smooth
from matplotlib import pyplot as plt
import numpy as np
import cv2

def structure_tensor(gradient10_x,gradient10_y,window_size,height,width):
	#k value from p.76
	k = 0.04
	#Ixx = gaussian_smooth(gradient10_x**2,size=10,show=0)
	#Ixy = gaussian_smooth(gradient10_y * gradient10_x,size=10,show=0)
	#Iyy = gaussian_smooth(gradient10_y**2,size=10,show=0)
	
	Ixx = gradient10_x**2
	Ixy = gradient10_y * gradient10_x
	Iyy = gradient10_y**2
	
	offset = int(window_size/2)
	
	harris_response = np.zeros([height,width])
	
	for y in range(offset, height-offset):
		for x in range(offset, width-offset):
			Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])/(window_size**2)
			Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])/(window_size**2)
			Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])/(window_size**2)
	
			det = (Sxx * Syy) - (Sxy**2)
			trace = Sxx + Syy
			r = det - k*(trace**2)
			harris_response[y,x] = r
	
	#print(harris_response)
	
	return harris_response