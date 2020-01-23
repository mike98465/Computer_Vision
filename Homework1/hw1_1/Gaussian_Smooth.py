from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import numpy as np
import cv2

def gaussian_smooth(img,size,show):

	#5*5 Gassian filter with sigma = 5
	if size == 5:
		x5, y5 = np.mgrid[-2:3, -2:3]
		kernel5 = np.exp(-(x5**2+y5**2)/(2*5*5))
		#Normalization
		kernel5 = kernel5 / kernel5.sum()
		result5 = convolve2d(img, kernel5, mode='same')
	
	elif size == 10:
		#10*10 Gassian filter with sigma = 5
		x10, y10 = np.mgrid[-5:5, -5:5]
		kernel10 = np.exp(-(x10**2+y10**2)/(2*5*5))
		#Normalization
		kernel10 = kernel10 / kernel10.sum()
		result10 = convolve2d(img, kernel10, mode='same')
	
	if show == 1:
		if size == 5:
			plt.figure()
			plt.title("gaussian_smooth5")
			plt.imshow(result5,cmap=plt.get_cmap('gray'))
			plt.savefig("results/gaussian_smooth5.png")
			plt.show()
			return result5
		elif size == 10:
			plt.figure()
			plt.title("gaussian_smooth10")
			plt.imshow(result10,cmap=plt.get_cmap('gray'))
			plt.savefig("results/gaussian_smooth10.png")
			plt.show()
			return result10
	
	elif show == 0:
		if size == 5:
			return result5
		elif size == 10:
			return result10