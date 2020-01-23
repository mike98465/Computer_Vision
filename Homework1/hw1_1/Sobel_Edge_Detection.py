from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import numpy as np
import cv2

def sobel_edge_detection(img5,img10,show):

	np.seterr(divide='ignore', invalid='ignore')
	kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

	result5_x = convolve2d(img5, kernel_x, mode='same')
	result5_y = convolve2d(img5, kernel_y, mode='same')

	result10_x = convolve2d(img10, kernel_x, mode='same')
	result10_y = convolve2d(img10, kernel_y, mode='same')

	gradient_magnitude5 = np.sqrt(np.square(result5_x) + np.square(result5_y))
	gradient_magnitude10 = np.sqrt(np.square(result10_x) + np.square(result10_y))
	
	#gradient_magnitude5 *= 255.0 / gradient_magnitude5.max()
	#gradient_magnitude10 *= 255.0 / gradient_magnitude10.max()
	direction_5 = np.arctan(np.divide(result5_y,result5_x))
	#print(direction_5)
	direction_5[gradient_magnitude5 < 80] = -1.5
	direction_10 = np.arctan(np.divide(result10_y,result10_x))
	#print(direction_10)
	direction_10[gradient_magnitude10 < 60] = -1.5
	
	if show == 1:
		plt.title("magnitude_5")
		plt.imshow(gradient_magnitude5,cmap=plt.get_cmap('gray'))
		plt.savefig("results/magnitude_5.png")
		plt.show()
		
		plt.title("direction_5")
		plt.imshow(direction_5,cmap=plt.get_cmap('nipy_spectral'))
		plt.savefig("results/direction_5.png")
		plt.show()
		
		plt.title("magnitude_10")
		plt.imshow(gradient_magnitude10,cmap=plt.get_cmap('gray'))
		plt.savefig("results/magnitude_10.png")
		plt.show()
		
		plt.title("direction_10")
		plt.imshow(direction_10,cmap=plt.get_cmap('nipy_spectral'))
		plt.savefig("results/direction_10.png")
		plt.show()

	return result10_x,result10_y