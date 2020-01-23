from matplotlib import pyplot as plt
from PIL import Image
import scipy.ndimage
import numpy as np
import cv2

def nms(img,harris,window,type):
	detection = np.zeros([harris.shape[0],harris.shape[1]]) 
	max_val = scipy.ndimage.maximum_filter(np.copy(harris),3)
	detection[(harris > 100000) & (harris == max_val)] = 1
	img_copy = np.copy(img)
	
	rs = [] 
	cs = []
	
	for i in range(harris.shape[0]):
		for j in range(harris.shape[1]):
			if detection[i][j] == 1:
				img_copy[i,j] = [255,0,0]
				cs.append(i)
				rs.append(j)

				
	if type == None:
		if window == 3:
			plt.title("nms3")
			plt.plot(rs,cs,'r+',markersize=3)
			plt.imshow(img_copy)
			plt.savefig("results/nms3.png")
			plt.show()
		elif window == 30:
			plt.title("nms30")
			plt.plot(rs,cs,'r+',markersize=3)
			plt.imshow(img_copy)
			plt.savefig("results/nms30.png")
			plt.show()
	
	if type == "rotate":
		plt.title("rotate")
		plt.plot(rs,cs,'r+',markersize=3)
		plt.imshow(img_copy)
		plt.savefig("results/rotate.png")
		plt.show()
	elif type == "scale":
		plt.title("scale")
		plt.plot(rs,cs,'r+',markersize=3)
		plt.imshow(img_copy)
		plt.savefig("results/scale.png")
		plt.show()