import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Circle

def evaluate(projection, point2D_append, point3D_append, point2D, filename, num):
	error_sum = 0.0
	result = np.zeros((36,2))
	n = 0
	for p2,p3 in zip(point2D_append, point3D_append):
		after = projection.dot(p3.T)
		after = after/after[2]
		result[n] = [after[0],after[1]]
		n += 1	
		error_sum += mean_squared_error(after,p2)

	error_avg = error_sum/n
	print("RMSE: " , error_avg)

	# Show the image
	img = plt.imread(filename)
	fig,ax = plt.subplots(1)
	ax.set_aspect('equal')
	ax.imshow(img)
	for ori,res in zip(point2D,result):
		circ = Circle((ori[0],ori[1]),2,color='y')
		circ1 = Circle((res[0],res[1]),2,color='r')
		ax.add_patch(circ)
		ax.add_patch(circ1)
	
	#plt.figure()
	plt.savefig('evaluation' + str(num) + '.png')
	plt.show()