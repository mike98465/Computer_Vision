import cv2
import numpy as np
from scipy import linalg

def projection(point2D_append, point3D_append):
	cnt = 0
	matrix_A = np.zeros((72,11))
	b = np.zeros(72)
	for p2,p3 in zip(point2D_append,point3D_append):
		matrix_A[cnt]= [p3[0], p3[1], p3[2], 1, 0, 0, 0, 0, -p2[0]*p3[0], -p2[0]*p3[1], -p2[0]*p3[2]]
		b[cnt] = p2[0]
		cnt = cnt + 1
		matrix_A[cnt]= [0, 0, 0, 0, p3[0], p3[1], p3[2], 1, -p2[1]*p3[0], -p2[1]*p3[1], -p2[1]*p3[2]]
		b[cnt] = p2[1]
		cnt = cnt + 1

	#method1
	x = np.linalg.solve(matrix_A.T.dot(matrix_A), matrix_A.T.dot(b))
	x = np.append(x,1)
	#print(x.shape)
	H = np.reshape(x,(3,4))
	H = H * np.sign(np.linalg.det(H[0:3,0:3]))
	#print(np.sign(np.linalg.det(H[0:3,0:3])))
	#print(H)

	#method2
	'''
	#print(matrix_A.shape)
	#print(b.shape)
	x = np.linalg.lstsq(matrix_A,b,rcond=None)[0]
	x = np.append(x,1)
	#print(x.shape)
	H = np.reshape(x,(3,4))
	H = H * np.sign(np.linalg.det(H[0:3,0:3]))
	#print(np.sign(np.linalg.det(H[0:3,0:3])))
	print(H)
	'''

	#normalization
	K,R = linalg.rq(H[0:3,0:3])
	
	#Kt = H[0:3,3]
	#Kt = Kt/K[2,2]
	#K = K/K[2,2]
	
	D = np.diag(np.sign(np.diag(K)))
	K = np.dot(K,D)
	R = np.dot(D,R)
	
	Kt = H[0:3,3]
	Kt = Kt/K[2,2]
	K = K/K[2,2]
	
	t = np.linalg.solve(K, Kt)
	
	
	KR = np.dot(K,R)

	#p = reconstructed projection matrix
	p = np.zeros((3,4))
	p[0:3,0:3] = KR
	p[0:3,3] = Kt
	p = p / p[2,3]
	#print(p)
	return p,R,t
