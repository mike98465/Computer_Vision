import numpy as np

def transform(point2D, point3D):
	point2D_append = np.zeros((36,3), dtype=np.int32)
	point3D_append = np.zeros((36,4), dtype=np.int32)
	cnt = 0

	for p1,p2 in zip(point2D,point3D):
		p1 = np.append(p1, 1)
		p2 = np.append(p2, 1)
		point2D_append[cnt] = p1
		point3D_append[cnt] = p2
		cnt = cnt + 1
	
	return point2D_append, point3D_append

