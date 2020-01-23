import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Circle
from PIL import Image

point2D = np.load('Point2D_two.npy')
point2D_obj1 = point2D[0:4,0:2]
point2D_obj2 = point2D[4:8,0:2]

### homogeneous coordinates ###
point2D_obj1_append = np.zeros((4,3), dtype=np.int32)
point2D_obj2_append = np.zeros((4,3), dtype=np.int32)
cnt = 0

for p1,p2 in zip(point2D_obj1,point2D_obj2):
	p1 = np.append(p1, 1)
	p2 = np.append(p2, 1)
	point2D_obj1_append[cnt] = p1
	point2D_obj2_append[cnt] = p2
	cnt = cnt + 1
### homogeneous coordinates ###

### projection matrix ###
cnt2 = 0
matrix_A = np.zeros((8,8))
b = np.zeros(8)
for p2_1,p2_2 in zip(point2D_obj1_append,point2D_obj2_append):
	#print(p2_1)
	#print(p2_2)
	matrix_A[cnt2]= [p2_2[0], p2_2[1], 1, 0, 0, 0, -p2_1[0]*p2_2[0], -p2_1[0]*p2_2[1]]
	b[cnt2] = p2_1[0]
	cnt2 = cnt2 + 1
	matrix_A[cnt2]= [0, 0, 0, p2_2[0], p2_2[1], 1, -p2_1[1]*p2_2[0], -p2_1[1]*p2_2[1]]
	b[cnt2] = p2_1[1]
	cnt2 = cnt2 + 1

x = np.linalg.solve(matrix_A.T.dot(matrix_A), matrix_A.T.dot(b))
x = np.append(x,1)
#print(x.shape)
H = np.reshape(x,(3,3))

#print(H)
### projection matrix ###

### forward warping ###
img = cv2.imread("../data/two_objects.jpg")
area1 = np.array([[point2D_obj2[0][0],point2D_obj2[0][1]],
				[point2D_obj2[1][0],point2D_obj2[1][1]],
				[point2D_obj2[3][0],point2D_obj2[3][1]],
				[point2D_obj2[2][0],point2D_obj2[2][1]]])
area2 = np.array([[point2D_obj1[0][0],point2D_obj1[0][1]],
				[point2D_obj1[1][0],point2D_obj1[1][1]],
				[point2D_obj1[3][0],point2D_obj1[3][1]],
				[point2D_obj1[2][0],point2D_obj1[2][1]]])
mask_img = cv2.fillPoly(img, [area1], (255, 255, 255))
mask_img = cv2.fillPoly(mask_img, [area2], (254, 254, 254))
mask_img = Image.fromarray(cv2.cvtColor(mask_img,cv2.COLOR_BGR2RGB)) 
mask_pix = mask_img.load()

forward_img = Image.open('../data/two_objects.jpg')
change_pix = forward_img.load()

backup_img = Image.open('../data/two_objects.jpg')
backup_pix = backup_img.load()

for y in range(img.shape[0]):
	for x in range(img.shape[1]):
		if(mask_pix[x,y] == (255, 255, 255)):
			before = np.zeros((3,1))
			before[0] = x
			before[1] = y
			before[2] = 1
			after = np.dot(H,before)
			after[0] = np.round((after[0]/after[2]))
			after[1] = np.round((after[1]/after[2]))
			change_pix[int(after[0]),int(after[1])] = backup_pix[x,y]

for y in range(img.shape[0]):
	for x in range(img.shape[1]):
		if(mask_pix[x,y] == (254, 254, 254)):
			before = np.zeros((3,1))
			before[0] = x
			before[1] = y
			before[2] = 1
			after = np.dot(np.linalg.inv(H),before)
			after[0] = np.round((after[0]/after[2]))
			after[1] = np.round((after[1]/after[2]))
			change_pix[int(after[0]),int(after[1])] = backup_pix[x,y]

forward_img.save("forward.jpg")

### forward warping ### 

'''
#change_img = np.asarray(change_img)  
#blurred_img = cv2.GaussianBlur(cv2.cvtColor(change_img,cv2.COLOR_RGB2BGR), (21, 21), 0) 
#for y in range(1080):
	#for x in range(1920):
		#if test_pix[x,y] == (255,255,255):	
#out = np.where(test_pix==np.array([255, 255, 255]), cv2.cvtColor(blurred_img,cv2.COLOR_BGR2RGB), change_img)
#out = Image.fromarray(out)
#out.save('modify.png')
#test.save('test.png')
'''

### backward warping ###

backward_img = Image.open('../data/two_objects.jpg')
change_pix1 = backward_img.load()

for y in range(img.shape[0]):
	for x in range(img.shape[1]):
		if(mask_pix[x,y] == (254, 254, 254)):
			before = np.zeros((3,1))
			before[0] = x
			before[1] = y
			before[2] = 1
			after = np.dot(np.linalg.inv(H),before)
			after[0] = np.round((after[0]/after[2]))
			after[1] = np.round((after[1]/after[2]))
			change_pix1[x,y] = backup_pix[int(after[0]),int(after[1])]
			
for y in range(img.shape[0]):
	for x in range(img.shape[1]):
		if(mask_pix[x,y] == (255, 255, 255)):
			before = np.zeros((3,1))
			before[0] = x
			before[1] = y
			before[2] = 1
			after = np.dot(H,before)
			after[0] = np.round((after[0]/after[2]))
			after[1] = np.round((after[1]/after[2]))
			change_pix1[x,y] = backup_pix[int(after[0]),int(after[1])]

backward_img.save("backward.jpg")

### backward warping ###

### 2-C ###
point2D_object1 = np.load('Point2D_obj1.npy')
point2D_object2 = np.load('Point2D_obj2.npy')

point2D_object1_append = np.zeros((4,3), dtype=np.int32)
point2D_object2_append = np.zeros((4,3), dtype=np.int32)
cnt1 = 0

for p1,p2 in zip(point2D_object1,point2D_object2):
	p1 = np.append(p1, 1)
	p2 = np.append(p2, 1)
	point2D_object1_append[cnt1] = p1
	point2D_object2_append[cnt1] = p2
	cnt1 = cnt1 + 1

cnt3 = 0
matrix_B = np.zeros((8,8))
b1 = np.zeros(8)
for p2_1,p2_2 in zip(point2D_object1_append,point2D_object2_append):
	#print(p2_1)
	#print(p2_2)
	matrix_B[cnt3]= [p2_2[0], p2_2[1], 1, 0, 0, 0, -p2_1[0]*p2_2[0], -p2_1[0]*p2_2[1]]
	b1[cnt3] = p2_1[0]
	cnt3 = cnt3 + 1
	matrix_B[cnt3]= [0, 0, 0, p2_2[0], p2_2[1], 1, -p2_1[1]*p2_2[0], -p2_1[1]*p2_2[1]]
	b1[cnt3] = p2_1[1]
	cnt3 = cnt3 + 1

x1 = np.linalg.solve(matrix_B.T.dot(matrix_B), matrix_B.T.dot(b1))
x1 = np.append(x1,1)
#print(x.shape)
H1 = np.reshape(x1,(3,3))
#print(H1)

'''
#test
error_sum = 0.0
n = 0
result = np.zeros((4,2))
for p2,p3 in zip(point2D_object1_append, point2D_object2_append):
	after = H1.dot(p3.T)
	after = after/after[2]
	result[n] = [after[0],after[1]]
	n += 1	
	error_sum += mean_squared_error(after,p2)

	error_avg = error_sum/n
print("RMSE: " , error_avg)
img = plt.imread("../data/object1.jpg")
fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(img)
for ori,res in zip(point2D_object1,result):
	circ = Circle((ori[0],ori[1]),2,color='y')
	#circ1 = Circle((res[0],res[1]),2,color='r')
	ax.add_patch(circ)
	#ax.add_patch(circ1)
	
#plt.figure()
plt.savefig('evaluation.png')
plt.show()
'''

### object1 ###
img1 = cv2.imread("../data/object1.jpg")
area3 = np.array([[point2D_object1[0][0],point2D_object1[0][1]],
				[point2D_object1[1][0],point2D_object1[1][1]],
				[point2D_object1[3][0],point2D_object1[3][1]],
				[point2D_object1[2][0],point2D_object1[2][1]]])

mask_img1 = cv2.fillPoly(img1, [area3], (254, 254, 254))
mask_img1 = Image.fromarray(cv2.cvtColor(mask_img1,cv2.COLOR_BGR2RGB)) 
mask_pix1 = mask_img1.load()

forward_img1 = Image.open('../data/object1.jpg')
change_pix1 = forward_img1.load()

backup_img1 = Image.open('../data/object1.jpg')
backup_pix1 = backup_img1.load()
### object1 ###

### object2 ###
img2 = cv2.imread("../data/object2.jpg")
area4 = np.array([[point2D_object2[0][0],point2D_object2[0][1]],
				[point2D_object2[1][0],point2D_object2[1][1]],
				[point2D_object2[3][0],point2D_object2[3][1]],
				[point2D_object2[2][0],point2D_object2[2][1]]])

mask_img2 = cv2.fillPoly(img2, [area4], (255, 255, 255))
mask_img2 = Image.fromarray(cv2.cvtColor(mask_img2,cv2.COLOR_BGR2RGB)) 
mask_pix2 = mask_img2.load()

forward_img2 = Image.open('../data/object2.jpg')
change_pix2 = forward_img2.load()

backup_img2 = Image.open('../data/object2.jpg')
backup_pix2 = backup_img2.load()
### object2 ###

### object1 forward ###
for y in range(img2.shape[0]):
	for x in range(img2.shape[1]):
		if(mask_pix1[x,y] == (254, 254, 254)):
			before = np.zeros((3,1))
			before[0] = x
			before[1] = y
			before[2] = 1
			after = np.dot(np.linalg.inv(H1),before)
			after[0] = np.round((after[0]/after[2]))
			after[1] = np.round((after[1]/after[2]))
			change_pix2[int(after[0]),int(after[1])] = backup_pix1[x,y]

forward_img2.save("forward1.jpg")
### object1 forward ###

### object2 forward ###
for y in range(img1.shape[0]):
	for x in range(img1.shape[1]):
		if(mask_pix2[x,y] == (255, 255, 255)):
			before = np.zeros((3,1))
			before[0] = x
			before[1] = y
			before[2] = 1
			after = np.dot(H1,before)
			after[0] = np.round((after[0]/after[2]))
			after[1] = np.round((after[1]/after[2]))
			change_pix1[int(after[0]),int(after[1])] = backup_pix2[x,y]
			
forward_img1.save("forward2.jpg")
### object2 forward ###

### object1 backward ###
backward_img1 = Image.open('../data/object2.jpg')
change_pix3 = backward_img1.load()
backward_img2 = Image.open('../data/object1.jpg')
change_pix4 = backward_img2.load()

for y in range(img1.shape[0]):
	for x in range(img1.shape[1]):
		if(mask_pix2[x,y] == (255, 255, 255)):
			before = np.zeros((3,1))
			before[0] = x
			before[1] = y
			before[2] = 1
			after = np.dot(H1,before)
			after[0] = np.round((after[0]/after[2]))
			after[1] = np.round((after[1]/after[2]))
			change_pix3[x,y] = backup_pix1[int(after[0]),int(after[1])]
			

backward_img1.save("backward1.jpg")
### object1 backward ###

### object2 backward ###
for y in range(img2.shape[0]):
	for x in range(img2.shape[1]):
		if(mask_pix1[x,y] == (254, 254, 254)):
			before = np.zeros((3,1))
			before[0] = x
			before[1] = y
			before[2] = 1
			after = np.dot(np.linalg.inv(H1),before)
			after[0] = np.round((after[0]/after[2]))
			after[1] = np.round((after[1]/after[2]))
			change_pix4[x,y] = backup_pix2[int(after[0]),int(after[1])]

backward_img2.save("backward2.jpg")
### object2 backward ###

### 2-C ###