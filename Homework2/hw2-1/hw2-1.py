import cv2
import numpy as np
import matplotlib.pyplot as plt
from homogeneous_coordinate import transform
from projection_matrix import projection
from evaluation import evaluate
from visualize import visualize

point2D = np.load('Point2D.npy')
point3D = np.loadtxt("../data/Point3D.txt",dtype=np.int32)

point2D_2 = np.load('Point2D_2.npy')
point2D_append, point3D_append = transform(point2D, point3D)
point2D_2_append, point3D_append = transform(point2D_2, point3D)

point2D_3 = np.load('Point2D_3.npy')
point2D_3_append, point3D_append = transform(point2D_3, point3D)

point2D_4 = np.load('Point2D_4.npy')
point2D_4_append, point3D_append = transform(point2D_4, point3D)

projection1,R1,t1 = projection(point2D_append, point3D_append)
projection2,R2,t2 = projection(point2D_2_append, point3D_append)
projection3,R3,t3 = projection(point2D_3_append, point3D_append)
projection4,R4,t4 = projection(point2D_4_append, point3D_append)

evaluate(projection1, point2D_append, point3D_append, point2D, "../data/chessboard_1.jpg", 1)
evaluate(projection2, point2D_2_append, point3D_append, point2D_2, "../data/chessboard_2.jpg", 2)
evaluate(projection3, point2D_3_append, point3D_append, point2D_3, "../data/1.jpg", 3)
evaluate(projection4, point2D_4_append, point3D_append, point2D_4, "../data/2.jpg", 4)

visualize(point3D,R1,t1.reshape([3,1]),R2,t2.reshape([3,1]))
visualize(point3D,R3,t3.reshape([3,1]),R4,t4.reshape([3,1]))