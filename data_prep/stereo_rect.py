import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Left camera
K1 = np.zeros((3, 3, 1), dtype="float")
K1[0, 0] = 590.4370145490491
K1[1, 1] = 588.181350239478
K1[0, 2] = 371.2452432653464
K1[1, 2] = 279.43789818681057
K1[2, 2] = 1
d1 = np.array([-0.2447722457948818, 0.11217603867590234, -0.0002902568544790136, -0.0003128038660481224], dtype="float")

# Right camera
K2 = np.zeros((3, 3, 1), dtype="float")
K2[0, 0] = 590.0371754733331
K2[1, 1] = 588.0160129549419
K2[0, 2] = 374.42794728986536
K2[1, 2] = 282.4914830123564
K2[2, 2] = 1
d2 = np.array([-0.2434466779887321, 0.10061534626749788, 0.00015979811794067017, -0.0007687882687650286], dtype="float")

# Relative transformation
R = np.array([[0.9999997192033521, 0.0002629467748130683, -0.000701749389925231],
              [-0.00026173748958397715, 0.9999984817608755, 0.0017227795640100704],
              [0.0007022013238319235, -0.0017225954061357116, 0.9999982697876848]], dtype="float")

T = np.array([-0.3018526145427248, 0.0008278606830287073, 0.002950378546332348], dtype="float")

# Right camera in left camera (so inverse)
R = np.transpose(R)
T = np.matmul(-R, T)

# Computing stereo rectification
R1 = np.zeros((3, 3), dtype="float")
R2 = np.zeros((3, 3), dtype="float")
P1 = np.zeros((3, 4), dtype="float")
P2 = np.zeros((3, 4), dtype="float")
cv.stereoRectify(K1, d1, K2, d2, (728, 544), R, T, R1, R2, P1, P2, alpha=0)

# Printing results
np.set_printoptions(precision=9)
np.set_printoptions(suppress=True)

print('R1 = np.', repr(R1))
print('P1 = np.', repr(P1))
print('R2 = np.', repr(R2))
print('P2 = np.', repr(P2))
print('\n\n')
print('R1 = np.', repr(np.reshape(R1, [-1])))
print('P1 = np.', repr(np.reshape(P1, [-1])))
print('R2 = np.', repr(np.reshape(R2, [-1])))
print('P2 = np.', repr(np.reshape(P2, [-1])))
