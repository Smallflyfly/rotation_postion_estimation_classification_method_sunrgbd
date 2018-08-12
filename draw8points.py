import os
import cv2
import numpy as np

test_result =np.array([0.4862614, 0.6379785, 0.4979368, 0.40192792, 0.49265927, 0.7253109,
  0.4972735, 0.45378903, 0.49765223, 0.6647498, 0.49417174, 0.40361905,
  0.49453893, 0.78572834, 0.4893658, 0.47917533, 0.8111751,  0.7281012,
  1.1302465]).astype(float)

points = test_result[:19]
l = points[-3]
w = points[-2]
h = points[-1]

img = cv2.imread('014.jpg')
maxdata = 899.993907
mindata = -199.90121
l = img.shape[1]
w = img.shape[0]
print(l,w)

points = points * (maxdata - mindata) + mindata
j = 0
print(points)
# fang[-1]
for i in range(8):

    x = int(points[j])
    y = int(points[j+1])
    
    cv2.circle(img, (x, y), 2, (255, 0, 255), 3)
    j += 2

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()