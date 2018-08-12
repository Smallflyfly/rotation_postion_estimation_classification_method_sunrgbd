import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

xmlpath = './data/VOCdevkit2007/VOC2007/Annotations/'
imgpath = './data/VOCdevkit2007/VOC2007/JPEGImages/'

test_name = '000001'

img = imgpath + test_name + '.jpg'
xml = xmlpath + test_name + '.xml'
img = cv2.imread(img)

tree = ET.parse(xml)
objs = tree.findall('object')
num_objs = len(objs)
gt_points2d = np.zeros((num_objs, 16), dtype=np.float32)
for ix, obj in enumerate(objs):
  points2d = obj.find('points2d')  #fang
  px1 = float(points2d.find('x1').text)
  py1 = float(points2d.find('y1').text)
  px2 = float(points2d.find('x2').text)
  py2 = float(points2d.find('y2').text)
  px3 = float(points2d.find('x3').text)
  py3 = float(points2d.find('y3').text)
  px4 = float(points2d.find('x4').text)
  py4 = float(points2d.find('y4').text)
  px5 = float(points2d.find('x5').text)
  py5 = float(points2d.find('y5').text)
  px6 = float(points2d.find('x6').text)
  py6 = float(points2d.find('y6').text)
  px7 = float(points2d.find('x7').text)
  py7 = float(points2d.find('y7').text)
  px8 = float(points2d.find('x8').text)
  py8 = float(points2d.find('y8').text)
  gt_points2d[ix, :] = [px1, py1, px2, py2, px3, py3, px4, py4, px5, py5, px6, py6, px7, py7, px8, py8]


  mindata = -199.901211
  maxdata = 899.993907

  j = 0
  for i in range(8):
    x = gt_points2d[ix, j:j+1]
    y = gt_points2d[ix, j+1:j+2]
    print(x,y)
    x = int((x -mindata) / (maxdata - mindata))
    y = int((y -mindata) / (maxdata - mindata))
    j = j + 2
    cv2.circle(img, (x,y), 3, (255, 0, 255), 2)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# plt.scatter(x, y)
# plt.show()