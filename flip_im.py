import cv2

pic = '006154.jpg'
im = cv2.imread(pic)
im2 = im[:, ::-1, :]

cv2.imshow('img1', im)
cv2.imshow('img2', im2)

cv2.waitKey(0)
cv2.destroyAllWindows()