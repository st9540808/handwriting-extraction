import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

img = cv2.imread('result0.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_,alpha = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY)
b, g, r = cv2.split(img)
rgba = [b,g,r, alpha]
transparent = cv2.merge(rgba, 4)

cv2.imwrite('test.png', transparent)
# plt.imshow(alpha, cmap='gray')
# plt.show()
