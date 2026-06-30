import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('data/lesson2/marbles.png')
#
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# lower_blue = np.array([100, 100, 50])
# upper_blue = np.array([130, 255, 255])
#
# mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
#
# cv2.imshow('Blue', mask_blue)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# lower_green = np.array([40, 50, 50])
# upper_green = np.array([80, 255, 255])
#
# mask_green = cv2.inRange(hsv, lower_green, upper_green)
#
# cv2.imshow('Green', mask_green)
#
# lower_red1 = np.array([0, 100, 50])
# upper_red1 = np.array([10, 255, 255])
#
# mask_red1 = cv2.inRange(hsv,lower_red1,upper_red1)
#
#
# lower_red2 = np.array([170, 100, 50])
# upper_red2 = np.array([180, 255, 255])
#
# mask_red2 = cv2.inRange(hsv, lower_red2,upper_red2)
#
# mask_red = cv2.bitwise_or(mask_red1, mask_red2)
#
# mask_green_red = cv2.bitwise_or(mask_green,
#                                 mask_red)
#
# cv2.imshow('Green + Red',
#            mask_green_red)
#
# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 255, 60])
#
# mask_black = cv2.inRange(hsv,
#                          lower_black,
#                          upper_black)
#
# lower_white = np.array([0, 0, 180])
# upper_white = np.array([180, 50, 255])
#
# mask_white = cv2.inRange(hsv,
#                          lower_white,
#                          upper_white)
#
# mask_all = cv2.bitwise_or(mask_blue,
#                           mask_green_red)
#
# mask_all = cv2.bitwise_or(mask_all,
#                           mask_black)
#
# mask_all = cv2.bitwise_or(mask_all,
#                           mask_white)
#
# cv2.imshow('All marbles',
#            mask_all)
#
# img = cv2.imread('data/lesson2/marbles.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# plt.imshow(img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Завдання 2
img = cv2.imread('data/lesson2/cell.png')

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

l_eq = cv2.equalizeHist(l)

lab_eq = cv2.merge((l_eq, a, b))
result_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

cv2.imshow('Original', img)
cv2.imshow('Histogram Equalization', result_eq)

cv2.waitKey(0)
cv2.destroyAllWindows()