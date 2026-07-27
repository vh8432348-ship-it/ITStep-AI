import cv2
import numpy as np
import matplotlib.pyplot as plt


#Завдання 1

img = cv2.imread("data/lesson3/sonet.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

binary = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

kernel = np.ones((3, 3), np.uint8)

clean = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Gray")

plt.subplot(1, 4, 2)
plt.imshow(blur, cmap="gray")
plt.title("Blur")

plt.subplot(1, 4, 3)
plt.imshow(binary, cmap="gray")
plt.title("Adaptive")

plt.subplot(1, 4, 4)
plt.imshow(clean, cmap="gray")
plt.title("Clean")

plt.show()



#Завдання 2

img = cv2.imread("data\lesson3\sonet_noised.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(gray, 5)

binary = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

kernel = np.ones((3, 3), np.uint8)

clean = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Gray noised")

plt.subplot(1, 4, 2)
plt.imshow(blur, cmap="gray")
plt.title("Median blur")

plt.subplot(1, 4, 3)
plt.imshow(binary, cmap="gray")
plt.title("Adaptive")

plt.subplot(1, 4, 4)
plt.imshow(clean, cmap="gray")
plt.title("Clean result")

plt.show()