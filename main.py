import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("data/lesson1/Lenna.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask1 = cv2.imread("data/lesson1/mask1.png", cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread("data/lesson1/mask2.png", cv2.IMREAD_GRAYSCALE)


mask_or = cv2.bitwise_or(mask1, mask2)

plt.figure(figsize=(5,5))
plt.imshow(mask_or, cmap="gray")
plt.title("Об'єднана маска")
plt.axis("off")
plt.show()

mask1_bool = mask1.astype(bool)
mask2_bool = mask2.astype(bool)
mask_or_bool = mask_or.astype(bool)

img_mask1 = np.zeros_like(image)
img_mask2 = np.zeros_like(image)
img_mask_or = np.zeros_like(image)

img_mask1[mask1_bool] = image[mask1_bool]
img_mask2[mask2_bool] = image[mask2_bool]
img_mask_or[mask_or_bool] = image[mask_or_bool]

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img_mask1)
plt.title("Mask1")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img_mask2)
plt.title("Mask2")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_mask_or)
plt.title("Mask1 OR Mask2")
plt.axis("off")

plt.show()