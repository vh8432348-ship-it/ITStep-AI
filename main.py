import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("data/lesson2/darken.png")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_equalized = hsv.copy()
hsv_equalized[:, :, 2] = cv2.equalizeHist(hsv_equalized[:, :, 2])

result_equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2RGB)

hsv_bright = hsv.copy()

value = hsv_bright[:, :, 2].astype(np.float32)
value = value * 1.3
value = np.clip(value, 0, 255)
value = value.astype(np.uint8)

hsv_bright[:, :, 2] = value

result_bright = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2RGB)

original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(original)
plt.title("Оригінал")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(result_equalized)
plt.title("Вирівнювання гістограми")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(result_bright)
plt.title("Value +30%")
plt.axis("off")

plt.tight_layout()
plt.show()