
import cv2
import numpy as np
# Завдання 1
img = cv2.imread("data/lesson1/Lenna.png")


print("Размер (shape):", img.shape)

print("Тип данных:", img.dtype)

print("Минимум:", np.min(img))
print("Максимум:", np.max(img))

cv2.imshow("Lenna Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
# Завдання 2

img = cv2.imread("data/lesson1/Lenna.png")

h, w = img.shape[:2]

top_left = img[0:50, 0:100]

center_x, center_y = w // 2, h // 2
center = img[center_y-50:center_y+50, center_x-50:center_x+50]

top_half = img[0:h//2, :]

bottom_half = img[h//2:, :]

left_half = img[:, 0:w//2]

right_half = img[:, w//2:]

cv2.imshow("Top Left 100x50", top_left)
cv2.imshow("Center 100x100", center)
cv2.imshow("Top Half", top_half)
cv2.imshow("Bottom Half", bottom_half)
cv2.imshow("Left Half", left_half)
cv2.imshow("Right Half", right_half)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Завдання 4
img = cv2.imread("data/lesson1/Lenna.png")

# делаем черно-белое
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w = gray.shape

img1 = gray.copy()
img1 = cv2.copyMakeBorder(
    img1,
    top=50,
    bottom=50,
    left=0,
    right=0,
    borderType=cv2.BORDER_CONSTANT,
    value=[0]
)


img1[-50:, :] = 255


img2 = gray.copy()
img2 = cv2.copyMakeBorder(
    img2,
    top=0,
    bottom=0,
    left=50,
    right=50,
    borderType=cv2.BORDER_CONSTANT,
    value=[0]
)

img2[:, -50:] = 255

img3 = gray.copy()


img3[:30, :] = 0
img3[-30:, :] = 0

img3[:, :30] = 0
img3[:, -30:] = 0

cv2.imshow("Gray", gray)
cv2.imshow("Top-Bottom", img1)
cv2.imshow("Left-Right", img2)
cv2.imshow("Frame Overlay", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()