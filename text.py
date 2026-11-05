import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("data/lesson_seg/brain-tumor-seg.pt")

img = cv2.imread("data/lesson_seg/tumor1.jpg")


results = model.predict(img, conf=0.25)

result = results[0]

if result.masks is None:
    print("Пухлина не виявлена")
    exit()

masks = result.masks.data.cpu().numpy()

mask = np.any(masks > 0.5, axis=0).astype(np.uint8) * 255

mask = cv2.resize(
    mask,
    (img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

pixel_area = cv2.countNonZero(mask)

cm_area = pixel_area * 0.0025

if cm_area < 10:
    tumor_type = "small"
elif cm_area <= 25:
    tumor_type = "middle"
else:
    tumor_type = "large"

output = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite(f"{tumor_type}.jpg", output)

print("Площадь в пікселях:", pixel_area)
print("Площадь в см^2:", cm_area)
print("Тип пухлини:", tumor_type)

cv2.imshow("Segmentation", result.plot())
cv2.imshow("Tumor mask", mask)
cv2.imshow("Tumor only", output)

cv2.waitKey(0)
cv2.destroyAllWindows()