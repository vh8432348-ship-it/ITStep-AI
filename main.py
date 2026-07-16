import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("data/lesson_seg/crop-seg.pt")

image = cv2.imread("data/lesson_seg/crop3.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model(image)

result = results[0]

for i in range(len(result.boxes)):
    class_id = int(result.boxes.cls[i])
    name = result.names[class_id]

    mask = result.masks.data[i].cpu().numpy()

    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    mask = mask > 0.5

    plant = np.full_like(image, 255)
    plant[mask] = image[mask]

    plt.figure(figsize=(5, 5))
    plt.imshow(plant)
    plt.title(name)
    plt.axis("off")
    plt.show()