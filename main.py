import cv2
from ultralytics import YOLO


video = cv2.VideoCapture(
    "data/lesson8/animals.mp4"
)


ret, frame = video.read()


if ret:

    cv2.imshow(
        "First frame",
        frame
    )

    cv2.waitKey(0)



model = YOLO("yolo11n.pt")


results = model(
    frame,
    conf=0.5,
    iou=0.4
)


image = results[0].plot()


cv2.imshow(
    "YOLO detection",
    image
)


for box in results[0].boxes:

    x1, y1, x2, y2 = box.xyxy[0]

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)


    crop = frame[y1:y2, x1:x2]


    cv2.imshow(
        "Detected object",
        crop
    )

    cv2.waitKey(0)


cv2.destroyAllWindows()

video.release()

# Завдання 2
model = YOLO("yolo11n.pt")
results = model.track(frame, persist=True)


for result in results:
    for box in result.boxes:
        if box.id is not None:
            print("Знайдений ID:", int(box.id[0]))


target_id = int(input("Введіть ID об'єкта: "))


while True:

    ret, frame = video.read()

    if not ret:
        break


    results = model.track(frame, persist=True)


    for result in results:

        for box in result.boxes:

            if box.id is None:
                continue


            object_id = int(box.id[0])


            if object_id == target_id:

                x1, y1, x2, y2 = map(int, box.xyxy[0])


                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    3
                )


                cv2.putText(
                    frame,
                    f"ID {object_id}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )


    cv2.imshow("YOLO Tracking", frame)


    if cv2.waitKey(1) == 27:
        break


video.release()
cv2.destroyAllWindows()
