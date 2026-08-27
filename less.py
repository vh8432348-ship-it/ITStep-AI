import cv2
from ultralytics import YOLO
import math

def get_angle(x1, y1, x2, y2, x3, y3):
    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) -
        math.atan2(y1 - y2, x1 - x2)
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


model = YOLO("yolo11n-pose.pt")

video = cv2.VideoCapture("data/lesson_pose/squat.mp4")

count = 0
down = False

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy

        if len(keypoints) > 0 and len(keypoints[0]) >= 16:

            keypoints = keypoints[0]

            hip = keypoints[11]
            knee = keypoints[13]
            ankle = keypoints[15]

            x1, y1 = hip
            x2, y2 = knee
            x3, y3 = ankle

            angle = get_angle(
                x1, y1,
                x2, y2,
                x3, y3
            )

            if angle < 100:
                down = True

            if angle > 160 and down:
                count += 1
                down = False

            cv2.putText(
                frame,
                f"Angle: {angle:.0f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Squats: {count}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Squat", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()