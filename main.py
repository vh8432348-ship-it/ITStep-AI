import cv2
from ultralytics import YOLO
# Завдання 1

video = cv2.VideoCapture("data/lesson_pose/sitting.mp4")

success, frame = video.read()
#
# if success:
#     frame = cv2.resize(frame, (805, 630))
#
#     cv2.imshow("First frame", frame)
#
#     cv2.waitKey(0)
#
# video.release()
# cv2.destroyAllWindows()
# Завдання 2

model = YOLO("yolo11n-pose.pt")

results = model.predict(frame, device="cpu")

result = results[0]

# print(result)

# Завдання 3

result_image = result.plot()
# cv2.imshow( 'result', result_image)
# cv2.waitKey(0)
# Завдання 4
print(result.keypoints)

xy = result.keypoints.xy.cpu().numpy()


# Завдання 4
# left_knee = xy[0][13]
# left_hand = xy[0][9]
# right_hand = xy[0][10]
#
# cv2.circle(result_image, (int(left_knee[0]), int(left_knee[1])), 20, (0, 255, 0), -1)
# cv2.circle(result_image, (int(left_hand[0]), int(left_hand[1])), 8, (0, 0, 255), -1)
# cv2.circle(result_image, (int(right_hand[0]), int(right_hand[1])), 8, (255, 255, 255), -1)
#
# cv2.imshow("result", result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Завдання 5
left_knee = xy[1][13]
left_hand = xy[1][9]
right_hand = xy[1][10]

cv2.circle(result_image, (int(left_knee[0]), int(left_knee[1])), 10, (0, 255, 0), -1)
cv2.circle(result_image, (int(left_hand[0]), int(left_hand[1])), 9, (0, 0, 255), -1)
cv2.circle(result_image, (int(right_hand[0]), int(right_hand[1])), 8, (255, 255, 255), -1)

cv2.imshow("result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(left_knee)
print(left_hand)
print(right_hand)
# Завдання 6
while True:
    success, frame = video.read()

    if not success:
        break

    results = model.predict(frame, device="cpu")
    result = results[0]

    xy = result.keypoints.xy.cpu().numpy()

    left_knee = xy[0][13]
    left_hand = xy[0][9]
    right_hand = xy[0][10]

    cv2.circle(
        frame,
        (int(left_knee[0]), int(left_knee[1])),
        10,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        frame,
        (int(left_hand[0]), int(left_hand[1])),
        9,
        (0, 0, 255),
        -1
    )

    cv2.circle(
        frame,
        (int(right_hand[0]), int(right_hand[1])),
        8,
        (255, 255, 255),
        -1
    )

    cv2.imshow("result", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

# Завдання 6

count = 0
sitting = False

while True:
    success, frame = video.read()

    if not success:
        break

    results = model.predict(frame, device="cpu")
    result = results[0]

    xy = result.keypoints.xy.cpu().numpy()

    left_knee = xy[0][13]
    left_hand = xy[0][9]
    right_hand = xy[0][10]

    if left_hand[1] > left_knee[1]:
        if not sitting:
            count += 1
            sitting = True
    else:
        sitting = False

    cv2.circle(
        frame,
        (int(left_knee[0]), int(left_knee[1])),
        10,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        frame,
        (int(left_hand[0]), int(left_hand[1])),
        9,
        (0, 0, 255),
        -1
    )

    cv2.circle(
        frame,
        (int(right_hand[0]), int(right_hand[1])),
        8,
        (255, 255, 255),
        -1
    )

    cv2.putText(
        frame,
        f"Squats: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("result", frame)
    print(count)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()