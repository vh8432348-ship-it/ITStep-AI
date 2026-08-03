import cv2
# Завдання 1
# video_path = r"data\lesson8\meetings.mp4"
#
# cap = cv2.VideoCapture(video_path)
#
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     frame = cv2.resize(frame, (800, 450))
#
#     boxes, weights = hog.detectMultiScale(
#         frame,
#         winStride=(8, 8),
#         padding=(8, 8),
#         scale=1.05
#     )
#
#     for (x, y, w, h) in boxes:
#         cv2.rectangle(
#             frame,
#             (x, y),
#             (x+w, y+h),
#             (0, 255, 0),
#             2
#         )
#
#     cv2.putText(
#         frame,
#         f"People: {len(boxes)}",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 0, 255),
#         2
#     )
#
#     cv2.imshow("People detection", frame)
#
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
#
#
# cap.release()
# cv2.destroyAllWindows()

# Завдання 2

video_path = r"data\lesson8\meetings.mp4"

cap = cv2.VideoCapture(video_path)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

start_show = False


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame = cv2.resize(frame, (800, 450))


    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8,8),
        padding=(8,8),
        scale=1.05
    )


    people = len(boxes)


    if people >= 5:
        start_show = True


    if start_show:

        for (x,y,w,h) in boxes:
            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )

        cv2.putText(
            frame,
            f"People: {people}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        cv2.imshow(
            "Video from 5 people",
            frame
        )


    if cv2.waitKey(20) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()