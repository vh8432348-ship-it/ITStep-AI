import cv2

# Завдання 1
video = cv2.VideoCapture("data/lesson7/text.mp4")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

new_width = width // 2
new_height = height // 4

writer = cv2.VideoWriter(
    "data/lesson7/new_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (new_width, new_height)
)

while True:
    ret, frame = video.read()

    if not ret:
        break

    resized_frame = cv2.resize(frame, (new_width, new_height))

    cv2.imshow("Video", resized_frame)

    writer.write(resized_frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


video.release()
writer.release()
cv2.destroyAllWindows()

# Завдання 2

video = cv2.VideoCapture("data/lesson7/text.mp4")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)


writer = cv2.VideoWriter(
    "data/lesson7/binary_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
    False
)


while True:
    ret, frame = video.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    cv2.imshow("Binary video", binary)

    writer.write(binary)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


video.release()
writer.release()
cv2.destroyAllWindows()
# Завдання 3

video = cv2.VideoCapture("data/lesson7/shapes.mp4")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)


writer = cv2.VideoWriter(
    "data/lesson7/edges_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
    False
)


while True:
    ret, frame = video.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    cv2.imshow("Edges", edges)

    writer.write(edges)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


video.release()
writer.release()
cv2.destroyAllWindows()