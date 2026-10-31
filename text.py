import cv2

cap = cv2.VideoCapture("data/lesson7/meter.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "data/lesson7/meter_binary.mp4",
    fourcc,
    fps,
    (width, height),
    isColor=False
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    _, binary = cv2.threshold(
        filtered,
        127,
        255,
        cv2.THRESH_BINARY
    )

    out.write(binary)

cap.release()
out.release()

cv2.destroyAllWindows()


