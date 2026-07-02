import cv2

img = cv2.imread("data/lesson3/notes.png", cv2.IMREAD_GRAYSCALE)


_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary", binary)

adaptive = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)
cv2.imshow("Adaptive Binary", adaptive)


kernels = [3, 5, 11]
sigmas = [0, 2, 10]

for k in kernels:
    for sigma in sigmas:

        blur = cv2.GaussianBlur(img, (k, k), sigma)

        cv2.imshow(f"Gaussian k={k} sigma={sigma}", blur)

        _, binary_blur = cv2.threshold(
            blur,
            127,
            255,
            cv2.THRESH_BINARY
        )

        cv2.imshow(
            f"Binary k={k} sigma={sigma}",
            binary_blur
        )

        adaptive_blur = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        cv2.imshow(
            f"Adaptive k={k} sigma={sigma}",
            adaptive_blur
        )

bilateral = cv2.bilateralFilter(
    img,
    9,
    75,
    75
)

cv2.imshow("Bilateral", bilateral)


_, binary_bilateral = cv2.threshold(
    bilateral,
    127,
    255,
    cv2.THRESH_BINARY
)

cv2.imshow(
    "Binary after Bilateral",
    binary_bilateral
)

# Адаптивная бинаризация после Bilateral
adaptive_bilateral = cv2.adaptiveThreshold(
    bilateral,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

cv2.imshow(
    "Adaptive after Bilateral",
    adaptive_bilateral
)

cv2.waitKey(0)
cv2.destroyAllWindows()