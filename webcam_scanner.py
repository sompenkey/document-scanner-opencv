import cv2

from scanner import scan_document

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow(
        "Original",
        frame
    )

    scanned = scan_document(frame)

    if scanned is not None:

        cv2.imshow(
            "Scanned",
            scanned
        )

    key = cv2.waitKey(1)

    if key == ord('s'):

        if scanned is not None:

            cv2.imwrite(
                "outputs/webcam_scan.jpg",
                scanned
            )

            print("Saved!")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

kernel = np.array([
    [-1,-1,-1],
    [-1, 9,-1],
    [-1,-1,-1]
])

scanned = cv2.filter2D(
    scanned,
    -1,
    kernel
)

scanned = cv2.medianBlur(
    scanned,
    3
)

from PIL import Image

img = Image.open(
    "outputs/scanned.jpg"
)

img.save(
    "outputs/document.pdf",
    "PDF"
)