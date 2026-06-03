import cv2

from scanner import scan_document


image_path = "images/sample.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found")
    exit()

scanned = scan_document(image)

if scanned is None:
    print("Document not detected")
    exit()

cv2.imwrite(
    "outputs/scanned.jpg",
    scanned
)

cv2.imshow(
    "Scanned Document",
    scanned
)

cv2.waitKey(0)
cv2.destroyAllWindows()