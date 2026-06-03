import cv2
import imutils
import numpy as np

from utils import four_point_transform


def scan_document(image):

    ratio = image.shape[0] / 500.0

    original = image.copy()

    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edged = cv2.Canny(
        gray,
        75,
        200
    )

    contours = cv2.findContours(
        edged.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = imutils.grab_contours(contours)

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )[:5]

    document_contour = None

    for contour in contours:

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:
            document_contour = approx
            break

    if document_contour is None:
        return None

    warped = four_point_transform(
        original,
        document_contour.reshape(4, 2) * ratio
    )

    warped = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    scanned = cv2.adaptiveThreshold(
        warped,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return scanned