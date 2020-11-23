import cv2
import numpy

from fractions import Fraction


def get_segmentation(image, rect):
    background_model = numpy.zeros((1, 65), numpy.float64)
    foreground_model = numpy.zeros((1, 65), numpy.float64)
    mask = numpy.zeros(image.shape[:2], numpy.uint8)
    cv2.grabCut(image, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return cv2.cvtColor(image * mask2[:, :, numpy.newaxis], cv2.COLOR_BGR2GRAY)


# Initial image processing
image = cv2.imread("samples/6 (3, 4).jpg")
ratio = Fraction(image.shape[1], image.shape[0])
image = cv2.resize(image, (480, int(480 / ratio.numerator * ratio.denominator)))

# Face tracking and segmentation
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
faces_rect = face_cascade.detectMultiScale(image, 1.1, 5)
faces_segmentation = tuple(get_segmentation(image, rect) for rect in faces_rect)
faces_segmentation_blur = tuple(cv2.GaussianBlur(segmentation, (5, 5), 0) for segmentation in faces_segmentation)
faces_threshold = tuple(
    cv2.adaptiveThreshold(segmentation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    for segmentation in faces_segmentation_blur
)

output = []
for i, threshold in enumerate(faces_threshold):
    x, y, w, h = faces_rect[i]
    output = numpy.where(cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR) != 0, (0, 0, 255), image).astype("uint8")
    cv2.imshow(f"output {i}.jpg", output)

cv2.waitKey(0)
