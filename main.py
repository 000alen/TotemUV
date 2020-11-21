import cv2
import numpy

from fractions import Fraction

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

image = cv2.imread("samples/3 (4, 3).jpg")
ratio = Fraction(image.shape[1], image.shape[0])
image = cv2.resize(image, (640, int(640 / ratio.numerator * ratio.denominator)))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces_rect = face_cascade.detectMultiScale(image, 1.1, 5)
eyes_rect = eye_cascade.detectMultiScale(image, 1.1, 5)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

faces_segmentation = []
for face_rect in faces_rect:
    background_model = numpy.zeros((1, 65), numpy.float64)
    foreground_model = numpy.zeros((1, 65), numpy.float64)
    mask = numpy.zeros(image.shape[:2], numpy.uint8)
    cv2.grabCut(image, mask, face_rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    segmentation = image * mask2[:, :, numpy.newaxis]
    faces_segmentation.append(segmentation)

for i, segmentation in enumerate(faces_segmentation):
    segmentation = numpy.where(segmentation == (0, 0, 0), (0, 255, 0), segmentation).astype("uint8")
    cv2.imwrite(f"segmentation {i}.jpg", segmentation)

