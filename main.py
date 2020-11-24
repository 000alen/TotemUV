import cv2
import numpy

from fractions import Fraction
from matplotlib import pyplot


def face_segment(_image, _face_rect):
    background_model = numpy.zeros((1, 65), numpy.float64)
    foreground_model = numpy.zeros((1, 65), numpy.float64)
    mask = numpy.zeros(_image.shape[:2], numpy.uint8)
    cv2.grabCut(_image, mask, _face_rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return cv2.cvtColor(_image * mask2[:, :, numpy.newaxis], cv2.COLOR_BGR2GRAY)


def face_threshold(_image, _face_features_rect):
    _image = cv2.adaptiveThreshold(_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    for _x, _y, _w, _h in _face_features_rect:
        _image[_y:_y + _h, _x:_x + _w] = 0
    return _image


def face_remove_contour(_image, _face_contour):
    mask = numpy.ones(_image.shape[:2], dtype="uint8") * 255
    cv2.drawContours(mask, [_face_contour], -1, 0, -1)
    return cv2.bitwise_and(_image, _image, mask=mask)


# -- Initial image processing --
image = cv2.imread("samples/6 (3, 4).jpg")
ratio = Fraction(image.shape[1], image.shape[0])
image = cv2.resize(image, (480, int(480 / ratio.numerator * ratio.denominator)))

# -- Eye tracking --
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
eyes_rect, _, eyes_weight = eye_cascade.detectMultiScale3(image, 1.1, 7, outputRejectLevels=True)

# Remove possible false-positives
eyes_weight_threshold = numpy.mean(eyes_weight)
eyes_rect = tuple(
    rect
    for i, rect in enumerate(eyes_rect)
    if eyes_weight[i] >= eyes_weight_threshold
)

# -- Mouth tracking --
mouth_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_mouth.xml")
mouths_rect, _, mouths_weight = mouth_cascade.detectMultiScale3(image, 1.1, 7, outputRejectLevels=True)

# Remove possible false-positives
mouths_weight_threshold = numpy.mean(mouths_weight)
mouths_rect = tuple(
    rect
    for i, rect in enumerate(mouths_rect)
    if mouths_weight[i] >= mouths_weight_threshold
)

# -- Nose tracking --
nose_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_nose.xml")
noses_rect, _, noses_weight = nose_cascade.detectMultiScale3(image, 1.1, 7, outputRejectLevels=True)

# Remove possible false-positives
noses_weight_threshold = numpy.mean(noses_weight)
noses_rect = tuple(
    rect
    for i, rect in enumerate(noses_rect)
    if noses_weight[i] >= noses_weight_threshold
)

# -- Face tracking --
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
faces_rect, _, faces_weight = face_cascade.detectMultiScale3(image, 1.1, 7, outputRejectLevels=True)

# Remove possible false-positives
faces_weight_threshold = numpy.mean(faces_weight)
faces_rect = tuple(
    rect
    for i, rect in enumerate(faces_rect)
    if faces_weight[i] >= faces_weight_threshold
)

face_features_rect = eyes_rect + mouths_rect + noses_rect

# -- Face segmentation --
faces_segmentation = tuple(
    face_segment(image, rect)
    for rect in faces_rect
)

faces_threshold = tuple(
    face_threshold(i, face_features_rect)
    for i in faces_segmentation
)

# faces_contours = tuple(
#     cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
#     for threshold in faces_threshold
# )
#
# faces_threshold = tuple(
#     face_remove_contour(threshold, max(faces_contours[i], key=lambda contour: cv2.arcLength(contour, True)))
#     for i, threshold in enumerate(faces_threshold)
# )

faces_opening = tuple(
    cv2.morphologyEx(i, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
    for i in faces_threshold
)

faces_dilate = tuple(
    cv2.dilate(i, numpy.ones((5, 5), numpy.uint8))
    for i in faces_opening
)

output = tuple(
    numpy.where(cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) != 0, (0, 0, 255), image).astype("uint8")
    for i in faces_dilate
)

titles = [
    "Image",
    "Face features",
    "Face features",
    "Output",
    "Face segmentation",
    "Face threshold"
]

face_features_image = image.copy()
for x, y, w, h in face_features_rect:
    cv2.rectangle(face_features_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

images = [
    image,
    face_features_image,
    face_features_image,
    output[0],
    faces_segmentation[0],
    faces_threshold[0]
]

for i in range(6):
    pyplot.subplot(2, 3, i + 1)
    pyplot.imshow(images[i])
    pyplot.title(titles[i])
    pyplot.xticks([])
    pyplot.yticks([])

pyplot.show()
