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
    cv2.drawContours(mask, [_face_contour], -1, 0, 10)
    return cv2.bitwise_and(_image, _image, mask=mask)


# -- Initial image processing --
image = cv2.imread("samples/6 (3, 4).jpg")
ratio = Fraction(image.shape[0], image.shape[1])
image = cv2.resize(image, (480, int(480 / ratio.denominator * ratio.numerator)))
# ----

# -- Eye tracking --
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
eyes_rect, _, eyes_weight = eye_cascade.detectMultiScale3(image, 1.1, 7, outputRejectLevels=True)

# Remove possible false-positives
eyes_weight_threshold = numpy.mean(eyes_weight)
eyes_rect = tuple(
    j
    for i, j in enumerate(eyes_rect)
    if eyes_weight[i] >= eyes_weight_threshold
)
# ----

# -- Mouth tracking --
mouth_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_mouth.xml")
mouths_rect, _, mouths_weight = mouth_cascade.detectMultiScale3(image, 1.1, 7, outputRejectLevels=True)

# Remove possible false-positives
mouths_weight_threshold = numpy.mean(mouths_weight)
mouths_rect = tuple(
    j
    for i, j in enumerate(mouths_rect)
    if mouths_weight[i] >= mouths_weight_threshold
)
# ----

# -- Nose tracking --
nose_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_nose.xml")
noses_rect, _, noses_weight = nose_cascade.detectMultiScale3(image, 1.1, 7, outputRejectLevels=True)

# Remove possible false-positives
noses_weight_threshold = numpy.mean(noses_weight)
noses_rect = tuple(
    j
    for i, j in enumerate(noses_rect)
    if noses_weight[i] >= noses_weight_threshold
)
# ----

# -- Face tracking --
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
faces_rect, _, faces_weight = face_cascade.detectMultiScale3(image, 1.1, 7, outputRejectLevels=True)

# Remove possible false-positives
faces_weight_threshold = numpy.mean(faces_weight)
faces_rect = tuple(
    j
    for i, j in enumerate(faces_rect)
    if faces_weight[i] >= faces_weight_threshold
)
# ----

face_features_rect = eyes_rect + mouths_rect + noses_rect

# -- Face processing --
faces_segmentation = tuple(
    face_segment(image, i)
    for i in faces_rect
)

faces_threshold = tuple(
    face_threshold(i, face_features_rect)
    for i in faces_segmentation
)

faces_contours = tuple(
    cv2.findContours(i, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    for i in faces_threshold
)

faces_contours_max = tuple(
    max(faces_contours[i], key=lambda contour: cv2.arcLength(contour, True))
    for i in range(len(faces_contours))
)

faces_threshold_no_contour = tuple(
    face_remove_contour(j, faces_contours_max[i])
    for i, j in enumerate(faces_threshold)
)

faces_opening = tuple(
    cv2.morphologyEx(i, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
    for i in faces_threshold_no_contour
)

faces_dilate = tuple(
    cv2.dilate(i, numpy.ones((5, 5), numpy.uint8), iterations=2)
    for i in faces_opening
)

outputs = []
for i in range(len(faces_dilate)):
    dilate = cv2.cvtColor(faces_dilate[i], cv2.COLOR_GRAY2BGR)
    threshold_no_contour = cv2.cvtColor(faces_threshold_no_contour[i], cv2.COLOR_GRAY2BGR)
    mask_red = numpy.where(dilate != (0, 0, 0), (0, 0, 255), (0, 0, 0)).astype("uint8")
    mask_green = numpy.where(threshold_no_contour != (0, 0, 0), (0, 255, 0), (0, 0, 0)).astype("uint8")
    mask = numpy.where(mask_red != (0, 0, 0), mask_red, mask_green).astype("uint8")
    output = numpy.where(mask != (0, 0, 0), mask, image).astype("uint8")
    outputs.append(output)

# -- Plotting --
plot_face_features = image.copy()
for x, y, w, h in face_features_rect:
    cv2.rectangle(plot_face_features, (x, y), (x + w, y + h), (255, 0, 0), 2)

plot_face_contour = image.copy()
cv2.drawContours(plot_face_contour, [faces_contours_max[0]], -1, 255, 10)

plots = (
    ("Input", image),
    ("Output", outputs[0]),
    ("Face features", plot_face_features),
    ("Face segmentation", faces_segmentation[0]),
    ("Face threshold", faces_threshold[0]),
    ("Face contour", plot_face_contour),
    ("Face threshold (no contour)", faces_threshold_no_contour[0]),
    ("Face opening", faces_opening[0]),
    ("Face dilate", faces_dilate[0])
)

for i, (title, img) in enumerate(plots):
    cv2.imwrite("output/" + title + ".jpg", img)
    pyplot.subplot(3, 3, i + 1)
    pyplot.imshow(img)
    pyplot.title(title)
    pyplot.xticks([])
    pyplot.yticks([])

pyplot.show()
