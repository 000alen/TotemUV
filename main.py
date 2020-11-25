import cv2
import numpy
import dlib

from fractions import Fraction
from matplotlib import pyplot
from imutils import face_utils
from typing import Tuple

Rect = Tuple[int, int, int, int]

# Indica el rango de indices correspondientes a los puntos de los elementos faciales definidos en:
# https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
dlib_facial_landmarks = (
    ("mouth", 48, 68),
    ("right_eyebrow", 17, 22),
    ("left_eyebrow", 22, 27),
    ("right_eye", 36, 42),
    ("left_eye", 42, 48),
    ("nose", 27, 35),
)


def face_segment(_image: numpy.ndarray, _face_rect: Rect) -> numpy.ndarray:
    """Calcula la segmentacion de la cara dada su ubicacion."""
    background_model = numpy.zeros((1, 65), numpy.float64)
    foreground_model = numpy.zeros((1, 65), numpy.float64)
    _mask = numpy.zeros(_image.shape[:2], numpy.uint8)
    cv2.grabCut(_image, _mask, _face_rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((_mask == 2) | (_mask == 0), 0, 1).astype("uint8")
    return cv2.cvtColor(_image * mask2[:, :, numpy.newaxis], cv2.COLOR_BGR2GRAY)


def face_threshold(_image: numpy.ndarray, _face_features_rect: Tuple[Rect]) -> numpy.ndarray:
    """Calcula el umbral de la cara exceptuando en las zonas de los elementos faciales."""
    _image = cv2.adaptiveThreshold(_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    for _x, _y, _w, _h in _face_features_rect:
        _image[_y:_y + _h, _x:_x + _w] = 0
    return _image


def face_remove_contour(_image: numpy.ndarray, _face_contour) -> numpy.ndarray:
    """Elimina el contorno facial de la imagen."""
    _mask = numpy.ones(_image.shape[:2], dtype="uint8") * 255
    cv2.drawContours(_mask, [_face_contour], -1, 0, 10)
    return cv2.bitwise_and(_image, _image, mask=_mask)


# Procesamiento inicial
image: numpy.ndarray = cv2.imread("samples/6 (3, 4).jpg")
ratio: Fraction = Fraction(image.shape[0], image.shape[1])
image: numpy.ndarray = cv2.resize(image, (480, int(480 / ratio.denominator * ratio.numerator)))

# Deteccion facial
# Se utiliza un modelo pre-entrenado para la deteccion facial. Luego, se calcula la media de los puntajes de confianza
# a modo de umbral para eliminar falsos positivos.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
faces_rect, _, faces_weight = face_cascade.detectMultiScale3(image, 1.1, 7, outputRejectLevels=True)
faces_weight_threshold = numpy.mean(faces_weight)
faces_rect: Tuple[Rect] = tuple(
    j
    for i, j in enumerate(faces_rect)
    if faces_weight[i] >= faces_weight_threshold
)

# Deteccion de elementos faciales
# Se utiliza un modelo pre-entrenado para la deteccion de elementos faciales. Se calcula el sector donde se ubican
# dicchos elementos.
dlib_face_detector = dlib.get_frontal_face_detector()
dlib_face_shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
dlib_faces_rectangle = dlib_face_detector(image, 1)
faces_features_rect = tuple(
    tuple(
        cv2.boundingRect(
            numpy.array([face_utils.shape_to_np(dlib_face_shape_predictor(image, face_rectangle))[j:k]])
        )
        for name, j, k in dlib_facial_landmarks
    )
    for i, face_rectangle in enumerate(dlib_faces_rectangle)
)

# Segmentacion facial
# Se utiliza un algoritmo de segmentacion (GrabCut) para delimitar la cara.
faces_segmentation: Tuple[numpy.ndarray, ...] = tuple(
    face_segment(image, i)
    for i in faces_rect
)

# Umbral facial
# Se aplica un umbral inteligente para encontrar desviaciones en el color tipico de la piel.
faces_threshold: Tuple[numpy.ndarray, ...] = tuple(
    face_threshold(j, faces_features_rect[i])
    for i, j in enumerate(faces_segmentation)
)

# Deteccion y eliminacion de contorno facial
# Se aplica un algoritmo para la busqueda de contornos en la cara. Luego, se elimina el contorno de mayor tamaño, que
# corresponde a la silueta de la cara.
faces_contours = tuple(
    cv2.findContours(i, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    for i in faces_threshold
)
faces_contours_max = tuple(
    max(faces_contours[i], key=lambda contour: cv2.arcLength(contour, True))
    for i in range(len(faces_contours))
)
faces_threshold_no_contour: Tuple[numpy.ndarray, ...] = tuple(
    face_remove_contour(j, faces_contours_max[i])
    for i, j in enumerate(faces_threshold)
)

# Transformacion morfologica de opening
# Se aplica una transformacion morfologica de opening para eliminar el ruido experimental en el umbral de la piel
# calculado anteriormente.
faces_opening: Tuple[numpy.ndarray, ...] = tuple(
    cv2.morphologyEx(i, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
    for i in faces_threshold_no_contour
)

# Transformacion morfologica de dilate
# Se aplica una transformacion morfologica de dilatacion para remarcar las detecciones en el umbral de la piel depurado.
faces_dilate: Tuple[numpy.ndarray, ...] = tuple(
    cv2.dilate(i, numpy.ones((5, 5), numpy.uint8), iterations=2)
    for i in faces_opening
)

# Visualizacion de las detecciones
# Se sobrepone el umbral sin depurar (color verde) y las detecciones realizadas (color rojo) sobre la imagen original.
outputs = []
for i in range(len(faces_dilate)):
    dilate = cv2.cvtColor(faces_dilate[i], cv2.COLOR_GRAY2BGR)
    threshold_no_contour = cv2.cvtColor(faces_threshold_no_contour[i], cv2.COLOR_GRAY2BGR)
    mask_red = numpy.where(dilate != (0, 0, 0), (0, 0, 255), (0, 0, 0)).astype("uint8")
    mask_green = numpy.where(threshold_no_contour != (0, 0, 0), (0, 255, 0), (0, 0, 0)).astype("uint8")
    mask = numpy.where(mask_red != (0, 0, 0), mask_red, mask_green).astype("uint8")
    output = numpy.where(mask != (0, 0, 0), mask, image).astype("uint8")
    outputs.append(output)

plot_face_features = image.copy()
for x, y, w, h in faces_features_rect[0]:
    cv2.rectangle(plot_face_features, (x, y), (x + w, y + h), (255, 0, 0), 2)

plot_face_contour = image.copy()
cv2.drawContours(plot_face_contour, [faces_contours_max[0]], -1, 255, 10)

plots = (
    ("Entrada", image),
    ("Salida", outputs[0]),
    ("Elementos", plot_face_features),
    ("Segmentacion", faces_segmentation[0]),
    ("Umbral", faces_threshold[0]),
    ("Contorno", plot_face_contour),
    ("Umbral (sin contorno)", faces_threshold_no_contour[0]),
    ("Opening", faces_opening[0]),
    ("Dilate", faces_dilate[0])
)

for i, (title, img) in enumerate(plots):
    cv2.imwrite("outputs/" + title + ".jpg", img)
    pyplot.subplot(3, 3, i + 1)
    pyplot.imshow(img)
    pyplot.title(title)
    pyplot.xticks([])
    pyplot.yticks([])

pyplot.show()
