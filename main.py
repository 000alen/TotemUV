import cv2
import numpy


def video_stream():
    capture = cv2.VideoCapture(0)
    while True:
        _, image = capture.read()
        yield image


def get_faces_rects(
        image,
        face_cascade,
        face_margin=(0, 20),
):
    return tuple(
        (
            max(0, x - face_margin[0]),
            max(0, y - face_margin[1]),
            image.shape[1] - x if x + w + face_margin[0] > image.shape[1] else w + face_margin[0],
            image.shape[0] - y if y + h + face_margin[1] > image.shape[0] else h + face_margin[1]
        )
        for x, y, w, h in face_cascade.detectMultiScale(image, 1.1, 4)
    )


def get_face_mask(image, rect):
    mask = numpy.zeros(image.shape[:2], numpy.uint8)
    background_model = numpy.zeros((1, 65), numpy.float64)
    foreground_model = numpy.zeros((1, 65), numpy.float64)
    cv2.grabCut(image, mask, rect, background_model, foreground_model, 7, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")    
    return image * mask2[:, :, numpy.newaxis]


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

image = cv2.imread("samples/6 (3, 4).jpg")
image = cv2.resize(image, (480, 640))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces_rect = face_cascade.detectMultiScale(image, 1.1, 5)
faces = tuple(image[y:y + h, x:x + w] for x, y, w, h in faces_rect)
# faces_mean = tuple(numpy.mean(face) for face in faces)
# faces_mask = tuple(cv2.threshold(face, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] for face in faces)
# faces_edge = tuple(cv2.Canny(face, 0.7 * faces_mean[i], 1.1 * faces_mean[i]) for i, face in enumerate(faces))
for i, face in enumerate(faces):
    cv2.imshow(f"face {i}", face)

eyes_rect = eye_cascade.detectMultiScale(image, 1.1, 5)
eyes = tuple(image[y:y + h, x:x + w] for x, y, w, h in eyes_rect)
for i, eye in enumerate(eyes):
    cv2.imshow(f"eye {i}", eye)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

faces_segmentation = []
for rect in faces_rect:
    background_model = numpy.zeros((1, 65), numpy.float64)
    foreground_model = numpy.zeros((1, 65), numpy.float64)

    mask = numpy.zeros(image.shape[:2], numpy.uint8)
    for x, y, w, h in eyes_rect:
        mask[y:y + h, x:x + w] = 3

    cv2.grabCut(image, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    segmentation = image * mask2[:, :, numpy.newaxis]
    faces_segmentation.append(segmentation)

for i, segmentation in enumerate(faces_segmentation):
    segmentation = numpy.where(segmentation == (0, 0, 0), (0, 255, 0), segmentation).astype("uint8")
    cv2.imshow(f"segmentation {i}", segmentation)

cv2.imshow("image", image)
cv2.waitKey()
