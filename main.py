import cv2
import numpy

FACE_MARGIN_WIDTH = 0
FACE_MARGIN_HEIGHT = 20

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0)


def get_faces_rects(image):
    return tuple(
        (
            max(0, x - FACE_MARGIN_WIDTH),
            max(0, y - FACE_MARGIN_HEIGHT),
            image.shape[1] - x if x + w + FACE_MARGIN_WIDTH > image.shape[1] else w + FACE_MARGIN_WIDTH,
            image.shape[0] - y if y + h + FACE_MARGIN_HEIGHT > image.shape[0] else h + FACE_MARGIN_HEIGHT
        )
        for x, y, w, h in face_cascade.detectMultiScale(image, 1.1, 4)
    )


def get_face_mask(image, rect):
    mask = numpy.zeros(image.shape[:2], numpy.uint8)
    background_model = numpy.zeros((1, 65), numpy.float64)
    foreground_model = numpy.zeros((1, 65), numpy.float64)
    cv2.grabCut(image, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")    
    return image * mask2[:, :, numpy.newaxis]


while True:
    _, image = capture.read()
    faces_rects = get_faces_rects(image)

    canvas = image.copy()
    for x, y, w, h in faces_rects:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 1)

    cv2.imshow("canvas", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if cv2.waitKey(1) % 0xFF == ord("c") and faces_rects:
        mask = get_face_mask(image, faces_rects[0])
        cv2.imshow("mask", mask)

capture.release()
cv2.destroyAllWindows()
