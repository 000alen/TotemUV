import cv2
import numpy


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

image = cv2.imread("samples/6 (3, 4).jpg")
image = cv2.resize(image, (480, 640))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces_rect = face_cascade.detectMultiScale(image, 1.1, 5)
# faces = tuple(image[y:y + h, x:x + w] for x, y, w, h in faces_rect)
# for i, face in enumerate(faces):
#     cv2.imshow(f"face {i}", face)

eyes_rect = eye_cascade.detectMultiScale(image, 1.1, 5)
# eyes = tuple(image[y:y + h, x:x + w] for x, y, w, h in eyes_rect)
# for i, eye in enumerate(eyes):
#     cv2.imshow(f"eye {i}", eye)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

faces_segmentation = []
for rect in faces_rect:
    background_model = numpy.zeros((1, 65), numpy.float64)
    foreground_model = numpy.zeros((1, 65), numpy.float64)

    mask = numpy.zeros(image.shape[:2], numpy.uint8)
    # for x, y, w, h in eyes_rect:
    #     mask[y:y + h, x:x + w] = 3

    cv2.grabCut(image, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    segmentation = image * mask2[:, :, numpy.newaxis]
    faces_segmentation.append(segmentation)

for i, segmentation in enumerate(faces_segmentation):
    segmentation = numpy.where(segmentation == (0, 0, 0), (0, 255, 0), segmentation).astype("uint8")
    cv2.imwrite(f"segmentation {i}.jpg", segmentation)

# cv2.imshow("image", image)
# cv2.waitKey()
