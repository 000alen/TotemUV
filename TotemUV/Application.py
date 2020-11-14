from math import floor
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2


class CV2Camera(Image):
    def __init__(self, video_capture, fps, **kwargs):
        super().__init__(**kwargs)
        self.video_capture = video_capture
        Clock.schedule_interval(self.update, 1 / fps)

    def update(self, dt):
        ret, frame = self.video_capture.read()
        if ret:
            flipped = cv2.flip(frame, 0)
            flipped = cv2.flip(flipped, 1)
            a = flipped.shape[0] / self.size[1]
            w = a * self.size[0]
            cropped = flipped[
                      :,
                      floor(abs(flipped.shape[1] - w) / 2):floor(abs(flipped.shape[1] + w) / 2)
            ]
            resized = cv2.resize(cropped, (self.size[0], self.size[1]))
            buffer = resized.tostring()
            image_texture = Texture.create(size=(resized.shape[1], resized.shape[0]), colorfmt='rgb')
            image_texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture


class TotemUVApp(App):
    def build(self):
        self.video_capture = cv2.VideoCapture(0)
        self.cv2_camera = CV2Camera(self.video_capture, 10)
        return self.cv2_camera

    def on_stop(self):
        self.video_capture.release()
