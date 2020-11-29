from picamera import PiCamera
from time import sleep

list_awb = ["antishake", "backlight", "nightpreview", "verylong"]
list_exposure = ["incandescent"]
list_iso = [800]

camera = PiCamera()

for awb in list_awb:
    for exposure in list_exposure:
        for iso in list_iso:
            camera.awb_mode = awb
            camera.exposure_mode = exposure
            camera.iso = iso
            camera.capture("_".join([awb, exposure, str(iso)]) + ".jpg")
            sleep(0.5)
