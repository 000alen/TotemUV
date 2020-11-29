from picamera import PiCamera

list_exposure = ["antishake", "backlight", "nightpreview", "verylong"]
list_awb = ["incandescent"]
list_iso = [800]

camera = PiCamera()
camera.resolution = (3280, 2464)

for awb in list_awb:
    for exposure in list_exposure:
        for iso in list_iso:
            camera.awb_mode = awb
            camera.exposure_mode = exposure
            camera.iso = iso
            camera.start_recording("_".join([awb, exposure, str(iso)]) + ".h264")
            camera.wait_recording(5)
            camera.stop_recording()
