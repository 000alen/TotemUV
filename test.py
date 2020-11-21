import picamera

camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.start_recording("test.mp4")
camera.wait_recording(10)
camera.stop_recording()
