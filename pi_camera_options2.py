import os
import time
import picamera

# Get valid Exposure and White Balance values
valid_ex = picamera.PiCamera.EXPOSURE_MODES
valid_awb = picamera.PiCamera.AWB_MODES
valid_iso = list(range(100, 801, 100))

# Valid Exposure and AWB values
print("Valid exposure values: [")
for value in valid_ex:
    print(value + ", ", end="")
print("]")

print("Valid AWB values: [")
for value in valid_awb:
    print(value + ", ", end="")
print("]")

print("Valid ISO values: [")
for value in valid_iso:
    print(str(value) + ", ", end="")
print("]")

# Test list of Exposure and White Balance options. 9 photos.
list_ex = valid_ex
list_awb = valid_awb
list_iso = valid_iso

# Specified Exposure and AWB values
print("\nSpecified exposure values:", list_ex)
print("Specified AWB values:", list_awb)

# Photo dimensions and rotation
# photo_width = 640
# photo_height = 480
# photo_rotate = 90

photo_interval = 0.5  # Interval between photos (seconds)
photo_counter = 0  # Photo counter

total_photos = len(list_ex) * len(list_awb) * len(list_iso)

# Delete all previous image files
try:
    os.remove("photo_*.jpg")
except OSError:
    pass

camera = picamera.PiCamera()
# camera.rotation = photo_rotate
# camera.resolution = (photo_width, photo_height)

print("\nStarting photo sequence")

for ex in list_ex:
    for awb in list_awb:
        for iso in list_iso:
            photo_counter += 1
            filename = "photo_" + ex + "_" + awb + "_" + str(iso) + ".jpg"
            print(" [" + str(photo_counter) + " of " + str(total_photos) + "] " + filename)
            camera.awb_mode = awb
            camera.exposure_mode = ex
            camera.iso = iso
            camera.capture(filename)
            time.sleep(photo_interval)

print("Finished photo sequence")
