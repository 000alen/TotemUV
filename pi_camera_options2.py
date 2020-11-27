#!/usr/bin/python
# --------------------------------------
#    ___  ___  _ ____
#   / _ \/ _ \(_) __/__  __ __
#  / , _/ ___/ /\ \/ _ \/ // /
# /_/|_/_/  /_/___/ .__/\_, /
#                /_/   /___/
#
# pi_camera_options2.py
# Takes a sequence of photos with the Pi camera
# using the PiCamera library with a range of Exposure and
# White Balance settings.
#
# Please see :
# https://www.raspberrypi-spy.co.uk/2013/06/testing-multiple-pi-camera-options-with-python/
# for more information.
#
# PiCamera library docs
# http://picamera.readthedocs.io/
#
# Author : Matt Hawkins
# Date   : 14/06/2018
#
# --------------------------------------

import os
import time
import picamera

# Get valid Exposure and White Balance values
valid_ex  = picamera.PiCamera.EXPOSURE_MODES
valid_awb = picamera.PiCamera.AWB_MODES

# Valid Exposure and AWB values
print("\nValid exposure values:\n[", end='')
for value in valid_ex:
    print("'"+value+"',", end='')
print("]")

print("\nValid AWB values:\n[", end='')
for value in valid_awb:
    print("'"+value+"',", end='')
print("]")

# Test list of Exposure and White Balance options. 9 photos.
list_ex  = ['off','auto','backlight']
list_awb = ['off','auto','sunlight']

# Specified Exposure and AWB values
print("\nSpecified exposure values:",list_ex)
print("Specified AWB values:",list_awb)

# Photo dimensions and rotation
photo_width  = 640
photo_height = 480
photo_rotate = 90

photo_interval = 0.5 # Interval between photos (seconds)
photo_counter  = 0   # Photo counter

total_photos = len(list_ex) * len(list_awb)

# Delete all previous image files
try:
  os.remove("photo_*.jpg")
except OSError:
  pass

camera = picamera.PiCamera()
camera.rotation=photo_rotate
camera.resolution=(photo_width,photo_height)
camera.annotate_background = picamera.Color('black')
camera.annotate_foreground = picamera.Color('white')

# Lets start taking photos!
try:

  print("\nStarting photo sequence")

  for ex in list_ex:
    for awb in list_awb:
      photo_counter = photo_counter + 1
      filename = 'photo_' + ex + '_' + awb + '.jpg'
      print(' [' + str(photo_counter) + ' of ' + str(total_photos) + '] ' + filename) 
      camera.awb_mode=awb
      camera.exposure_mode=ex
      camera.annotate_text = filename
      camera.capture(filename)
      time.sleep(photo_interval)
  
  print("Finished photo sequence")
  
except KeyboardInterrupt:
  # User quit
  print("\nGoodbye!")
