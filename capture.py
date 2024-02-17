#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:19:02 2023

@author: mbaloglu
Stereo cam calibration more information visit :https://albertarmea.com/post/opencv-stereo-camera/
:https://medium.com/analytics-vidhya/distance-estimation-cf2f2fd709d8#:~:text=Stereo%20vision%20is%20a%20technique,of%20rays%20from%20multiple%20viewpoints
"""
import numpy as np
import cv2
import os

LEFT_PATH = "capture/left/{:06d}.jpg"
RIGHT_PATH = "capture/right/{:06d}.jpg"

os.makedirs("capture/left/", exist_ok= True)
os.makedirs("capture/right/", exist_ok= True)

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
left = cv2.VideoCapture(0)
right = cv2.VideoCapture(2)

# Increase the resolution
left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960
def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]

frameId = 0

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    if not (left.grab() and right.grab()):
        print("No more frames")
        break

    _, leftFrame = left.retrieve()
    leftFrame = cropHorizontal(leftFrame)
    _, rightFrame = right.retrieve()
    rightFrame = cropHorizontal(rightFrame)

    cv2.imwrite(LEFT_PATH.format(frameId), leftFrame)
    cv2.imwrite(RIGHT_PATH.format(frameId), rightFrame)

    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frameId += 1

left.release()
right.release()
cv2.destroyAllWindows()