#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:03:04 2023

@author: mbaloglu
Stereo cam calibration more information visit :https://albertarmea.com/post/opencv-stereo-camera/
:https://medium.com/analytics-vidhya/distance-estimation-cf2f2fd709d8#:~:text=Stereo%20vision%20is%20a%20technique,of%20rays%20from%20multiple%20viewpoints
"""




import cv2
import numpy as np

right = cv2.VideoCapture("/dev/video2")
right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Genişlik
right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Yükseklik
right.set(cv2.CAP_PROP_FPS,30)


left = cv2.VideoCapture("/dev/video0")
left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Genişlik
left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Yükseklik
left.set(cv2.CAP_PROP_FPS,30)

res=(640,720)

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
name_right = "output/right/video-right.mp4"
out_right = cv2.VideoWriter(name_right, fourcc, 30, (1280,720))
name_left = "output/left/video-left.mp4"
out_left = cv2.VideoWriter(name_left, fourcc, 30, (1280,720))

while(True):
    # if not (left.grab() ):
    #     print("No more frames")
    #     break
   
    status2, rightFrame = right.read()
    
    status, leftFrame = left.read()
    
    
    if status or status2:
        #re_rightFrame=cv2.resize(rightFrame,res)
        #re_leftFrame=cv2.resize(leftFrame,res)
        #hor=np.hstack((leftFrame,rightFrame))
        cv2.imshow('left', leftFrame)
        leftFrame=cv2.resize(leftFrame, (1280,720))
        out_left.write(leftFrame)        
        cv2.imshow("right",rightFrame)
        rightFrame=cv2.resize(rightFrame, (1280,720))
        out_right.write(rightFrame)
    else:
        print("Somethings went wrong")
        print(f"Left status : {status} and Right Status : {status2}")
        break
    # if status==False and status2==False:
    #     print("all false")
    #     break
    # if status2:
        
    #     cv2.imshow('right', rightFrame)
        
    # if status:
        
    #     cv2.imshow('left', leftFrame)
        
    
    # cv2.imshow('left', leftFrame)
    # cv2.imshow('right', rightFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.release()
right.release()
out_left.release()
out_right.release()
cv2.destroyAllWindows()