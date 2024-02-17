#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:41:23 2023

@author: mbaloglu
"""
import cv2
import os
cap=cv2.VideoCapture(2)
#for Ip cam :
#cap = cv2.VideoCapture("rtsp://admin:123456@192.168.1.13")
save_path="images-cam/"
os.makedirs(save_path,exist_ok=True)

i=0
while True:
   

    ret,frame=cap.read()
    
    
    if ret:
        frame=cv2.resize(frame,(1920,1080))
        cv2.imshow("Output Frame", frame)
        
        k=cv2.waitKey(1)
        
        if k==ord("q"):
            break
        elif k==ord("s"):
            cv2.imwrite(save_path+f"/test-frame_{str(i).zfill(6)}.png", frame)
            print("image saved!")
           
    else :
        print("somethings went wrong!")
        break
    i+=1 
cap.release()

cv2.destroyAllWindows()    




