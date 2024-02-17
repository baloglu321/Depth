#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:25:23 2023

@author: mbaloglu

Camera Zed 2i 
For install zed sdk visit :https://www.stereolabs.com/docs
"""
import pyzed.sl as sl
import cv2
import numpy as np
import math
import os

class OpenCVYoloDetector:
    def __init__(self, network_size=(512, 512), confidence=0.4, device_id=0):
        self.confidence = confidence
        self.network_size = network_size
        self.net = None
        self.output_layers = None
        self.net_initialized = False
        self.device_id = device_id

    def run(self):
        # Load YOLO

        cv2.cuda.setDevice(self.device_id)
        self.net = cv2.dnn.readNetFromDarknet(
            "yolov4-pf-v21.cfg",
            "yolov4-pf-v21_last.weights",
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]

    def detect(self, frame):
        if not self.net_initialized:
            self.run()
            self.net_initialized = True
        height, width, channels = frame.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, self.network_size, (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence:
                    # object detected
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    boxes.append((cx, cy, w, h))  # put all rectangle areas
                    class_ids.append(class_id)  # name of the object tha was detected
                    confidences.append(float(confidence))

        iou_thresh = 0.4
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, iou_thresh)
        engine_results = []
        for index in range(len(boxes)):
            if index in indices:
                cx, cy, w, h = boxes[index]
                top = int(cy - h / 2)
                right = int(cx + w / 2)
                bottom = int(cy + h / 2)
                left = int(cx - w / 2)
                engine_results.append(
                    (
                        left,
                        top,
                        right,
                        bottom,
                        int(class_ids[index]),
                        round(confidences[index], 2),
                    )
                )

        return engine_results
    
    
    
zed = sl.Camera()

    # Set configuration parameters
sl.RuntimeParameters.enable_fill_mode
   
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_fps = 30

# Create a InitParameters object and set configuration parameters
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
         print("eror-code:2")
         exit(1)


fourcc = cv2.VideoWriter_fourcc(*"avc1")       
resolution = (1920, 1080)

#out = cv2.VideoWriter("blended.mp4", fourcc,30.0,resolution) 
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter("video.avi", fourcc, 15.0, resolution)

names=["person","forklift"]
#out2 = cv2.VideoWriter("depth.mp4", fourcc,30.0,resolution)          
detector = OpenCVYoloDetector(network_size=(512, 512), confidence=0.4)
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8') 
i=0
save_folder="images/frame/"
depth_folder="images/depth/"
os.makedirs(save_folder,exist_ok=True)
os.makedirs(depth_folder,exist_ok=True)
while True:
    
    
           
    image_zed = sl.Mat(1920, 1080, sl.MAT_TYPE.U8_C4)
    image_depth_zed = sl.Mat(1920, 1080, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    
    # Retrieve data in a numpy array with get_data()
    
    if zed.grab() == sl.ERROR_CODE.SUCCESS :
      
        # Retrieve the left image in sl.Mat
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)

        # Use get_data() to get the numpy array
        image_ocv = image_zed.get_data()
        # Display the left image from the numpy array
        #cv2.imshow("Image_org", image_ocv)
        # Retrieve the normalized depth image
        zed.retrieve_image(image_depth_zed, sl.VIEW.DEPTH)

        # Use get_data() to get the numpy array
        image_depth_ocv = image_depth_zed.get_data()
        # Display the depth view from the numpy array
        #cv2.imshow("Image_depth-1", image_depth_ocv)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) # Retrieve colored point cloud
        
    
    re_frame=cv2.cvtColor(image_ocv,cv2.COLOR_RGBA2RGB) 
   
    
    h,w,_=re_frame.shape
    x=int(w/2)
    y=int(h/2)
    err, point_cloud_value = point_cloud.get_value(x, y)
    distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                         point_cloud_value[1] * point_cloud_value[1] +
                         point_cloud_value[2] * point_cloud_value[2])
    cv2.rectangle(
        re_frame,
        (x-2,y-15),
        (x+2, y+15),
        (255, 0,0),
        #(int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
        -1)
    cv2.rectangle(
        re_frame,
        (x-15,y-2),
        (x+15, y+2),
        (255, 0,0),
        #(int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
        -1)
    cv2.putText(re_frame, str(round(distance,2))+"M", (x-15,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                (255,255,51),
                #(int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
                2)    
    
    re_frame=cv2.resize(re_frame,(960,1080))
    re_output=cv2.cvtColor(image_depth_ocv,cv2.COLOR_RGBA2RGB)
    cv2.rectangle(
        re_output,
        (x-2,y-15),
        (x+2, y+15),
        (255, 0,0),
        #(int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
        -1)
    cv2.rectangle(
        re_output,
        (x-15,y-2),
        (x+15, y+2),
        (255, 0,0),
        #(int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
        -1)
    cv2.putText(re_output, str(round(distance,2))+"M", (x-15,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                (255,255,51),
                #(int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])),
                2)    
    re_output=cv2.resize(re_output,(960,1080))
    hor=np.hstack((re_frame,re_output))
    
    
    #cv2.imshow("Image_depth", hor)
    blended=cv2.addWeighted(src1=image_ocv, alpha=1, src2=image_depth_ocv, beta=0.8, gamma=0) 
    blended=cv2.cvtColor(blended,cv2.COLOR_RGBA2RGB)    
    re_frame=cv2.resize(blended,(1920,1080))
    
    #print(blended.shape)
    cv2.imshow("Image_depth-2", hor)
    #out.write(blended)

    #out2.write(hor)
    # cv2.imwrite("image+depth-performance.png", blended)
    # cv2.imwrite("image-performance.png", image_ocv)
    k=cv2.waitKey(1)
    
    if k==ord("q"):
        break
    elif k==ord("s"):
        print(image_depth_ocv.shape)
        print(image_ocv.shape)
        cv2.imwrite(save_folder+f"test-frame_{str(i).zfill(6)}_5m.png", image_ocv)
        cv2.imwrite(depth_folder+f"test-frame_{str(i).zfill(6)}_5m.png", image_depth_ocv)

    i+=1
#out.release()
#out2.release()  
cv2.destroyAllWindows()    
zed.close()  