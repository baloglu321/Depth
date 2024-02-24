import cv2
import torch
import time
import numpy as np
import nvidia_smi
import matplotlib.pyplot as plt


model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
print("model updated successfull")    
    
def load_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(img).to(device)
    
    
    
    with torch.no_grad():
        prediction = midas(input_batch)
    
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=True,
        ).squeeze()
    
    output = prediction.cpu().numpy()
    output=cv2.normalize(output,None,0,1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
    
    return output


nvidia_smi.nvmlInit()

deviceCount = nvidia_smi.nvmlDeviceGetCount()

cap=cv2.VideoCapture(0)
fps=0
while True:
   
    #frame = stream.read()
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,640))
    
    if ret:
        img=frame.copy()
        start_time =time.time()

        output=load_img(img)
        end_time=time.time()
        fps=1/(end_time-start_time)
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu="Memory : ({:.2f}% free): {}(total) ".format( 100*info.free/info.total, info.total )
        gpu2="{} (free), {} (used)".format(info.free, info.used)
        
        
        
        output=(output*255).astype(np.uint8)
        output=cv2.applyColorMap(output, cv2.COLORMAP_MAGMA)
        hor=np.hstack((frame,output))
        
        cv2.putText(hor,gpu, (15,30), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,255,0), 2)
        cv2.putText(hor,gpu2, (15,60), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,255,0), 2)
        cv2.putText(hor,f"{fps:.2f} FPS " , (15,90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,255,0), 2)
        
        cv2.imshow("Horizontal",hor)
        
        #cv2.imshow("f",frame)
        #cv2.imshow("m",output)
  

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break   

nvidia_smi.nvmlShutdown()    
cap.release()
cv2.destroyAllWindows()