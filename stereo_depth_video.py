#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:09:03 2023

@author: mbaloglu

Try with Diffirent calibration file
"""
import sys
import numpy as np
import cv2
from cv2 import cuda

REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048
conf_list=["output/6x4-128.npz","output/6x4-300.npz","output/7x4-64.npz","output/7x4-128.npz","output/7x4-300.npz","output/8x6-64.npz","output/8x6-128.npz","output/8x6-300.npz","output/13x9-64.npz","output/13x9-128.npz","output/a3-7x4-128.npz"]

class StereoWrapper:
    """
    This class takes care of the CUDA input such that such that images
    can be provided as numpy array
    """
    def __init__(self,
                 num_disparities: int = 128,
                 block_size: int = 21,
                 bp_ndisp: int = 64,
                 min_disparity: int = 4,
                 uniqueness_ratio: int = 5
                 ) -> None:
        self.stereo_bm_cuda = cuda.createStereoBM(numDisparities=num_disparities,
                                                  blockSize=block_size)
        self.stereo_bp_cuda = cuda.createStereoBeliefPropagation(ndisp=bp_ndisp)
        self.stereo_bcp_cuda = cuda.createStereoConstantSpaceBP(min_disparity)
        self.stereo_sgm_cuda = cuda.createStereoBM(numDisparities=num_disparities,
                                                  blockSize=block_size )
    @staticmethod
    def __numpy_to_gpumat(np_image: np.ndarray) -> cv2.cuda_GpuMat:
        """
        This method converts the numpy image matrix to a matrix that
        can be used by opencv cuda.
        Args:
            np_image: the numpy image matrix
        Returns:
            The image as a cuda matrix
        """
        image_cuda = cv2.cuda_GpuMat()
        image_cuda.upload(cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY))
        return image_cuda
    def compute_disparity(self, left_img: np.ndarray,
                          right_img: np.ndarray,
                          algorithm_name: str = "stereo_sgm_cuda"
                          ) -> np.ndarray:
        """
        Computes the disparity map using the named algorithm.
        Args:
            left_img: the numpy image matrix for the left camera
            right_img: the numpy image matrix for the right camera
            algorithm_name: the algorithm to use for calculating the disparity map
        Returns:
            The disparity map
        """
        algorithm = getattr(self, algorithm_name)
        leftCuda = self.__numpy_to_gpumat(left_img)
        right_cuda = self.__numpy_to_gpumat(right_img)
        #print(type(leftCuda))
        if algorithm_name == "stereo_sgm_cuda":

            disparity_sgm_cuda_1 = algorithm.compute(leftCuda,right_cuda,cv2.cuda_Stream.Null())
            return disparity_sgm_cuda_1.download()
        else:
            disparity_cuda = algorithm.compute(leftCuda, right_cuda, cv2.cuda_Stream.Null())
            return disparity_cuda.download()

def apply_interpolation(depth_map):
    """
    Derinlik haritasına interpolasyon uygular.

    Args:
        depth_map: Giriş derinlik haritası.

    Returns:
        İnterpolasyon uygulanmış derinlik haritası.
    """
    # Boş veya eksik bölgeleri belirlemek için maske oluşturun
    mask = (depth_map == 0).astype(np.uint8)
    mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


    # Boş bölgeleri doldurmak için interpolasyon yapın (örneğin, bir inpainting yöntemi kullanabilirsiniz)
    interpolated_depth_map = cv2.inpaint(depth_map, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
 
    return interpolated_depth_map


def apply_bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Bilateral filtreleme uygular.

    Args:
        image: Giriş görüntü.
        diameter: Filtre boyutu.
        sigma_color: Renk benzerlik parametresi.
        sigma_space: Konum benzerlik parametresi.

    Returns:
        Filtrelenmiş görüntü.
    """
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return filtered_image


def cvt_gray(image, threshold_green=50, threshold_red=50):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    #bw_colored_depth = np.zeros_like(image)
    mask = np.logical_or(green > threshold_green, red > threshold_red)
    blue[mask]=255
    green[mask]=255
    red[mask]=255
    blue[np.logical_not(mask)]=0
    green[np.logical_not(mask)]=0
    red[np.logical_not(mask)]=0
    bw_colored_depth = np.stack([blue, green, red], axis=-1)

    return bw_colored_depth

def median_map(image):


    # Derinlik haritasını siyah-beyaz yapın
    gray_depth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azaltmak için medyan filtresi uygulayın
    filtered_depth = cv2.medianBlur(gray_depth, ksize=5)

    # Eksik veya boş bölgeleri doldurmak için interpolasyon yapın
    interpolated_depth = cv2.inpaint(filtered_depth, np.uint8(filtered_depth == 0), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Renkli derinlik haritası oluşturun
    colorized_depth = cv2.applyColorMap(interpolated_depth, cv2.COLORMAP_JET)

    return colorized_depth
def gauss_map(image):


    # Derinlik haritasını siyah-beyaz yapın
    gray_depth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azaltmak için Gauss filtresi uygulayın
    filtered_depth = cv2.GaussianBlur(gray_depth, (5, 5), 0)

    # Eksik veya boş bölgeleri doldurmak için interpolasyon yapın
    interpolated_depth = cv2.inpaint(filtered_depth, np.uint8(filtered_depth == 0), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Renkli derinlik haritası oluşturun
    colorized_depth = cv2.applyColorMap(interpolated_depth, cv2.COLORMAP_JET)

    return colorized_depth
def apply_histogram_equalization(depth_map):
    """
    Histogram eşitleme uygular.

    Args:
        depth_map: Giriş derinlik haritası.

    Returns:
        Histogram eşitlenmiş derinlik haritası.
    """
    # Derinlik değerlerini normalleştirin
    normalized_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    normalized_depth_map_8bit = (normalized_depth_map * 255).astype(np.uint8)
    
    # Histogram eşitleme uygulayın
    equalized_depth_map = cv2.equalizeHist(normalized_depth_map_8bit)
    
    return equalized_depth_map
    
for conf in conf_list:

    calibration = np.load(conf, allow_pickle=False)
    imageSize = tuple(calibration["imageSize"])
    leftMapX = calibration["leftMapX"]
    leftMapY = calibration["leftMapY"]
    leftROI = tuple(calibration["leftROI"])
    rightMapX = calibration["rightMapX"]
    rightMapY = calibration["rightMapY"]
    rightROI = tuple(calibration["rightROI"])
    
    
    
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    
    # TODO: Use more stable identifiers
    left = cv2.VideoCapture("output/left/video-left.mp4")
    right = cv2.VideoCapture("output/right/video-right.mp4")
    
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
    
    # TODO: Why these values in particular?
    # TODO: Try applying brightness/contrast/gamma adjustments to the images
        
    stereoMatcher = cv2.StereoBM_create()
    stereoMatcher.setMinDisparity(4)
    stereoMatcher.setNumDisparities(128)
    stereoMatcher.setBlockSize(21)
    stereoMatcher.setROI1(leftROI)
    stereoMatcher.setROI2(rightROI)
    stereoMatcher.setSpeckleRange(16)
    stereoMatcher.setSpeckleWindowSize(45)
    
    # Grab both frames first, then retrieve to minimize latency between cameras
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    conf_name=conf.split('.')[0].split('/')[1]
    print(conf_name)
    name = f"output/reworked/deneme/video-{conf_name}.mp4"
    out_1 = cv2.VideoWriter(name, fourcc, 30, (1280, 1080))
    
    
    

    
    i=0
    while(True):
        if not left.grab() or not right.grab():
            print("No more frames")
            break
    
        _, leftFrame = left.retrieve()
        leftFrame = cropHorizontal(leftFrame)
        leftHeight, leftWidth = leftFrame.shape[:2]
        _, rightFrame = right.retrieve()
        rightFrame = cropHorizontal(rightFrame)
        rightHeight, rightWidth = rightFrame.shape[:2]
    
        # if (leftWidth, leftHeight) != imageSize:
        #     print("Left camera has different size than the calibration data")
        #     break
    
        # if (rightWidth, rightHeight) != imageSize:
        #     print("Right camera has different size than the calibration data")
        #     break
    
        fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
        fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)
    
        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        depth = stereoMatcher.compute(grayLeft, grayRight)
        # wrapper = StereoWrapper()
        # depth = wrapper.compute_disparity(fixedLeft, fixedRight)
        hist_depth=apply_histogram_equalization(depth)
        colored_depth_hist = cv2.applyColorMap(hist_depth, cv2.COLORMAP_JET)
        # saved_left=cv2.resize(grayLeft,(420,720))
        # saved_right=cv2.resize(grayRight,(420,720))
        # saved_depth=cv2.resize(depth / DEPTH_VISUALIZATION_SCALE , (440,720))
        # hor=np.hstack((saved_left,saved_right,saved_depth))
        normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())
    
    # 3 kanallı gri tonlamalı görüntü oluşturun
        gray_depth = (normalized_depth * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET)
        
        # Görüntüleri yeniden boyutlandırın (isteğe bağlı)
        saved_left = cv2.resize(fixedLeft, (640, 540))
        saved_right = cv2.resize(fixedRight, (640, 540))
        color_3ch=cv2.resize(colored_depth,(640,540))
        depth_3ch= cvt_gray(colored_depth)
        #depth_3ch=apply_histogram_equalization(color_3ch)
        #depth_3ch=cv2.resize(colored_depth_hist,(640,540))
        saved_colored_depth = cv2.resize(depth_3ch, (640, 540))
        
        # Görüntüleri yatayda birleştirin
        hor = np.hstack((saved_left, saved_right))
        hor2 = np.hstack((saved_colored_depth,color_3ch))
        ver=np.vstack((hor,hor2))
        out_1.write(ver)
        #cv2.imwrite(f"outpu/images/depth_image_{i}.png", saved_depth)
        
        #cv2.imshow('left', fixedLeft)
        #cv2.imshow('right', fixedRight)
        #cv2.imshow('depth', depth_3ch)
        cv2.imshow('ver', ver)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i+=1
    left.release()
    right.release()
    cv2.destroyAllWindows()