import cv2
import os
import math
import numpy as np

# given an image process it to improve the matching performance
def preprocessIR(img):
    #Histogram Equalisation to reduce shadows

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsvImg)
    v = clahe.apply(v)

    hsvImg = cv2.merge([h, s, v])
    imgO = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    #Use Canny Edge Detection

    edges = cv2.Canny(imgO,100,200)
    imgOHSV = cv2.cvtColor(imgO,cv2.COLOR_BGR2HSV)
    imgOV = imgOHSV[:,:,2]
    # cv2.imshow("Canny", edges)
    imgOH = imgOV-edges
    imgOHSV[:,:,2] = imgOV
    imgO = cv2.cvtColor(imgOHSV,cv2.COLOR_HSV2BGR)

    # Return the processed image

    return imgO
