import cv2
from os import listdir, makedirs, walk
from os.path import isfile, join, exists
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np

def crop_image(img):
    """
    croppa l'immagine in modo che sia di forma quadrata: l*l
    prende come dimensione la dimensione piu piccola dell immagine
    """
    # cropImg = img[rowStart:rowEnd, colsStart:colsEnd]

    crop_img = []
    width = img.shape[1]
    height = img.shape[0]

    if width > height:
        margin = int((width - height) / 2)
        crop_img = img[0:height, margin:(width - margin)]
    else:
        margin = int((height - width) / 2)
        crop_img = img[margin:(height - margin):, 0:width]

    # cv2.imshow("cropped", crop_img)

    return crop_img