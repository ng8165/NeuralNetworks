# Written by Aarush Vailaya and modified by Nelson Gou

import cv2
from PIL import Image
import numpy as np
import scipy.ndimage as ndi

def process_image(dataset, finger):
    imgFileNum = 3698 + 5*(dataset-1) + (finger-1)
    img = cv2.imread("image_src/IMG_" + str(imgFileNum) + ".jpg")

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=50)       # increase contrast (mess with numbers until it looks good)
    img = cv2.bitwise_not(img)                              # invert the image colors
    
    # set all pixels with saturation less than the threshold to black
    saturation_threshold = 70
    img[img_hsv[:, :, 1] < saturation_threshold] = 0

    # find center of mass
    com = *(int(i) for i in ndi.center_of_mass(img.astype(int))),

    # crop, blur, and process image
    img = img[com[0]-600:com[0]+500, com[1]-400:com[1]+1100]    # crop image
    img = cv2.blur(img, (12, 12))                               # blur to make edges smoother
    img = cv2.resize(img, dsize=(105, 77))                      # scale down to reduce training inputs

    # save the image as BMP
    image = Image.fromarray(img.astype(np.uint8))
    image.save('image_processed/' + str(dataset) + "_" + str(finger) + '.bmp')

    return img

# process_image(1, 1)

for dataset in range(1, 7):
   for finger in range(1, 6):
       process_image(dataset, finger)