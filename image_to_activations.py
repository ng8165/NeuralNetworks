import cv2
import numpy as np

# process images for training

open('image_activations_train.txt', 'w').close()

for dataset in range(1, 6):
   for finger in range(1, 6):
        img = cv2.imread('image_processed/' + str(dataset) + "_" + str(finger) + '.bmp')

        # save the image as activations
        activations = (img/255).flatten()[None]
        
        with open("image_activations_train.txt", "a") as stream:
            np.savetxt(stream, activations, fmt="%.17f")

# process images for running

open('image_activations_run.txt', 'w').close()

for finger in range(1, 6):
    img = cv2.imread('image_processed/6_' + str(finger) + '.bmp')

    # save the image as activations
    activations = (img/255).flatten()[None]
    
    with open("image_activations_run.txt", "a") as stream:
        np.savetxt(stream, activations, fmt="%.17f")
