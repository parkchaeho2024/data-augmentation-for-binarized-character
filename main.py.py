import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image
import cv2
import random

from torchvision import datasets, transforms

def blur_img(img):
   
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
    threshold_value = np.random.randint(100, 157)

    # Apply binary thresholding to the image
    _, binary_thin_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_thin_image

def binarize_image(np_image, threshold=128):    
    binary_image = (np_image > threshold).astype(np.uint8)
    return binary_image

def augment_markov(input_image, ratio_a, ratio_b, size):
    distance = distance_transform_edt(1 - input_image)
    gradient_direction = np.arctan2(*np.gradient(-distance))
    boundary_points = np.argwhere(distance == 1)

    augmented_image = np.stack((input_image,)*3, axis=-1) * 255
    prev_state = 0  # initial state
    fixed_move_distance = 0  # 변initial size of line value

    for point in boundary_points:
        y, x = point

        if prev_state == 0:
            current_state = 1 if np.random.rand() < ratio_a else 0
            if current_state == 1:
                # deciding size of line and direction when start the modification
                fixed_move_distance = np.random.randint(-size, size + 1)
        else:
            current_state = 0 if np.random.rand() < ratio_b else 1

        if current_state == 1:
            direction = np.sign(fixed_move_distance)
            theta = gradient_direction[y, x]

            end_y = int(y + direction * abs(fixed_move_distance) * np.sin(theta))
            end_x = int(x + direction * abs(fixed_move_distance) * np.cos(theta))

            if 0 <= end_x < input_image.shape[1] and 0 <= end_y < input_image.shape[0]:
                color = (0, 0, 0) if direction > 0 else (255, 255, 255)
                cv2.line(augmented_image, (x, y), (end_x, end_y), color, 1)
        
        prev_state = current_state  # update the state
    augmented_image = blur_img(augmented_image)

    return augmented_image

class RandomBoundary_markov(object):
    def __init__(self, ratio_a, ratio_b, size, threshold=128):
        self.ratio_a = ratio_a
        self.ratio_b = ratio_b
        self.size = size
        self.threshold = threshold

    def __call__(self, img):
        # PIL => grayscale => numpy
        gray_img = np.array(img.convert('L'))        
        # binarization
        binary_image = (gray_img > self.threshold).astype(np.uint8)        
        # apply Markovian Stroke Thickness Variation
        augmented_img = augment_markov(binary_image, self.ratio_a, self.ratio_b, self.size)      
    
    
        return Image.fromarray(augmented_img.astype(np.uint8))


# usage
tf = [        
        RandomBoundary_markov(ratio_a=0.1, ratio_b=0.1, size=2),  # proposed MSTV method (1-ratio = p_m, size = h)     
        transforms.RandomRotation(5),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    ]