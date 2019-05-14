import os
import sys
from PIL import Image
import cv2
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) # Mean pixel values used for normalization

class Utils:

    def __init__(self):
        self.means = MEANS                 

    def reshape_and_normalize_image(self, image):
        """
        Reshape and normalize the input image (content or style)
        ---
        - Reshape image to match expected input of VGG19
        
        - Subtract the mean to match the expected input of VGG19
        """

        image = np.reshape(image, ((1,) + image.shape))
    
        image = image - self.means
            
        return image
    
    def save_image(self, save_path, image):
        
        # Un-normalize the image so that it looks good
        image = image + self.means
        
        # Clip and Save the image
        image = np.clip(image[0], 0, 255).astype('uint8')
        scipy.misc.imsave(save_path, image)



    @staticmethod
    def resize_image(image, base_width, mod_aspect_ratio, target_shape=None):

        # Calculate scale factor
        scale_factor = base_width/image.shape[1]
        
        if target_shape is None:
            new_width = int(scale_factor * image.shape[1] * mod_aspect_ratio)
            new_height = int(scale_factor * image.shape[0] * mod_aspect_ratio)
        else:
            new_width = target_shape[1]
            new_height = target_shape[0]
            
        print(new_width, new_height)
        # resize image
        image_resized = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_NEAREST)
        
        return image_resized

    @staticmethod
    def generate_noise_image(content_image, noise_ratio = 0.3):
        """
        Generates a noisy image by adding random noise to the content_image (after normalisation)
        """
        
        # Generate a random noise_image, shape=(1, height, width, color_channels)
        noise_image = np.random.uniform(-20, 20, (1, content_image.shape[1], content_image.shape[2], 3)).astype('float32')
        print('content_image: {}, {}'.format(content_image.shape[0], content_image.shape[1]))
        # noise_image = np.random.uniform(-10, 10, (1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)).astype('float32')
        # Set the input_image to be a weighted average of the content_image and a noise_image
        input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
        # input_image = content_image
        
        return input_image
        
    @staticmethod
    def generate_video_seq(image_list, save_path, fps):
        # generate gif of stylized images
        # save_path points to the file
        clip = ImageSequenceClip(image_list, fps=fps)
        # clip.write_gif(save_path)
        clip.speedx(0.5).to_gif(save_path)