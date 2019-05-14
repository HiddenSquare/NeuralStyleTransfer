import os
import sys
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image

from utils import Utils


class ReadImage:
    def __init__(self, base_width, mod_aspect_ratio):
        self.base_width = base_width
        self.mod_aspect_ratio = mod_aspect_ratio
        self.utils = Utils()

    @staticmethod
    def assert_image_shape(content_image, style_images):
        assert style_images[0].shape == content_image.shape, "Dimensions of style image(s) and content images do not match! shapes: {}, {}".format(style_images[0].shape, content_image.shape)


    def read_local(self, path_content, path_style):
        # Load content image
        content_image_base = scipy.misc.imread(path_content)

        # Resize image to fit model
        content_image =  self.utils.resize_image(content_image_base, self.base_width, self.mod_aspect_ratio)

        style_images = []
        for style_fname in path_style:
            style_image_base = scipy.misc.imread(style_fname)
            style_images.append(self.utils.resize_image(style_image_base, self.base_width, self.mod_aspect_ratio, target_shape = content_image.shape))

        self.assert_image_shape(content_image, style_images)

        return content_image, style_images

    def read_from_url(self, url_content, url_style_list):

        import urllib.request
        import io
        import ssl

        # This restores the same behavior as before.
        context = ssl._create_unverified_context()

        print(url_content)
        with urllib.request.urlopen(url_content, context=context) as url_c:
            content_file = io.BytesIO(url_c.read())
            print("content file type:", type(content_file))

        # decoded = cv2.imdecode(np.frombuffer(content_file, np.uint8), -1))
        content_image_base = np.array(Image.open(content_file))
        # content_image_base = scipy.misc.imread(decoded)
        print("content image type:", content_image_base.shape)
        # Resize image to fit model
        content_image =  self.utils.resize_image(content_image_base, self.base_width, self.mod_aspect_ratio)


        style_images = []
        for url_style in url_style_list:
            with urllib.request.urlopen(url_style, context=context) as url_s:
                style_file = io.BytesIO(url_s.read())

            # decoded = cv2.imdecode(np.frombuffer(content_file, np.uint8), -1))
            style_image_base = np.array(Image.open(style_file))
            # style_image_base = scipy.misc.imread(decoded)
            style_images.append(self.utils.resize_image(style_image_base, self.base_width, self.mod_aspect_ratio, target_shape = content_image.shape))

        self.assert_image_shape(content_image, style_images)

        return content_image, style_images

    # Verify that content and style images are of the same size    
