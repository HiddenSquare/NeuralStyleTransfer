import os, sys
from argparse import ArgumentParser
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import tensorflow as tf

from optimizer import Optimizer
from utils import Utils
# import moviepy
import time


ITERATIONS = 1000
CHECKPOINT_ITER = 200
IMAGE_INFLUENCE = [1]
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 5e2
NOISE_RATIO = 0.2

def build_parser():
    parser = ArgumentParser(add_help=True)
    
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)
        
    parser.add_argument('--base-width', type=int,
            dest='base_width', help='base width of image',
            metavar='BASEWIDTH', required=True)           
           
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    
    parser.add_argument('--checkpoint-iter', type=int,
            dest='checkpoint_iter', help='number of iterations between each progress printout and image save',
            metavar='CHECKPOINT_ITER', default=CHECKPOINT_ITER)
            
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight value',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight value',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
            
    parser.add_argument('--style-image-influence', type=float,
            dest='style_image_influence',
            nargs='+', help='if using several style images. fraction representing image influence on style, one value  for each style image',
            metavar='IMAGE_INFLUENCE', default=IMAGE_INFLUENCE)
            
    parser.add_argument('--noise-ratio', type=float,
            dest='noise_ratio', help='Amount of random noise perturbation when initialising generated image',
            metavar='NOISE_RATIO', default=NOISE_RATIO)
           
       
    return parser

MODEL_PATH = "./model/imagenet-vgg-verydeep-19.mat"
SAVE_PATH = "./output/"
MOD_ASPECT_RATIO = 1


def main():
    parser = build_parser()

    # Parse input arguments          
    options = parser.parse_args()

    assert len(options.style_image_influence) == len(options.styles), 'Number of weights needs to match number of style images'
    assert sum(options.style_image_influence) == 1, 'Weights do not add up to 1'   

    # Load content image
    content_image_base = scipy.misc.imread(options.content)

    # Resize image to fit model
    content_image =  Utils().resize_image(content_image_base, options.base_width, MOD_ASPECT_RATIO)

    style_images = []
    for style_fname in options.styles:
        style_image_base = scipy.misc.imread(style_fname)
        style_images.append(Utils().resize_image(style_image_base, options.base_width, MOD_ASPECT_RATIO, target_shape = content_image.shape))

    # Verify that content and style images are of the same size    
    assert style_images[0].shape == content_image.shape, "Dimensions of style image(s) and content images do not match! shapes: {}, {}".format(style_images[0].shape, content_image.shape)
       

    # Instanciate the optimizer
    optimizer = Optimizer(iterations=options.iterations, 
                        checkpoint_iter=options.checkpoint_iter, 
                        style_image_influence=options.style_image_influence, 
                        content_weight=options.content_weight, 
                        style_weight=options.style_weight,
                        noise_ratio=options.noise_ratio,
                        content_image=content_image,
                        style_images=style_images,
                        model_path=MODEL_PATH,
                        save_path=SAVE_PATH
                        )
    
    # Execute optimizer
    optimizer.execute()


if __name__ == "__main__":
    main()
