import os
import sys
from argparse import ArgumentParser
import time
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
import tensorflow as tf

# Local scripts
from optimizer import Optimizer
from utils import Utils
from read_image import ReadImage
# import moviepy

ITERATIONS = 1000
CHECKPOINT_ITER = 200
STYLE_IMAGE_INFLUENCE = [1]
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 5e2
EVAL_CONTENT_LAYERS = ["conv4_2", "conv5_2"]
EVAL_STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
CONTENT_LAYER_INFLUENCE = None
STYLE_LAYER_INFLUENCE = None
LEARNING_RATE = 1e0
NOISE_RATIO = 0.2

def build_parser():
    parser = ArgumentParser(add_help=True)
    
    parser.add_argument("-c", "--content",
        dest="content", help="content image",
        metavar="CONTENT", required=True)
    
    parser.add_argument("-s", "--styles",
        dest="styles", nargs="+",
        help="one or more style images",
        metavar="STYLE", required=True)
        
    parser.add_argument("-bw", "--base-width", type=int,
        dest="base_width", help="base width of image",
        metavar="BASEWIDTH", required=True)           
           
    parser.add_argument("-i", "--iterations", type=int,
        dest="iterations", help="iterations (default %(default)s)",
        metavar="ITERATIONS", default=ITERATIONS)
    
    parser.add_argument("-ci", "--checkpoint-iter", type=int,
        dest="checkpoint_iter", help="number of iterations between each progress printout and image save",
        metavar="CHECKPOINT_ITER", default=CHECKPOINT_ITER)
            
    parser.add_argument("-cw", "--content-weight", type=float,
        dest="content_weight", help="content weight value",
        metavar="CONTENT_WEIGHT", default=CONTENT_WEIGHT)
    
    parser.add_argument("-sw", "--style-weight", type=float,
        dest="style_weight", help="style weight value",
        metavar="STYLE_WEIGHT", default=STYLE_WEIGHT)
            
    parser.add_argument("-sii", "--style-image-influence", type=float,
        dest="style_image_influence", nargs="+",
        help="if using several style images. fraction representing image influence on style, one value  for each style image",
        metavar="STYLE_IMAGE_INFLUENCE", default=STYLE_IMAGE_INFLUENCE)
    
    parser.add_argument("-ecl", "--eval-content-layers", type=str,
        dest="eval_content_layers", nargs="+",
        help="Choose which of the VGG19 that will be used when evaluating content loss",
        default=EVAL_CONTENT_LAYERS)

    parser.add_argument("-esl", "--eval-style-layers", type=str,
        dest="eval_style_layers", nargs="+",
        help="Choose which of the VGG19 that will be used when evaluating style loss",
        default=EVAL_STYLE_LAYERS)

    parser.add_argument("-cli", "--content-layer-influence", type=float,
        dest="content_layer_influence", nargs="+",
        help="Influence of each selected content layer on content loss",
        default=CONTENT_LAYER_INFLUENCE)

    parser.add_argument("-sli", "--style-layer-influence", type=float,
        dest="style_layer_influence", nargs="+",
        help="Influence of each selected style layer on style loss",
        default=STYLE_LAYER_INFLUENCE)

    parser.add_argument("-lr", "--learning-rate", type=float,
        dest="learning_rate", help="Optimizer learning rate, governs how quickly style updates are propagated in the generated image ",
        metavar="LEARNING_RATE", default=LEARNING_RATE)

    parser.add_argument("-nr", "--noise-ratio", type=float,
        dest="noise_ratio", help="Amount of random noise perturbation when initialising generated image",
        metavar="NOISE_RATIO", default=NOISE_RATIO)
           
       
    return parser

MODEL_PATH = "./model/imagenet-vgg-verydeep-19.mat" #could add this as argument
SAVE_PATH = "./output/"
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)
MOD_ASPECT_RATIO = 1


def main():
    parser = build_parser()

    # Parse input arguments          
    options = parser.parse_args()

    assert len(options.style_image_influence) == len(options.styles), 'Number of weights needs to match number of style images'
    assert sum(options.style_image_influence) == 1, 'Weights do not add up to 1'   

    # Get images from local source
    content_image, style_images = ReadImage(options.base_width, MOD_ASPECT_RATIO).read_local(options.content, options.styles)

    # Instantiate the optimizer
    optimizer = Optimizer(iterations=options.iterations, 
                        checkpoint_iter=options.checkpoint_iter,
                        style_image_influence=options.style_image_influence, 
                        eval_content_layers=options.eval_content_layers,
                        eval_style_layers=options.eval_style_layers,
                        content_weight=options.content_weight, 
                        style_weight=options.style_weight,
                        noise_ratio=options.noise_ratio,
                        learning_rate=options.learning_rate,
                        content_image=content_image,
                        style_images=style_images,
                        model_path=MODEL_PATH,
                        save_path=SAVE_PATH,
                        content_layer_influence=options.content_layer_influence,
                        style_layer_influence=options.style_layer_influence
                        )
  
    # Execute optimizer
    optimizer.execute()


if __name__ == "__main__":
    main()
