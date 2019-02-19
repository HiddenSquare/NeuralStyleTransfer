# Neural Style Transfer
This repo contains my attempt at Neural Style Transfer

Code is based on a lab from Andrew Ng's Deep Learning Specialization (course 4) on Coursera

https://www.coursera.org/specializations/deep-learning


## How to run

Run the main script by entering the following terminal command: 
python main.py --content [content_image] --styles [style_image(s)] --base-width [image_base_width] --iterations [no_of_iterations] --checkpoint-iter [iterations_between_checkpoints]

**Base arguments:**

- content - path to the image containing the content you wish to stylize
- styles - path to the image or images containing the styles you wish to apply to your content image
- base-width - width of the image to feed into the model (all images reshaped to correspont to this size)
- iterations - the number of iterations you wish to run
- checkpoint-iter - the number of iterations between each saved checkpoint image

**Additional arguments:**
- content-weight - Level of content image influence on generated image
- style-weight - Level of style image influence on generated image
- style-image-influence - [If more than one style image is added as input] Fraction representing influence on generated style for each image.
- noise-ratio - Amount of random noise perturbation when initialising the generated image

The code runs on tensorflow with gpu acceleration (might add an option to include cpu, otherwise modify the code a your leisure)

Requires the vgg19 pretrained model, which can be found here:
http://www.vlfeat.org/matconvnet/models/beta16/


** TO DO:**
Add some example images
Add more style customization
