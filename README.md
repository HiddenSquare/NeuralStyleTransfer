# Neural Style Transfer
This repo contains my attempt at Neural Style Transfer

Code is based on a lab from Andrew Ng's Deep Learning Specialization (course 4) on Coursera

https://www.coursera.org/specializations/deep-learning


## How to run

Run the main script by entering the following terminal command: 
python main.py --content [content_image] --styles [style_image(s)] --base-width [image_base_width] --iterations [no_of_iterations] --checkpoint-iter [iterations_between_checkpoints]

**Arguments:**

- content - path to the image containing the content you wish to stylize
- styles - path to the image or images containing the styles you wish to apply to your content image
- base-width - width of the image to feed into the model (all images reshaped to correspont to this size)
- iterations - the number of iterations you wish to run
- checkpoint-iter - the number of iterations between each saved checkpoint image


The code runs on tensorflow with gpu acceleration (might add an option to include cpu, otherwise modify the code a your leisure)


** TO DO:**
Add more style customization
