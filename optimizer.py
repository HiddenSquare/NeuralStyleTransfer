import os
import sys
import time
from argparse import ArgumentParser
import numpy as np
import scipy.io
import scipy.misc

from PIL import Image
import tensorflow as tf

# Local scripts
from builder import Builder
from utils import Utils

class Optimizer:
    '''Sets up and performs the optimization of the neural style transfer model'''

    def __init__(self, 
                iterations, 
                checkpoint_iter,
                style_image_influence,
                eval_content_layers,
                eval_style_layers,
                content_weight, 
                style_weight, 
                noise_ratio, 
                learning_rate, 
                content_image, 
                style_images, 
                model_path, 
                save_path,
                content_layer_influence=None,
                style_layer_influence=None
                ):

        self.iterations = iterations
        self.checkpoint_iter = checkpoint_iter
        self.style_image_influence = style_image_influence
        self.eval_content_layers = eval_content_layers
        self.eval_style_layers = eval_style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.noise_ratio = noise_ratio
        self.learning_rate = learning_rate

        self.total_variation_weight = 1e7

        self.content_image = content_image
        self.style_images = style_images
        
        self.save_path = save_path
        self.device = "/gpu:0"
        
        self.builder = Builder(model_path, content_image)
        self.utils = Utils()

        self.sess = tf.Session()

        # Check if influence factor exists for each content and style layer. If that is not the case, reset influence to default list of ones.
        if content_layer_influence is None or (content_layer_influence is not None and len(eval_content_layers) != len(content_layer_influence)):
            self.content_layer_influence = [1 for x in eval_content_layers]
        else:
            self.content_layer_influence = content_layer_influence
            
        if style_layer_influence is None or (style_layer_influence is not None and len(eval_style_layers) != len(style_layer_influence)):
            self.style_layer_influence = [1 for x in eval_style_layers]
        else:
            self.style_layer_influence = style_layer_influence

    def execute(self):
        '''Execute training of the style transfer model'''

        # self.set_layer_weight_distribution()
 
        tf.reset_default_graph()
        
        # Create tensorflow model
        model = self.builder.build_model_graph()
        
        self.content_image = self.utils.reshape_and_normalize_image(self.content_image)
        print("content_image shape: {}".format(self.content_image.shape))
        assert len(self.content_image.shape) == 4, "Content image reshaped incorrectly"

        noise_image = self.utils.generate_noise_image(self.content_image, self.noise_ratio)
        print(noise_image.shape)
        for i in range(len(self.style_images)):
            self.style_images[i] = self.utils.reshape_and_normalize_image(self.style_images[i])

        with tf.Session() as sess:
            content_cost = 0
            style_cost = 0

            # Assign the content image to be the input of the VGG model and run the image through the network
            sess.run(model['input'].assign(self.content_image))

            # Select the output tensor of layer conv4_2, corresponding to activations of selected layer
            # Choice of activation layer impacts detail of content in generated image
          
            for i, content_layer in enumerate(self.eval_content_layers):
                # out = model[output_layer]
                eval_layer = model[content_layer]

                # Get the activation from the selected layer when running the content image through the network
                a_C = sess.run(eval_layer)

                # Save the hidden layer activation from same layer. a_G references model[content_layer[i]] 
                # and remains an unevalutated tensor.
                a_G = eval_layer

                # Compute the content cost
                content_cost += self.compute_content_cost(a_C, a_G) * self.content_layer_influence[i]

            # Loop through all style images
            for i, style_image in enumerate(self.style_images):
                # Load the model with the style image
                sess.run(model['input'].assign(style_image))

                # Compute the style cost (session is run within the function to evaluate at different layers)
                style_cost += self.compute_style_cost(sess, model) * self.style_image_influence[i]
                
            
            # Declare the total cost variable
            total_cost = self.compute_total_cost(content_cost, style_cost)
            total_cost += self.total_variation_weight * self.total_variation_loss(model["input"])

            # Run the model
            print("Running model..\n")
            start_time = time.clock()
            self.model_nn(sess, model, noise_image, total_cost, content_cost, style_cost)
            
            time_passed = time.clock() - start_time
            print('Image generated in {} min and {} s'.format(int(np.floor_divide(time_passed, 60)) ,int(time_passed%60)))
            # print('Image generated in', time.clock() - start_time, 's')
            
            # image_list = './output/video_input/'
            # out_file = './output/video/clip.gif'
            # generate_video_seq(image_list, out_file, 2)
    
    def model_nn(self, sess, model, input_image, total_cost, content_cost, style_cost, store_in_list = False):
        '''Runs the style transfer model
        
        Arguments:
        sess - tensorflow session
        model - pretrained model (based on vgg19)
        input_image - pregenerated image acting as a base for styling
        total_cost - tensorflow graph defining the total cost calculation
        content_cost - tensorflow graph defining the content cost calculation
        style_cost - tensorflow graph defining the style cost calculation

        Returns:
        generated_image
        '''

        # define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = optimizer.minimize(total_cost)
        
        # Initialize global variables (run the session on the initializer)
        sess.run(tf.global_variables_initializer())
        
        # Run the noisy input image (initial generated image) through the model.
        sess.run(model["input"].assign(input_image))
        
        image_list = []
        with tf.device(self.device):
            for i in range(self.iterations):        
            # Run the session on the train_step (optimizer.minimize(J)) to minimize the total cost
            
                sess.run(train_step)

            # Compute the generated image by running the session on the current model['input']
                generated_image = sess.run(model["input"])

            # Print progress as specified by checkpoint_iter (default 200)
                if self.checkpoint_iter > 0 and i % self.checkpoint_iter == 0:
                    Jt, Jc, Js = sess.run([total_cost, content_cost, style_cost])
                    print("Iteration " + str(i) + " :")
                    print("total cost = " + str(Jt))
                    print("content cost = " + str(Jc))
                    print("style cost = " + str(Js))
                    
                    # save current generated image in the "/output" directory
                    if store_in_list:
                        image_list.append(generated_image)
                        
                    self.utils.save_image(self.save_path + str(i) + ".png", generated_image)                    

                            
        # save last generated image
        self.utils.save_image(self.save_path + 'generated_image.jpg', generated_image)
        return image_list if store_in_list else generated_image

    def compute_content_cost(self, a_C, a_G):
        """Computes the content cost
        
        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
        
        Returns: 
        content_cost -- scalar, content cost function.
        """
    
        # Retrieve dimensions from a_G
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        # Reshape a_C and a_G
        a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
        a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

        # compute the cost with tensorflow
        content_cost = 1/(4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
        
        return content_cost

    def compute_layer_style_cost(self, a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
        
        Returns: 
        style_layer_cost -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        
        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        # Reshape the images to have them of shape (n_C, n_H*n_W)
        a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))

        # Compute gram_matrices for both images S and G
        GS = self.gram_matrix(a_S)
 
        GG = self.gram_matrix(a_G)

        # Compute the loss (using np.square instead of ** resulted in large negative cost for some reason)
        style_layer_cost = 1/(4 * n_C**2 * (n_H * n_W)**2) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
        
        return style_layer_cost

    def compute_style_cost(self, sess, model):
        """
        Computes the overall style cost from several chosen layers
        
        Arguments:
        model -- tensorflow model
        style_layers -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them
        
        Returns: 
        style_cost -- tensor representing a scalar value, weighted by each supplied style layer
        """
        
        # initialize the overall style cost
        style_cost = 0

        for i, layer_name in enumerate(self.eval_style_layers):

            # Select the output tensor of the currently selected layer
            out = model[layer_name]

            # Set a_S to be the hidden layer activation from the layer we have selected
            a_S = sess.run(out)

            # Set a_G to be the hidden layer activation from same layer.
            a_G = out
            
            # Compute style_cost for the current layer
            style_layer_cost = self.compute_layer_style_cost(a_S, a_G)

            # Add coeff * J_style_layer of this layer to overall style cost
            style_cost += self.style_layer_influence[i] * style_layer_cost

        return style_cost

    @staticmethod
    def gram_matrix(A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)
        
        Returns:
        gram -- Gram matrix of A, of shape (n_C, n_C)
        """
        
        gram = tf.matmul(A, tf.transpose(A))
        
        return gram

    def compute_total_cost(self, content_cost, style_cost):
        """
        Computes the total cost function
        
        Arguments:
        content_cost -- content cost coded above
        style_cost -- style cost coded above
        content_weight -- hyperparameter weighting the importance of the content cost
        style_weight -- hyperparameter weighting the importance of the style cost
        
        Returns:
        total_cost -- total cost as defined by the formula above.
        """
        
        total_cost = self.content_weight * content_cost + self.style_weight * style_cost
        

        return total_cost

    @staticmethod
    def total_variation_loss(image):
        """Calculate total variation loss using difference in x and y directions, respectively
        """
        # Define 
        x_delta = image[:,:,1:,:] - image[:,:,:-1,:]
        y_delta = image[:,1:,:,:] - image[:,:-1,:,:]
        return tf.reduce_mean(x_delta**2) + tf.reduce_mean(y_delta**2)


    def set_style_weight_distribution(self, style_weights=(1,1,1,1,1)):
        '''
        Define influence of the different convolutional layer on generated style
        '''
        self.style_layers = [
        ('conv1_1', style_weights[0]),
        ('conv2_1', style_weights[1]),
        ('conv3_1', style_weights[2]),
        ('conv4_1', style_weights[3]),
        ('conv5_1', style_weights[4])]