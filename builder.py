import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

class Builder:

    def __init__(self, model_path, content_image):
        self.model_path = model_path
        self.image_width = content_image.shape[1]
        self.image_height = int(content_image.shape[0])
        self.color_channels = 3
        self.OUTPUT_DIR = 'output/'

    def load_vgg_model(self):
        """
        Returns a model for the purpose of 'painting' the picture.
        Takes only the convolution layer weights and wrap using the TensorFlow
        Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
        the paper indicates that using AveragePooling yields better results.
        The last few fully connected layers (classifier part of the vgg) are not used .
        
		"""
        vgg = scipy.io.loadmat(self.model_path)
        self.vgg_layers = vgg['layers'][0]

    # Constructs the graph model.
    def build_model_graph(self):
        
        graph = {}
        self.load_vgg_model()

        graph['input'] = tf.Variable(np.zeros((1, self.image_height, self.image_width, self.color_channels)), dtype = 'float32')
        
        x = graph['input']
        target_layer = "pool5"

        for layer_idx, layer in enumerate(self.vgg_layers):

            # Get layer name and layer type
            try:
                layer_name_type = layer[0][0][2][0] # try if name is conv
                if layer_name_type != "conv":
                    layer_name_type = layer[0][0][3][0]#get pool name
            except:
                layer_name_type = layer[0][0][0][0]#get relu name

            try:
                layer_name = layer[0][0][3][0]
                if layer_name[:4] != "conv":
                    layer_name = layer[0][0][0][0]
            except:
                layer_name = layer[0][0][1][0]

            # generate graph based on layer name
            if layer_name_type == "conv":
                graph[layer_name] = self._conv2d_relu(x, layer_idx, layer_name)
                x = graph[layer_name]

            elif layer_name_type == "pool":
                graph[layer_name] = self._avgpool(x)
                x = graph[layer_name]

            if layer_name == target_layer:
                break

        return graph

    def _weights(self, layer_idx, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        layer_idx - integer representing index of layer in the vgg model from which to extract weights
        """
        # wb = self.vgg_layers[0][layer_idx][0][0][2] # based on old vgg-model shape
        wb = self.vgg_layers[layer_idx][0][0][0]
        W = wb[0][0]
        b = wb[0][1]
        # layer_name = self.vgg_layers[0][layer_idx][0][0][0][0] # based on old vgg-model shape
        layer_name = self.vgg_layers[layer_idx][0][0][3][0]

        # assert layer_name == expected_layer_name

        return W, b

    def _conv2d(self, prev_layer, layer_idx, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = self._weights(layer_idx, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(self, prev_layer, layer_idx, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        Calls _relu and _conv2d functions
        """
        return self._relu(self._conv2d(prev_layer, layer_idx, layer_name))

    @staticmethod
    def _relu(layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(layer)

    @staticmethod
    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
