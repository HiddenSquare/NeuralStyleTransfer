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
        self.vgg_layers = vgg['layers']

    # Constructs the graph model.
    def build_model_graph(self):
        
        graph = {}
        self.load_vgg_model()

        # Declare input layer as trainable (tf.Variable)
        graph['input'] = tf.Variable(np.zeros((1, self.image_height, self.image_width, self.color_channels)), dtype = 'float32')

        # Declare remaining layers in the graph using imported weights from vgg19 and storing them as tf.constant
        graph['conv1_1']  = self._conv2d_relu(graph['input'], 0, 'conv1_1')
        graph['conv1_2']  = self._conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = self._avgpool(graph['conv1_2'])
        graph['conv2_1']  = self._conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2']  = self._conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = self._avgpool(graph['conv2_2'])
        graph['conv3_1']  = self._conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2']  = self._conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3']  = self._conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4']  = self._conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = self._avgpool(graph['conv3_4'])
        graph['conv4_1']  = self._conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2']  = self._conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3']  = self._conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4']  = self._conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = self._avgpool(graph['conv4_4'])
        graph['conv5_1']  = self._conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2']  = self._conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3']  = self._conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4']  = self._conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = self._avgpool(graph['conv5_4'])

        return graph

    def _weights(self, layer_idx, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        layer_idx - integer representing index of layer in the vgg model from which to extract weights
        """
        # wb = self.vgg_layers[0][layer_idx][0][0][2] # based on old vgg-model shape
        wb = self.vgg_layers[0][layer_idx][0][0][0]
        W = wb[0][0]
        b = wb[0][1]
        # layer_name = self.vgg_layers[0][layer_idx][0][0][0][0] # based on old vgg-model shape
        layer_name = self.vgg_layers[0][layer_idx][0][0][3][0]
        assert layer_name == expected_layer_name

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
