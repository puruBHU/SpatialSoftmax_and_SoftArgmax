# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:10:51 2021

@author: Purnendu Mishra
"""

import numpy as np

import tensorflow as tf
from   tensorflow.keras.initializers import he_normal, RandomUniform, Constant
from   tensorflow.keras.layers import Layer, InputSpec, Softmax, Conv2D



def spatial_softmax(x):
    y = tf.exp(x - tf.reduce_max(x, axis=(1,2), keepdims=True))
    y = y / tf.reduce_sum(y, axis= (1,2), keepdims=True)
    return y
    

class SpatialSoftmax(Layer):
    def __init__(self, name):
        super(SpatialSoftmax, self).__init__(name = name)
        self.scale        = 1.0
        
        
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)] 
        
        initializer     = Constant(100.0)
        
        self.K          = self.add_weight(shape       = (input_shape[-1], ),
                                          initializer = initializer,
                                          trainable   = True)
        
        super(SpatialSoftmax, self).build(input_shape)
        
        
    def call(self, x):
        z = self.K * x
        
        output = spatial_softmax(z) 
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
    
class SoftArgMaxConv(Layer):
    def __init__(self, **kwargs):
        super(SoftArgMaxConv, self).__init__(**kwargs)
        
    def call(self,x):
        _, H, W, K = x.shape
        
        i = np.arange(H) / H
        j = np.arange(W) / W
        
        coords       = np.meshgrid(i,j, indexing = 'ij')
        image_coords = np.array(coords, dtype = np.float32).T
        kernel       = np.tile(image_coords[:,:,None,:],[1,1,K,1])
        
        kernel       = tf.constant(kernel, dtype = tf.float32) 
        
        y = tf.nn.depthwise_conv2d(x, kernel, strides= [1,1,1,1], padding='VALID')
        return y
        
    def comput_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, input_shape[3] * 2)