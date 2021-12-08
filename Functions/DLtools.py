# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:26:44 2021

@author: Gabriel Vallat
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


#define a custom padding layer to use reflect padding
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        h_pad,w_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
#define a custom convolution layer
class ReflectionConv2D(Layer):
    def __init__(self,filters, kernel,activation = 'relu',**kwargs):
        super().__init__(**kwargs)
        self.pad = ReflectionPadding2D(padding=(kernel[0]//2,kernel[1]//2))
        self.conv = layers.Conv2D(filters,kernel,padding='valid',activation=activation)
    def call(self,inputs):
        padded = self.pad(inputs)
        out = self.conv(padded)
        return out
def replace_conv_with_ReflectionConv(model,name=None,outputs_ordered=True):
    model_layers = [l for l in model.layers]
    outputs = []
    outputs_id = 0
    outputs_desordo = []
    visible = layers.Input(shape=model_layers[0].input.shape[1:])
    x = visible
    for layer in model_layers[1:]:
        if isinstance(layer,layers.Conv2D):
            x = ReflectionConv2D(filters = layer.filters,
                                 kernel = layer.kernel.shape[0:2],
                                 name=layer.name)(x)
        else:
            x = layer(x)
        if outputs_ordered:
            if layer.output is model.outputs[outputs_id]:
                outputs.append(x)
                outputs_id+=1
        else:
            for idx,model_output in enumerate(model.outputs):
                if layer.output is model_output:
                    outputs_desordo.append((x,idx))
    if not outputs_ordered:
        def takeidx(elem):
            return elem[1]
        
        outputs_desordo.sort(key=takeidx)
        outputs = [el[0] for el in outputs_desordo]
    return Model(inputs = visible, outputs = outputs,name=name)
