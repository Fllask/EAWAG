# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:19:28 2021

@author: Gabriel Vallat
"""

import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import Model

def naive_conv(size = 200, n_out = 10):
    inputs = layers.Input((size,size,3))
    x = layers.Conv2D(16,3,activation='relu')(inputs)
    x = layers.Conv2D(16,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation = 'relu')(x)
    x = layers.Dense(n_out,activation = 'softmax')(x)
    
    model = Model(inputs = inputs, outputs = x)
    return model



def dropout_conv(size = 200, n_out = 10):
    inputs = layers.Input((size,size,3))
    x = layers.Conv2D(16,3,activation='relu')(inputs)
    x = layers.Conv2D(16,3,activation='relu')(x)
    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation = 'relu')(x)
    x = layers.Dense(n_out,activation = 'softmax')(x)
    
    model = Model(inputs = inputs, outputs = x)
    return model

def dropout_low_conv(size = 200, n_out = 10):
    inputs = layers.Input((size,size,3))
    x = layers.Conv2D(16,3,activation='relu')(inputs)
    x = layers.Conv2D(16,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128,3,activation='relu')(x)
    x = layers.Conv2D(128,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(256,3,activation='relu')(x)
    x = layers.Conv2D(256,3,activation='relu')(x)
    x = layers.SpatialDropout2D(0.6)(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(128, activation = 'relu')(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(n_out,activation = 'softmax')(x)
    
    model = Model(inputs = inputs, outputs = x)
    return model
def dropout_low_conv2(size = 200, n_out = 10):
    inputs = layers.Input((size,size,3))
    x = layers.Conv2D(16,3,activation='relu')(inputs)
    x = layers.Conv2D(16,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128,3,activation='relu')(x)
    x = layers.Conv2D(128,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(256,3,activation='relu')(x)
    x = layers.Conv2D(256,3,activation='relu')(x)
    x = layers.SpatialDropout2D(0.6)(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(256, activation = 'relu')(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(256, activation = 'relu')(x)

    x = layers.Dense(n_out,activation = 'softmax')(x)
    
    model = Model(inputs = inputs, outputs = x)
    return model
def dropout_low_conv3(size = 200, n_out = 10):
    inputs = layers.Input((size,size,3))
    x = layers.Conv2D(16,3,activation='relu')(inputs)
    x = layers.Conv2D(16,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128,3,activation='relu')(x)
    x = layers.Conv2D(128,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(256,3,activation='relu')(x)
    x = layers.Conv2D(512,3,activation='relu')(x)
    x = layers.SpatialDropout2D(0.6)(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(256, activation = 'relu')(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(256, activation = 'relu')(x)

    x = layers.Dense(n_out,activation = 'softmax')(x)
    
    model = Model(inputs = inputs, outputs = x)
    return model
def dropout_low_conv4(size = 200, n_out = 10):
    inputs = layers.Input((size,size,3))
    x = layers.Conv2D(16,3,activation='relu')(inputs)
    x = layers.Conv2D(16,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128,3,activation='relu')(x)
    x = layers.Conv2D(128,3,activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(256,3,activation='relu')(x)
    x = layers.Conv2D(512,3,activation='relu')(x)
    x = layers.SpatialDropout2D(0.6)(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(512, activation = 'relu')(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(512, activation = 'relu')(x)

    x = layers.Dense(n_out,activation = 'softmax')(x)
    
    model = Model(inputs = inputs, outputs = x)
    return model

################# transfer learning #############################3
def top_dropout(x,n_out,dropout=0.8):
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(100,activation='relu')(x)
    x = layers.Dense(100,activation='relu')(x)
    x = layers.Dense(n_out,activation = 'softmax')(x)
    return x

def transer_NASnet(size=200, n_out = 10, init = None ):
    inputs = layers.Input((size,size,3))
    NASnet = keras.applications.NASNetMobile(include_top=False,
                                                input_tensor=inputs,
                                                weights=init)
    #test 1: dropout_rate = 0.8
    #test 2: dropout_rate = 0.95
    out = top_dropout(NASnet.output,n_out,dropout=0.95)
    model = Model(inputs = inputs, outputs = out)
    return model