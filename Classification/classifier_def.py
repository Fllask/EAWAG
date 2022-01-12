# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:56:01 2021

@author: Gabriel Vallat
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import os
import sklearn.model_selection
import tensorboard
import pickle
import models
def dataset_creator(image_size = 200):
    base = "R:\\3.Masters_projects\\2021_Dominic_Rebindaine\\ROI"
    #extract the name from the file name:
    name_l =[]
    path_l = []
    for fname in os.listdir(base):
        pathto = os.path.join(base,fname)
        
        if os.path.isdir(pathto):
            path_l.append(pathto)
            split = fname.split(sep='_')
            if len(split)>2:
                name = split[1]
                name_l.append(name)
    label_name = np.unique(name_l)
    file_names = tf.data.Dataset.list_files(
        'R:\\3.Masters_projects\\2021_Dominic_Rebindaine\\ROI\\*\\*\\*.png'
        )
    train, val = sklearn.model_selection.train_test_split(list(file_names),test_size= 0.2)
    dt = tf.data.Dataset.from_tensor_slices(train)
    dv = tf.data.Dataset.from_tensor_slices(val)
    dt = input_pipeline(dt,image_size = image_size, label_name = label_name, 
                        augmentation=True)
    dv = input_pipeline(dv,image_size = image_size, label_name = label_name, 
                        augmentation=False)
    return dt,dv
#open the images and find their label
def data_opener(name,size,label_name):
    img = tf.io.decode_png(tf.io.read_file(name))
    X = tf.image.resize(img,(size,size))
    Xn = tf.image.per_image_standardization(X)
    part = tf.strings.split(name,os.sep)
    fname = part[4]
    fpart = tf.strings.split(fname,'_')
    if len(fpart)<2:
        y = -1
    else:
        el_label = fpart[1]
        y = -1
        for idx, label in enumerate(label_name):
            if label == el_label:
                y = idx
    data = (Xn,y)
    return data


        
        
def data_aug(img,label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    rnd = np.random.randint(4)
    img = tf.image.rot90(img,k=rnd)
    img = tf.image.random_hue(img,0.1)
    #stddev = tf.random.uniform(shape=[1],maxval=0.05)
    #img+= tf.random.normal((size,size,3),stddev=stddev)
    return (img,label)


def input_pipeline(ds, image_size = 200, label_name = np.arange(10),
                   augmentation = False):
    ds = ds.shuffle(10000)
    ds = ds.map(lambda x: data_opener(x,image_size, label_name))
    #filter out the unlabeled elements
    ds = ds.filter(lambda X,y: y>=0)
    if augmentation:
        ds = ds.map(data_aug)
    ds = ds.batch(32)
    ds = ds.prefetch(10)
    return ds
def build_and_compile():
    model = models.dropout_low_conv4()
    model.compile(optimizer="Adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits = False))
    return model