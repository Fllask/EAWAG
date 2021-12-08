# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:24:24 2021

@author: valla
"""

#loading of the packages
import numpy as np
from scipy import ndimage
from scipy import signal
import scipy.misc
import imageio
import skimage.io
from skimage import morphology
from skimage import filters
from skimage import transform
from skimage import draw
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import shutil
import random
import pandas as pd
import sys
import time
import pickle
import importlib
import cv2
import pywt
import matplotlib.patches as patches

from multiprocessing import Pool, Process, Queue


#custom packages
function_dir = os.path.join(os.pardir, 'Functions')
sys.path.append(function_dir)

import segmentationv2 as seg
importlib.reload(seg) #to make sure the last version of seg is used

def scan_file(base):
    #scan the arborescence from the base until finding a folder called img
    #return a list of all path to img folders
    #create 'objects', img_flat' and 'crops' folders if they do not exist yet
    try:
        os.listdir(base)
    except OSError as error : 
        return []
    #check if there are any jpg in the folder
    is_img = ['.jpg' in name for name in os.listdir(base)]
    #if there is an image, add the folder to the list
    if True in is_img:
        return [base]
    else:
        path = []
        for folder in os.listdir(base): 
            folder_path = os.path.join(base,folder)
            path+=scan_file(folder_path)
    return path

def reset_file(base, objects = True, img_flat = True, ROI = True):
    if objects:
        if 'objects' in os.listdir(base):
            shutil.rmtree(os.path.join(base,'objects'))
    if img_flat:
        if 'img_flat' in os.listdir(base):
            shutil.rmtree(os.path.join(base,'img_flat'))
    if ROI:
        if 'ROI' in os.listdir(base):
            shutil.rmtree(os.path.join(base,'ROI'))
def create_tree(paths):
    l_path_obj = []
    l_path_flat = []
    l_path_ROI = []
    for path in paths:
        head, tail = path.split(sep='img\\')
        path_obj =os.path.join(head,'objects',tail)
        path_flat = os.path.join(head,'img_flat',tail)
        path_ROI = os.path.join(head,'ROI',tail)
        os.makedirs(path_obj)
        os.makedirs(path_flat)
        os.makedirs(path_ROI)
        l_path_obj.append(path_obj)
        l_path_flat.append(path_flat)
        l_path_ROI.append(path_ROI)
    return zip(paths, l_path_obj, l_path_flat,l_path_ROI),len(paths)
def find_new_tree(paths):
    l_path_new = []
    l_path_obj = []
    l_path_flat = []
    l_path_ROI = []
    for path in paths:
        head, tail = path.split(sep='img\\')
        path_obj =os.path.join(head,'objects',tail)
        path_flat = os.path.join(head,'img_flat',tail)
        path_ROI = os.path.join(head,'ROI',tail)
        if not os.path.isdir(path_ROI):
            #clean the other trees 
            shutil.rmtree(path_flat)
            shutil.rmtree(path_obj)
            
            os.makedirs(path_obj)
            os.makedirs(path_flat)
            os.makedirs(path_ROI)
            
            l_path_new.append(path)
            l_path_obj.append(path_obj)
            l_path_flat.append(path_flat)
            l_path_ROI.append(path_ROI)
    return zip(l_path_new, l_path_obj, l_path_flat,l_path_ROI),len(l_path_new)
def criterion(bundle):
    #if the ROI is big enough
    if bundle['area']>4000:
        if bundle['aspect ratio'] < 20 and bundle['aspect ratio']>1/20:
            if np.sum(bundle['freq hist'][100:])/np.sum(bundle['freq hist'])>0.1:
                return True
    return False
        




if __name__ == '__main__':
    threshold  =0.45
    mindiff = 10
    count = 0
    reset = False
    if len(sys.argv) <= 1:
        print ("Please input a dirtectory to scan and segment, aborting.")
    else:
        if len(sys.argv)>=3:
            reset = sys.argv[2]
        base = sys.argv[1]
        #reset previous segmentation
        if reset:
            reset_file(base)
        #find paths to images
        paths = scan_file(os.path.join(base,'img'))
        #create folders:
        if reset:
            paths_dest,n = create_tree(paths)
        else:
            paths_dest,n = find_new_tree(paths)
        print(str(n)+" different folder have been found, flattening")
        for idx_folder,(path_img,path_obj,path_flat,path_ROI) in enumerate(paths_dest):
            print("processing folder "+str(idx_folder+1)+"/"+str(len(paths)))
            img_ar = seg.create_aquisition_array(path_img)
            if 'img_flat.png' not in os.listdir(path_flat):
                print("flattening of the aquisition")
                med_img = seg.flatten_acquisition(img_ar,max_length=30).astype(np.uint8)
                imageio.imwrite(os.path.join(path_flat,'img_flat.png'), med_img)
            else:
                print("loading flatten image")
                med_img = skimage.io.imread(os.path.join(path_flat,'img_flat.png'))
            print("labelling")
            for idx_img,img in enumerate(img_ar):
                img_name = os.listdir(path_img)[idx_img]
                img_diff = img.astype(int)-med_img
                #get rid of empty images
                if np.min(img_diff)>-mindiff:
                    continue
                imgn = seg.normalize(img_diff)
                if idx_img ==3:
                    show = True
                else:
                    show = False
                label,sobel_frame,smooth = seg.label_img_noBG(img, med_img, threshold, shrink_factor=2, dil_size = 30, sigma = 10,
                                                        timed = 0,sobel_scale=10,plot=show,return_features=True)
                
                ROIs, corners, masks = seg.multichannel_segment([imgn,sobel_frame], label, margin = 30,
                         return_corners = True,return_mask = True, plot = show)
                bundle_l = []
                for idx_obj, (ROI, corner, mask) in enumerate(zip(ROIs, corners, masks)):
                    bundle = {}
                    bundle['id']= str(idx_folder)+'_'+str(idx_img)+'_'+str(idx_obj)
                    bundle['img source']=os.path.join(path_img,img_name)
                    bundle['freq hist'],trash=np.histogram(ROI[1],256,range = (0,255))
                    bundle['position'] = corner
                    bundle['area']=np.sum(mask)
                    bundle['mask']=mask
                    bundle['aspect ratio'] = ROI[0].shape[0]/ROI[0].shape[1]
                    bundle_l.append(bundle)
                    
                    if criterion(bundle):
                        imageio.imwrite(os.path.join(path_ROI,bundle['id']+'.png'), (255*ROI[0]).astype(np.uint8))
                        count +=1
                bundle_f = pd.DataFrame(bundle_l)
                with open(os.path.join(path_obj, "bundle_frame_"+str(idx_folder)+"_"+str(idx_img)+".pickle"), 'wb') as f:
                    pickle.dump(bundle_f, f)
