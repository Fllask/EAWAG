# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:05:19 2021

@author: Gabriel Vallat

Take all images from a folder and build a mosaic, and print the image in the 
folder defined. the size of the mosaic can be defined (in pixels) (3000x3000 by
default)

usage:
    python mosaic_from_file S:/ource/directory D:/estination/directory 
    or 
    python mosaic_from_file S:/ource/directory D:/estination/directory max_size
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import sys


def find_img(base):
    #return a list of all path to img, going iteratively from base
    img_list = []
    try:
        os.listdir(base)
    except OSError: 
        return []
    #check if there are any jpg in the folder
    for name in os.listdir(base):
        path = os.path.join(base,name)
        if '.jpg' in name or '.png' in name:
            img_list += [path]
        else: 
            img_list += find_img(path)
    return img_list

def build_mosaic(img_list, max_size=1000):

    
    img_list.sort(key=lambda a: a.shape[0]*a.shape[1] ,reverse=True)
    img_arr = np.array(img_list,dtype=object)
    mos = np.zeros((max_size,max_size,3))
    available = np.ones((max_size,max_size))
    interest_corners = [(0,0)]
    for idx_img,img in enumerate(img_arr):
        for idx, ic in enumerate(interest_corners):
            if ic[0]+img.shape[0]<max_size and ic[1]+img.shape[1]<max_size:
                if available[ic[0]:ic[0]+img.shape[0],ic[1]:ic[1]+img.shape[1]].all():
                    mos[ic[0]:ic[0]+img.shape[0],ic[1]:ic[1]+img.shape[1],:]=img
                    available[ic[0]:ic[0]+img.shape[0],ic[1]:ic[1]+img.shape[1]]=0
                    interest_corners.pop(idx)
                    interest_corners.append((ic[0],ic[1]+img.shape[1]))
                    interest_corners.append((ic[0]+img.shape[0],ic[1]))
                    break
    if np.max(mos) > 1:
        mos /= np.max(mos)
    return mos
def build_mosaic_from_folder(source,dest, max_size = 1000):
    name_list = find_img(source)
    img_list = [plt.imread(name) for name in name_list]
    mos = build_mosaic(img_list,max_size = max_size)
    plt.imsave(os.path.join(dest,'mosaic.png'),mos)
    plt.imshow(mos)
    plt.show()
if __name__ == '__main__':
    max_size = 3000
    if len(sys.argv) <= 2:
        print ("Please input a directory of images and a destination aborting.")

    else:
        if len(sys.argv) > 3:
            max_size = sys.argv[3]
        source = sys.argv[1]
        dest = sys.argv[2]
        build_mosaic_from_folder(source,dest,max_size= max_size)