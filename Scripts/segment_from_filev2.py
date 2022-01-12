# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:24:24 2021

@author: valla

Create segmented images from a directory, as well as a list of ROI with calculated features
Usage:
    python segment_from_filev2 D:/irectory/to/scan args
    where args can be:
        reset: recompute all ROI
        mosaic: build a mosaic with the biggest roi (default size = 3000)
        mosaic=x: build a mosaic of size (x,x)
    
Requirement:
    -run this script on a folder containing an img folder
    -it will create an object, ROI and img_flat directories itself
    -if the reset keyworld is given, it will delete everything in the above 
     directories and recompute them, otherwise it will only compute the ones 
     where ROI does not exist
    -make sure that the dependencies for this script are in a folder called
     'Functions' in the parent directory
    
    
output:
    This script copies the arborecence of the img subfolder 3 times:
        -img_flat contains the median image of the dataset, and can be used to
         add some other combined pictures like mosaics
        -objects contains pandas dataframe (one dataframe per image) containing 
         information about the detected objects.
        -ROI contains png images of the objects that passed the selection
         criterion (currently using the size and sharpness of the objects to 
         only save usefull objects)

"""

#loading of the packages
import numpy as np
import imageio
import skimage.io
import os
import shutil
import pandas as pd
import sys
import pickle
import importlib




#custom packages:
#add the Function folder to the path
function_dir = os.path.join(os.pardir, 'Functions')
sys.path.append(function_dir)
#import it
import segmentationv2 as seg
import mosaic_from_file as mos

importlib.reload(seg) #to make sure the last version of seg is used

def scan_file(base):
    #scan the arborescence from the base until finding a folder called img
    #return a list of all path to img folders
    #create 'objects', img_flat' and 'crops' folders if they do not exist yet
    try:
        os.listdir(base)
    except OSError: 
        return []
    #check if there are any jpg in the folder
    is_img = ['.jpg' in name or '.png' in name for name in os.listdir(base)]
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
    #delete the given subfolders from the base
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
    #copy the substructure of img into objects, img_flat and ROI
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
    #create a list of paths where the ROI folder is missing, and reset the
    #corresponding objects and img_flat folders
    l_path_new = []
    l_path_obj = []
    l_path_flat = []
    l_path_ROI = []
    #create head folders
    for path in paths:
        head, tail = path.split(sep='img\\') #this line might be problematic with other os
        path_obj =os.path.join(head,'objects',tail)
        path_flat = os.path.join(head,'img_flat',tail)
        path_ROI = os.path.join(head,'ROI',tail)
        
        if not os.path.isdir(path_ROI):
            #clean the other trees 
            
            os.makedirs(path_ROI)
            try:
                os.makedirs(path_flat)
            except:
                pass
            try:
                shutil.rmtree(path_obj)
            except:
                pass
            os.makedirs(path_obj)
                
                
            l_path_new.append(path)
            l_path_obj.append(path_obj)
            l_path_flat.append(path_flat)
            l_path_ROI.append(path_ROI) 
    return zip(l_path_new, l_path_obj, l_path_flat,l_path_ROI),len(l_path_new)
def criterion(bundle):
    #if the ROI is big enough
    if bundle['area']>4000:
        #and if the aspect ratio is close enough to 1
        if bundle['aspect ratio'] < 20 and bundle['aspect ratio']>1/20:
            #and if the image is globally sharp
            if np.sum(bundle['freq hist'][100:])/np.sum(bundle['freq hist'])>0.1:
                return True
    return False
        




if __name__ == '__main__':
    threshold  =0.45
    mindiff = 10
    count = 0
    reset = False
    print_mosaic = False
    max_size=3000
    if len(sys.argv) <= 1:
        print ("Please input a dirtectory to scan and segment, aborting.")
    else:
        if len(sys.argv)>=3:
            if 'reset' in sys.argv:
                #delete the previously computed ROI
                reset = True
            for arg in sys.argv:
                if 'mosaic' in arg:
                    #print the mosaic of the aquisition in img_flat
                    print_mosaic = True
                    split = arg.split(sep='=')
                    if len(split)>1:
                        max_size = int(split[1])
    
                
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
            
        print(str(n)+" different folder have been found")
        
        for idx_folder,(path_img,path_obj,path_flat,path_ROI) in enumerate(paths_dest):
            print("processing folder "+str(idx_folder+1)+"/"+str(n))
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
                    id = str(idx_folder)+'_'+str(idx_img)+'_'+str(idx_obj)
                    image_source = os.path.join(path_img,img_name)
                    bundle = seg.create_bundle(ROI[1],corner,mask, id, image_source)
                    bundle_l.append(bundle)
                    if criterion(bundle):
                        imageio.imwrite(os.path.join(path_ROI,bundle['id']+'.png'), (255*ROI[0]).astype(np.uint8))
                        count +=1
                        
                bundle_f = pd.DataFrame(bundle_l)
                with open(os.path.join(path_obj, "bundle_frame_"+str(idx_folder)+"_"+str(idx_img)+".pickle"), 'wb') as f:
                    pickle.dump(bundle_f, f)
            mos.build_mosaic_from_folder(path_ROI, path_flat,max_size = max_size)
    print('done, '+str(count)+' png files were created')