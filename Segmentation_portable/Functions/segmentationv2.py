# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:04:37 2021

@author: Gabriel Vallat
"""

import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage import filters
import skimage
import matplotlib.pyplot as plt
from matplotlib import patches
import math
import time
import os
def normalize(img, epsilon = 0.001):
    #epsilon is here to stabilize the image
    if type(img)==int:
        epsilon*=256
    min_img = np.min(img)
    range_img = np.max(img)-min_img+epsilon
    img_n = (img-min_img)/range_img
    return img_n
def create_aquisition_array(img_path, idx = 0, maxsize = 1000):
    img_list = []
    list_name =  os.listdir(img_path)
    if len(list_name)-idx<=maxsize:
        list_name = list_name[idx:]
    else:
        list_name = list_name[idx:idx+maxsize]
        print("multibatch mode")
    for name in list_name:
        try:
            img_list.append(skimage.io.imread(os.path.join(img_path,name)))
        except:
            continue
    if len(img_list)==0:
        print ("No images were found")
        return
    else:
        return np.array(img_list)
def flatten_acquisition(img_ar, max_length = None):
    if max_length is not None and max_length < img_ar.shape[0]:
        flat_img = np.median(img_ar[0:max_length],axis=0)
    else:
        flat_img = np.median(img_ar,axis = 0)
    return flat_img
def label_standard(imgn, threshold,shrink_factor=2, dil_size = 24, median_size = 20,
                   sigma = 7, timed = 0,sobel_factor=10, plot=False,return_features=False):
    #correct the different parameters according to the shrink factor:
    dil_size //= shrink_factor
    median_size //=shrink_factor
    
    if timed:
        t0 = time.perf_counter()
    #remove static elements that stays during the whole aquisition, as well as
    #hue shift throughout the image
    #shrink the image to fasten the calculations, and convert it to grayscale
    img_small = (np.sum(imgn[0::shrink_factor,0::shrink_factor],axis=2)/3)    
    if timed:
        t1 = time.perf_counter()
    img_smooth = filters.gaussian(img_small)
    if timed:
        t2 = time.perf_counter()
    
    
    
    #binarize
    img_bin = img_smooth < threshold
    if timed:
        t3 = time.perf_counter()
    
    #morphological dilation to connect the different parts
    img_op = morphology.binary_erosion(img_bin)
    img_dil = morphology.binary_dilation(img_op,selem = morphology.disk(dil_size))
    
    if timed:
        t4 = time.perf_counter()
    
    #binary watershed
    img_label, n_ROI = ndimage.label(img_dil.astype(int))
    #recover a full size img:
    img_full = img_label.repeat(shrink_factor, axis=0).repeat(shrink_factor, axis=1)
    if timed:
        t5 = time.perf_counter()          
    
    if return_features:
        sobel = filters.sobel(img_smooth)
        #the image is not normalized to have a comparation possible bewteen 
        #different images
        sobel_uint8 = (sobel*255*sobel_factor).astype(np.uint8)
        sobel_med = filters.rank.median(sobel_uint8, morphology.disk(median_size))
        sobel_full = sobel_med.repeat(shrink_factor, axis=0).repeat(shrink_factor, axis=1)
        smooth_full = img_smooth.repeat(shrink_factor, axis=0).repeat(shrink_factor, axis=1)
    #print performances:
    if timed > 1:
        print('the preprocessing took: \n - '
             +str(t1-t0)+'s to normalize and substract the mean\n - '
             +str(t2-t1)+'s to smoth the image\n - '
             +str(t3-t2)+'s to binarize it\n - '
             +str(t4-t3)+'s to dilate it\n - '
             +str(t5-t4)+'s to label it\n ----- TOTAL: '+str(t5-t0)+'s')
    
    if plot:
        #plot
        fig, ax = plt.subplots(3, 2, figsize=(15, 18))
        
        ax[0,0].imshow(imgn)
        ax[0,0].set_title('original img (normalized over the dataset)')
        ax[0,0].axis('off')
        
        ax[0,1].set_title('smooth img')
        ax[0,1].imshow(img_smooth,vmin =0,vmax=1,cmap='jet')
        ax[0,1].axis('off')
        
        ax[1,0].set_title('binarized img')
        ax[1,0].imshow(img_bin)
        ax[1,0].axis('off')
        
        ax[1,1].set_title('img dilated')
        ax[1,1].imshow(img_dil)
        ax[1,1].axis('off')
        
        ax[2,0].set_title('img labeled')
        ax[2,0].imshow(img_label)
        ax[2,0].axis('off')
        
        if return_features:
            ax[2,1].set_title('median sobel')
            ax[2,1].imshow(sobel_med,cmap='jet')
            ax[2,1].axis('off')
        plt.show()
    if return_features:
        ret = [img_full,sobel_full,smooth_full]
    else:
        ret = img_full
    if timed:
        return ret, t5-t0
    
    return ret
def label_img_noBG(img, flat_img, threshold=0.35, shrink_factor=2, dil_size = 12, sigma = 14,
                   timed = 0,sobel_scale=10,plot=False,return_features=False,op_size = 3):
    #correct the different parameters according to the shrink factor:
    dil_size //= shrink_factor
    op_size //= shrink_factor
    sigma //= shrink_factor
    if timed:
        t0 = time.perf_counter()
    #remove static elements that stays during the whole aquisition, as well as
    #hue shift throughout the image
    img_meanless = img.astype(int)-flat_img
    #shrink the image to fasten the calculations, and convert it to grayscale
    img_small = np.sum(img_meanless[0::shrink_factor,0::shrink_factor],axis=2)
    img_n = normalize(img_small,epsilon=0.01)
    
    if timed:
        t1 = time.perf_counter()
    img_smooth = filters.gaussian(img_n)
    if timed:
        t2 = time.perf_counter()
    
    
    
    #binarize
    img_bin = img_smooth < threshold
    if timed:
        t3 = time.perf_counter()
    
    #morphological dilation to connect the different parts
    img_op = morphology.binary_erosion(img_bin,selem = morphology.disk(op_size))
    img_dil = morphology.binary_dilation(img_op,selem = morphology.disk(dil_size))
    
    if timed:
        t4 = time.perf_counter()
    
    #binary watershed
    img_label, n_ROI = ndimage.label(img_dil.astype(int))
    #recover a full size img:
    img_full = img_label.repeat(shrink_factor, axis=0).repeat(shrink_factor, axis=1)
    if timed:
        t5 = time.perf_counter()          
    
    if return_features:
        sobel = filters.sobel(img_smooth)
        #the image is not normalized to have a comparation possible bewteen 
        #different images
        sobel_uint8 = (sobel*255*sobel_scale).astype(np.uint8)
        sobel_med = filters.rank.mean(sobel_uint8, morphology.disk(5))
        sobel_full = sobel_med.repeat(shrink_factor, axis=0).repeat(shrink_factor, axis=1)
        smooth_full = img_smooth.repeat(shrink_factor, axis=0).repeat(shrink_factor, axis=1)
    #print performances:
    if timed > 1:
        print('the preprocessing took: \n - '
             +str(t1-t0)+'s to normalize and substract the mean\n - '
             +str(t2-t1)+'s to smoth the image\n - '
             +str(t3-t2)+'s to binarize it\n - '
             +str(t4-t3)+'s to dilate it\n - '
             +str(t5-t4)+'s to label it\n ----- TOTAL: '+str(t5-t0)+'s')
    
    if plot:
        #plot
        fig, ax = plt.subplots(3, 2, figsize=(15, 18))
        
        ax[0,0].imshow(img)
        ax[0,0].set_title('original img ')
        ax[0,0].axis('off')
        
        ax[0,1].set_title('meanless img')
        ax[0,1].imshow(img_n,vmin = 0,vmax = 1)
        ax[0,1].axis('off')
        
        ax[1,0].set_title('smooth img')
        ax[1,0].imshow(img_smooth,vmin=0,vmax=1)
        ax[1,0].axis('off')
        
        ax[1,1].set_title('binarized img')
        ax[1,1].imshow(img_bin)
        ax[1,1].axis('off')
        
        ax[2,0].set_title('img dilated')
        ax[2,0].imshow(img_dil)
        ax[2,0].axis('off')
        
        ax[2,1].set_title('img labeled')
        ax[2,1].imshow(img_label)
        ax[2,1].axis('off')
        

        plt.show()
    if return_features:
        ret = [img_full,sobel_full,smooth_full]
    else:
        ret = img_full
    if timed:
        return ret, t5-t0
    
    return ret

def sharpRegion(ROI, threshold=0.02, patchsize = 100,sigma=10):
    lp_roi = filters.gaussian(ROI,sigma)
    hp_roi = filters.sobel(lp_roi)
    nx = int(math.ceil(ROI.shape[0]/patchsize))
    ny = int(math.ceil(ROI.shape[1]/patchsize))
    patches = np.zeros(shape=(nx,ny))
    for x in range(nx):
        for y in range(ny):
            if x < nx-1:
                slicex = slice(patchsize*x,patchsize*(x+1))
            else:
                slicex = slice(patchsize*x,None,None)
            if y < ny-1:
                slicey = slice(patchsize*y,patchsize*(y+1))
            else:
                slicey = slice(patchsize*y,None,None)
            patch_to_test = hp_roi[slicex,slicey]
            if np.max(patch_to_test)>threshold:
                patches[x,y]=1
    return patches




def segment(img, img_label, shrink_factor=1,margin=20,return_tl = False,
            return_mask = False, plot=False):
    ROI_list = []
    tl_list = []
    mask_list = []
    
    n = np.max(img_label)
    h,w = img_label.shape
    if plot:
        fig = plt.subplot()
        fig.imshow(img)
    for i in range(1,n+1):
        x,y = np.where(img_label ==i)
        mask = np.zeros_like(img_label,dtype=np.int8)
        mask[x,y]=1
        xmin = shrink_factor*max(np.min(x)-margin,0)
        ymin = shrink_factor*max(np.min(y)-margin,0)
        xmax = shrink_factor*min(np.max(x)+margin,h)
        ymax = shrink_factor*min(np.max(y)+margin,w)
        ROI = img[xmin:xmax,ymin:ymax]
        
        ROI_list.append(ROI)
        mask_list.append(mask[xmin:xmax,ymin:ymax])
        tl_list.append((xmin,ymin))
        if plot:
            rect = patches.Rectangle((ymin,xmin), ymax-ymin, xmax-xmin, linewidth=1, edgecolor='r', facecolor='none')
            fig.add_patch(rect)
    if plot:
        plt.show()
    if not return_tl and not return_mask:
        return ROI_list
    else:
        ret = [ROI_list]
        if return_tl:
            ret.append(tl_list)
        if return_mask:
            ret.append(mask_list)
        return ret
def multichannel_segment(img_channels, img_label, shrink_factor = 1, margin = 25,
                         return_corners = False,return_mask = False, plot = False):
    #takes a list of images of the same size and return a list of crops taken with the 
    #same labels
    ROI_list = []
    corners_list = []
    mask_list = []
    ROI_multichannel = []
    n = np.max(img_label)
    h,w = img_label.shape
    if plot:
        fig,axis = plt.subplots(1,len(img_channels),figsize=(15, 18))
        for i,ax in enumerate(axis):
            ax.imshow(img_channels[i])
    for i in range(1,n+1):
        # if i==0, the background will be segmented
        x,y = np.where(img_label ==i)
        if return_mask:
            mask = np.zeros_like(img_label,dtype=np.int8)
            mask[x,y]=1
        xmin = shrink_factor*max(np.min(x)-margin,0)
        ymin = shrink_factor*max(np.min(y)-margin,0)
        xmax = shrink_factor*min(np.max(x)+margin,h)
        ymax = shrink_factor*min(np.max(y)+margin,w)
        ROI_multichannel = []
        for img in img_channels:
            ROI_multichannel.append(img[xmin:xmax,ymin:ymax])
        
        ROI_list.append(ROI_multichannel)
        if return_mask:
            mask_list.append(mask[xmin:xmax,ymin:ymax])
        if return_corners:
            corners_list.append(((xmin,ymin),(xmax,ymax)))
        if plot:
            for ax in axis:
                rect = patches.Rectangle((ymin,xmin), ymax-ymin, xmax-xmin, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
    if plot:
        plt.show()
    if not return_corners and not return_mask:
        return ROI_list
    else:
        ret = [ROI_list]
        if return_corners:
            ret.append(corners_list)
        if return_mask:
            ret.append(mask_list)
        return ret
    
    
    
    
    
  
def segment_corners(img_label, shrink_factor = 1, margin = 20):
    h,w = img_label.shape
    corner_list = []
    n = np.max(img_label)

    for i in range(1,n+1):
        # if i==0, the background will be segmented
        x,y = np.where(img_label ==i)
        xmin = shrink_factor*max(np.min(x)-margin,0)
        ymin = shrink_factor*max(np.min(y)-margin,0)
        xmax = shrink_factor*min(np.max(x)+margin,h)
        ymax = shrink_factor*min(np.max(y)+margin,w)
        corner_list.append(((xmin,ymin),(xmax,ymax)))
    return corner_list

def get_ROI(bundle):
    img_ROI = skimage.io.imread(bundle['img source'])
    ROI = img_ROI[bundle['xmin']:bundle['xmax'],bundle['ymin']:bundle['ymax']]
    return ROI