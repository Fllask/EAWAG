# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:06:39 2022

@author: valla
"""
import numpy as np
import pandas as pd
import warnings
import copy

def score(y_true,y_pred,mean_y_train, epsilon=1e-16):
    
    MAEn = 1-np.nanmean(np.abs(y_true-y_pred))/np.nanmean(np.abs(y_true-mean_y_train)+epsilon)
    R2_cat = 1-(np.nanmean(np.square(y_true-y_pred),axis=0)/
                np.nanmean(np.square(y_true-mean_y_train),axis=0)+epsilon)
    #note: the aggregation of R2 is quite tricky, as the two variables have highly different variances
    # the chosen method was to simply take the mean of the R2 coefficients of each variables
    R2 = np.nanmean(R2_cat)
    return {'R2':R2,'MAEn':MAEn,'R2_cat':R2_cat}

def kfold(model,input_tv,target_tv,n_fold):
    breakpoints = np.linspace(0,len(input_tv),n_fold+1,dtype=int)
    res = []
    for fold in range(n_fold):
        print(f"                                                 Fitting fold n°{fold}        ",end='\r')
        #define the validation split
        input_v = input_tv[breakpoints[fold]:breakpoints[fold+1]]
        y_v  = target_tv[breakpoints[fold]:breakpoints[fold+1]]
        #define the training split
        input_t = pd.concat([input_tv[:breakpoints[fold]],input_tv[breakpoints[fold+1]:]])
        y_tr = pd.concat([target_tv[:breakpoints[fold]],target_tv[breakpoints[fold+1]:]])
        #train the model
        model.fit(input_t,y_tr,[breakpoints[fold],breakpoints[fold+1]])
        #predict both the training data, and the validation
        y_predtr = model.pred(input_t,[breakpoints[fold],breakpoints[fold+1]])
        y_predv = model.pred(input_v,[0,0])
        #store the mean of the training data
        mean_ytrain = np.nanmean(y_tr,axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s_train = score(y_tr.to_numpy(),y_predtr,mean_ytrain)
            s_val = score(y_v.to_numpy(),y_predv,mean_ytrain)
        res.append({'model':copy.deepcopy(model),'fold':fold,'bp': breakpoints,'y_predtr':y_predtr, 'y_predv':y_predv,
                    's_train':s_train,'s_val':s_val,'outkeys':target_tv.keys(),'mean_ytrain':mean_ytrain})
    print(" "*100,end='\r')
    return res