# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:54:01 2022

@author: valla
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import signal
import warnings

### models
class AR:
    def __init__(self,target_keys:list,k,CP=False):
        
        #for the implementation, a different model is used for each variables:
        self.skmodels = [LinearRegression() for _ in target_keys]
        self.k = k
        self.target_keys = target_keys
        self._scalerin = StandardScaler()
        self._scalerout = StandardScaler()
        self.idxtarget = []
        self.CP = CP
    def fit(self,inputs,target,breakpoints):
        if not self.CP:
            inputs = inputs.drop(columns=['Datetime'],errors='ignore')
            self._scalerin = self._scalerin.fit(inputs.values)
            self._scalerout = self._scalerout.fit(target.values)
            ninp = self._scalerin.transform(inputs.values)
            #fill nans with 0s
            ninp = np.nan_to_num(ninp)
            ntar = self._scalerout.transform(target.values)
            isnan = np.isnan(ntar);
            ntar = np.nan_to_num(ntar)
            for idx,target in enumerate(self.target_keys):
                #stack the input variable so that k points are given for each target:
                idx_x = inputs.keys().get_loc(target)
                self.idxtarget.append(idx_x)
                x = np.zeros((len(inputs)-self.k+1,self.k))
                for i in range(self.k):
                    x[:,i] = ninp[i:len(inputs)-self.k+1+i,idx_x]
                y = ntar[self.k-1:,idx]
                w = ~isnan[self.k-1:,idx]
                self.skmodels[idx].fit(x,y,sample_weight=w)
    def pred(self,inputs,breakpoints):
        if self.CP:
            if len(inputs)==0:
                return np.zeros((0,len(self.idxtarget)))
            else:
                return(inputs[self.target_keys].values)
        else:
            inputs = inputs.drop(columns=['Datetime'],errors='ignore')
            if len(inputs)==0:
                return np.zeros((len(inputs),len(self.target_keys)))
            ninp = self._scalerin.transform(inputs.values)
            ninp = np.nan_to_num(ninp)
            npred = np.empty((len(inputs),len(self.target_keys)))
            npred[:,:] = np.nan
            for idx,target in enumerate(self.target_keys):
                x = np.zeros((len(inputs)-self.k+1,self.k))
                for i in range(self.k):
                    x[:,i] = ninp[i:len(inputs)-self.k+i+1,self.idxtarget[idx]]
                npred[self.k-1:,idx] = self.skmodels[idx].predict(x)
            return self._scalerout.inverse_transform(npred)    
class Linear:
    def __init__(self,target_keys:list,k):
        
        #for the implementation, a different model is used for each variables:
        self.skmodels = [LinearRegression() for _ in target_keys]
        self.k = k
        self.target_keys = target_keys
        self._scalerin = StandardScaler()
        self._scalerout = StandardScaler()
        self.idxtarget = []
    def fit(self,inputs,target,breakpoints):
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        self._scalerin = self._scalerin.fit(inputs.values)
        self._scalerout = self._scalerout.fit(target.values)
        ninp = self._scalerin.transform(inputs.values)
        #fill nans with 0s
        ninp = np.nan_to_num(ninp)
        ntar = self._scalerout.transform(target.values)
        isnan = np.isnan(ntar);
        ntar = np.nan_to_num(ntar)
        for idx,target in enumerate(self.target_keys):
            #stack the input variable so that k points are given for each target:
            x = np.zeros((len(inputs)-self.k+1,self.k*len(inputs.keys())))
            for i in range(self.k):
                for j in range(len(inputs.keys())):
                    x[:,i*len(inputs.keys())+j] = ninp[i:len(inputs)-self.k+1+i,j]
            y = ntar[self.k-1:,idx]
            w = ~isnan[self.k-1:,idx]
            self.skmodels[idx].fit(x,y,sample_weight=w)
    def pred(self,inputs,breakpoints):
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        if len(inputs)==0:
            return np.zeros((len(inputs),len(self.target_keys)))
        ninp = self._scalerin.transform(inputs.values)
        ninp = np.nan_to_num(ninp)
        npred = np.empty((len(inputs),len(self.target_keys)))
        npred[:,:] = np.nan
        for idx,target in enumerate(self.target_keys):
            x = np.zeros((len(inputs)-self.k+1,self.k*len(inputs.keys())))
            for i in range(self.k):
                for j in range(len(inputs.keys())):
                    x[:,i*len(inputs.keys())+j] = ninp[i:len(inputs)-self.k+1+i,j]
            npred[self.k-1:,idx] = self.skmodels[idx].predict(x)
        return self._scalerout.inverse_transform(npred)   
class Lasso:
    def __init__(self,target_keys:list,k, **lasso_kwarg):
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
        self.skmodels = [Lasso(**lasso_kwarg) for _ in target_keys]
        self._scalerin = StandardScaler()
        self._scalerout = StandardScaler()
        self.target_keys = target_keys
        self.k = k
    def fit(self,inputs,target,breakpoints):
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        self._scalerin = self._scalerin.fit(inputs.values)
        target = target.to_numpy()
        self._scalerout = self._scalerout.fit(target)
        ninp = self._scalerin.transform(inputs.values)
        #fill nans with 0s
        ninp = np.nan_to_num(ninp)
        ntar = self._scalerout.transform(target)
        isnan = np.isnan(ntar);
        ntar = np.nan_to_num(ntar)
        #stack the input variable so that k points are given for each target:
        x = np.zeros((len(inputs)-self.k+1,self.k*len(inputs.keys())))
        for keyid in range(len(self.target_keys)):
            for i in range(self.k):
                for j in range(len(inputs.keys())):
                    x[:,i*len(inputs.keys())+j] = ninp[i:len(inputs)-self.k+1+i,j]
            y = ntar[self.k-1:,keyid]
            w = ~isnan[self.k-1:,keyid]
            self.skmodels[keyid].fit(x,y,sample_weight=w)
    def pred(self,inputs,breakpoints):
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        if len(inputs)==0:
            return np.zeros((len(inputs),len(self.target_keys)))
        ninp = self._scalerin.transform(inputs.values)
        ninp = np.nan_to_num(ninp)
        npred = np.empty((len(inputs),len(self.target_keys)))
        npred[:,:] = np.nan
        for idx,target in enumerate(self.target_keys):
            x = np.zeros((len(inputs)-self.k+1,self.k*len(inputs.keys())))
            for i in range(self.k):
                for j in range(len(inputs.keys())):
                    x[:,i*len(inputs.keys())+j] = ninp[i:len(inputs)-self.k+1+i,j]
            if npred.shape[1]==1:
                npred[self.k-1:]=np.expand_dims(self.skmodels[idx].predict(x),-1)
            else:
                npred[self.k-1:,idx] = self.skmodels[idx].predict(x)
        return self._scalerout.inverse_transform(npred)
class Sparse:
    def __init__(self,target_keys:list,k, freq):
        
        #for the implementation, a different model is used for each variables:
        self.skmodels = [LinearRegression() for _ in target_keys]
        self.kf = k*freq
        self.k = k
        self.target_keys = target_keys
        self._scalerin = StandardScaler()
        self._scalerout = StandardScaler()
        self.freq = freq
        self.idxtarget = []
    def fit(self,inputs,target,breakpoints):
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        self._scalerin = self._scalerin.fit(inputs.values)
        self._scalerout = self._scalerout.fit(target.values)
        ninp = self._scalerin.transform(inputs.values)
        #fill nans with 0s
        ninp = np.nan_to_num(ninp)
        ntar = self._scalerout.transform(target.values)
        isnan = np.isnan(ntar);
        ntar = np.nan_to_num(ntar)
        for idx,target in enumerate(self.target_keys):
            #stack the input variable so that k points are given for each target:
            x = np.zeros((len(inputs)-self.kf+self.freq,self.k*len(inputs.keys())))
            for i in range(0,self.k):
                for j in range(len(inputs.keys())):
                    x[:,i*len(inputs.keys())+j] = ninp[i:len(inputs)-self.kf+self.freq+i,j]
            y = ntar[self.kf-self.freq:,idx]
            w = ~isnan[self.kf-self.freq,idx]
            self.skmodels[idx].fit(x,y,sample_weight=w)
    def pred(self,inputs,breakpoints):
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        if len(inputs)==0:
            return np.zeros((len(inputs),len(self.target_keys)))
        ninp = self._scalerin.transform(inputs.values)
        ninp = np.nan_to_num(ninp)
        npred = np.empty((len(inputs),len(self.target_keys)))
        npred[:,:] = np.nan
        for idx,target in enumerate(self.target_keys):
            x = np.zeros((len(inputs)-self.kf+self.freq,self.k*len(inputs.keys())))
            for i in range(0,self.k):
                for j in range(len(inputs.keys())):
                    x[:,i*len(inputs.keys())+j] = ninp[i*self.freq:len(inputs)-self.kf+self.freq+i*self.freq,j]
            npred[self.kf-self.freq:,idx] = self.skmodels[idx].predict(x)
        return self._scalerout.inverse_transform(npred)   
class ElastNet:
    def __init__(self,target_keys:list,k, freq=1,**elast_kwarg):
        from sklearn.linear_model import ElasticNet

        #for the implementation, a different model is used for each variables:
        self.skmodels = [ElasticNet(**elast_kwarg) for _ in target_keys]
        self.kf = k*freq
        self.k = k
        self.target_keys = target_keys
        self._scalerin = StandardScaler()
        self._scalerout = StandardScaler()
        self.freq = freq
        self.idxtarget = []
    def fit(self,inputs,target,breakpoints):
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        self._scalerin = self._scalerin.fit(inputs.values)
        self._scalerout = self._scalerout.fit(target.values)
        ninp = self._scalerin.transform(inputs.values)
        #fill nans with 0s
        ninp = np.nan_to_num(ninp)
        ntar = self._scalerout.transform(target.values)
        isnan = np.isnan(ntar);
        ntar = np.nan_to_num(ntar)
        #stack the input variable so that k points are given for each target:
        x = np.zeros((len(inputs)-self.kf+self.freq,self.k*len(inputs.keys())))
        for keyid in range(len(self.target_keys)):
            for i in range(0,self.k):
                for j in range(len(inputs.keys())):
                    x[:,i*len(inputs.keys())+j] = ninp[i:len(inputs)-self.kf+self.freq+i,j]
            y = ntar[self.kf-self.freq:,keyid]
            w = ~isnan[self.kf-self.freq:,keyid]
            self.skmodels[keyid].fit(x,y,sample_weight=w)
    def pred(self,inputs,breakpoints):
        
        
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        if len(inputs)==0:
            return np.zeros((len(inputs),len(self.target_keys)))
        ninp = self._scalerin.transform(inputs.values)
        ninp = np.nan_to_num(ninp)
        npred = np.empty((len(inputs),len(self.target_keys)))
        npred[:,:] = np.nan
        x = np.zeros((len(inputs)-self.kf+self.freq,self.k*len(inputs.keys())))
        for i in range(0,self.k):
            for j in range(len(inputs.keys())):
                x[:,i*len(inputs.keys())+j] = ninp[i*self.freq:len(inputs)-self.kf+self.freq+i*self.freq,j]
        for idx,target in enumerate(self.target_keys):
            npred[self.kf-self.freq:,idx] = self.skmodels[idx].predict(x)
        return self._scalerout.inverse_transform(npred)   
    
#implementation of the correlation:
class CorrElNet:
    def __init__(self,target_keys:list,n_feat,n_para=1,**elast_kwarg):
        from sklearn.linear_model import ElasticNet
        #for the implementation, a different model is used for each variables:
        self.skmodels = [ElasticNet(**elast_kwarg,fit_intercept=False) for _ in target_keys]
        self.n_feat = n_feat
        self.n_para = n_para
        assert n_feat%n_para == 0, "n_para must divide n_feat"
        self.target_keys = target_keys
        self._scalerin = StandardScaler()
        self._scalerout = StandardScaler()
        self.featureUsed = np.empty((len(target_keys),n_feat,2))
        self.featureUsed[:] = np.nan
    def fit(self,inputs,target,breakpoints):
        self.featureUsed[:] = np.nan
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        self._scalerin = self._scalerin.fit(inputs.values)
        self._scalerout = self._scalerout.fit(target.values)
        ninp = self._scalerin.transform(inputs.values)
        #fill nans with 0s
        ninp = np.nan_to_num(ninp)
        ntar = self._scalerout.transform(target.values)
        isnan = np.isnan(ntar);
        ntar = np.nan_to_num(ntar)
        
        #use the breakpoint to pad the correlation function with 0s
        ninp_pad = np.concatenate([ninp[:breakpoints[0]],
                                   np.zeros((breakpoints[1]-breakpoints[0],ninp.shape[1])),
                                   ninp[breakpoints[0]:]])
        ntar_pad = np.concatenate([ntar[:breakpoints[0]],
                                   np.zeros((breakpoints[1]-breakpoints[0],ntar.shape[1])),
                                   ntar[breakpoints[0]:]])
        isnan = np.concatenate([isnan[:breakpoints[0]],
                                np.true((breakpoints[1]-breakpoints[0],isnan.shape[1])),
                                isnan[breakpoints[0]:]])
        for idx,target in enumerate(self.target_keys):
            for tries in range(self.n_para):
                #find compute the correlation function
                marginal_target = np.copy(np.expand_dims(ntar_pad[:,idx],-1))
                for feature in range(self.n_feat//self.n_para):
                    cor = signal.correlate(ninp_pad,marginal_target, mode='full')/len(marginal_target)
                    cor_lag = signal.correlation_lags(ninp_pad.shape[0],marginal_target.shape[0], mode='full')
                    cor_f = cor[cor_lag<=0,:]
                    cor_lag_f = cor_lag[cor_lag<=0]
                    used_idx = np.reshape(self.featureUsed[idx,:][self.featureUsed[idx,:,:]==self.featureUsed[idx,:,:]].astype(int),
                                          [-1,2])
                    
                    used_idx[:,0]= len(cor_lag_f)-1-used_idx[:,0]
                    if len(used_idx)>0:
                        cor_f[used_idx]=np.nan
                    max_idx = np.unravel_index(np.nanargmax(np.abs(cor_f),axis=None),cor_f.shape)
                    best_lag = -cor_lag_f[max_idx[0]]
                    self.featureUsed[idx,feature+self.n_feat*tries//self.n_para,:]=np.array([best_lag,max_idx[1]])
                    marginal_target[best_lag:,0] = marginal_target[best_lag:,0] - cor[max_idx]*ninp_pad[:len(marginal_target)-best_lag,max_idx[1]]
                    marginal_target = (np.reshape(marginal_target,[-1,1]) - np.mean(marginal_target))/np.std(marginal_target)
        #stack the input variable so that n_feat points are given for each target:
        x = np.zeros((len(ninp_pad),self.n_feat))
        for idx_tar in range(len(self.target_keys)):
            shift = int(np.max(self.featureUsed[idx_tar,:,0]))
            ninp_shifted = np.concatenate([np.zeros((shift,ninp.shape[1])),ninp_pad])
            for i in range(self.n_feat):
                shift_feat = shift-self.featureUsed[idx_tar,i,0].astype(int)
                x[:,i] = ninp_shifted[shift_feat:len(ninp_shifted)-self.featureUsed[idx_tar,i,0].astype(int),
                                              self.featureUsed[idx_tar,i,1].astype(int)]
            y = ntar_pad[:,idx]
            w = ~isnan[:,idx]
            self.skmodels[idx_tar].fit(x,y,sample_weight=w)
    def pred(self,inputs,breakpoints):
        inputs = inputs.drop(columns=['Datetime'],errors='ignore')
        if len(inputs)==0:
            return np.zeros((0,len(self.target_keys)))
        ninp = self._scalerin.transform(inputs.values)
        #fill nans with 0s
        ninp = np.nan_to_num(ninp)
        
        #use the breakpoint to pad the correlation function with 0s
        ninp_pad = np.concatenate([ninp[:breakpoints[0]],
                                   np.zeros((breakpoints[1]-breakpoints[0],ninp.shape[1])),
                                   ninp[breakpoints[0]:]])
        npred = np.empty((len(inputs),len(self.target_keys)))
        npred[:,:] = np.nan
     
        #stack the input variable so that the point founds during training are used to predict y
        x = np.zeros((len(ninp),self.n_feat))
        x_pad = np.zeros((len(ninp_pad),self.n_feat))
        for idx_tar in range(len(self.target_keys)):
            shift = int(np.max(self.featureUsed[idx_tar,:,0]))
            ninp_shifted = np.concatenate([np.zeros((shift,ninp.shape[1])),ninp_pad])
            for i in range(self.n_feat):
                shift_feat = shift-self.featureUsed[idx_tar,i,0].astype(int)
                x_pad[:,i] = ninp_shifted[shift_feat:len(ninp_shifted)-self.featureUsed[idx_tar,i,0].astype(int),
                                              self.featureUsed[idx_tar,i,1].astype(int)]
            #remove the padding
            x = np.concatenate([x_pad[:breakpoints[0],:],x_pad[breakpoints[1]:,:]])
            npred[:,idx_tar] = self.skmodels[idx_tar].predict(x)
        return self._scalerout.inverse_transform(npred)

#plotting function
def R2_vs_lag(lags,res_model:list,tar_data:list,title="R2 vs lags",model_names = None,train = False):
    keys = res_model[0][0][0]['outkeys']
    fig,axs = plt.subplots(1,len(keys),figsize=(len(keys)*5,4))
    if len(keys)==1:
        axs = [axs]
    if train:
        artists =np.empty((2*len(res_model),len(keys)),dtype=plt.Line2D)
    else:
        artists = np.empty((len(res_model),len(keys)),dtype=plt.Line2D)
    if len(tar_data)==1:
        tar_data = tar_data * len(res_model)
    for idx_model,(res_lag,tar_lag) in enumerate(zip(res_model,tar_data)):
        r2mean_lag = []
        r2mean_lag_train = []
        for res,tar in zip(res_lag,tar_lag):
            if train:
                r2mean_lag_train.append(np.nanmean([res_fold['s_train']['R2_cat'] for res_fold in res],axis=0))
            r2mean_lag.append(np.nanmean([res_fold['s_val']['R2_cat'] for res_fold in res],axis=0))
        r2mean_lag = np.array(r2mean_lag)
        r2mean_lag_train = np.array(r2mean_lag_train)
        fig.suptitle(title)
        for idx,(ax,key) in enumerate(zip(axs,keys)):
            if train:
                artists[len(res_model)+idx_model,idx] = ax.plot(lags,r2mean_lag_train[:,idx])[0]
            artists[idx_model,idx] = ax.plot(lags,r2mean_lag[:,idx])[0]
            ax.hlines(0,min(lags),max(lags),color='k',linestyle='dashed')
            ax.set_title(key)
            ax.set_xlabel("lag [day]")
            ax.set_ylabel("mean R2 over all folds")
            ax.set_ylim(max(-0.1,np.min(r2mean_lag[:,idx])-0.1),1)
    if model_names is not None:
        if train:
            model_namest = [model_name + " (on training)" for model_name in model_names]
            [ax.legend(artists[:,idx],model_names+model_namest) for idx,ax in enumerate(axs)]
        else:
            [ax.legend(artists[:,idx],model_names) for idx,ax in enumerate(axs)]
    plt.show()
    
    
def plot_val(targets_tv,res,targets_test=None,title='',scale='linear'):
    
    fig,axs = plt.subplots(len(targets_tv.keys()),figsize = (15,6*len(targets_tv.keys())))
    if len(targets_tv.keys())==1:
        axs=[axs]
    for idx,(ax,key) in enumerate(zip(axs,targets_tv.keys())):
        
        
        artists = [ax.scatter(targets_tv.index,targets_tv[key].to_numpy(),s=1,c='k')]
        labels = ['True data']
        ax.set_title(f"{title} {key} (median R2={np.nanmedian([f['s_val']['R2_cat'][idx] for f in res]):.2f})")
        for fold in range(len(res[0]['bp'])-1):
            artists.append(ax.scatter(targets_tv.index[res[0]['bp'][fold]:res[0]['bp'][fold+1]],
                              res[fold]['y_predv'][:,idx],s=0.5))
            labels.append(f"Validation of fold {fold} (R2={res[fold]['s_val']['R2_cat'][idx]:.2f}))")
        if scale == 'linear':
            ax.set_ylabel("Concentration (linear units)")
        elif scale == 'log':
            ax.set_ylabel("Log concentration [log(c+1)]")
        ax.legend(artists,labels)
        if targets_test is not None:
            #the test predictions are not used for now
            #plot the mean and std of the test predictions
            #remove empty slice warning occuring when only nans are provided.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                testres = np.array([resi['y_predte'] for resi in res])
                test_mean = np.nanmean(testres,axis=0)
                test_std = np.nanstd(testres,axis= 0)
            ax.scatter(targets_test.index,targets_test[key],c='k',s=1)
            ax.scatter(targets_test.index,test_mean[:,idx],s=0.5)
            ax.fill_between(targets_test.index,
                            test_mean[:,idx] - 1.96*test_std[:,idx],
                            test_mean[:,idx] + 1.96*test_std[:,idx],
                            color='gray', alpha=0.2)
            
#display all none-zero coefficients
def show_coefs(res,in_keys):
    
    out_keys = res[0]['outkeys']
    param_idx = np.zeros((len(in_keys),len(out_keys)))
    for outidx in range(len(out_keys)):
        for fold in range(len(res)):
            abscoefs = np.abs(res[fold]['model'].skmodels[outidx].coef_)
            non_zero = np.where(abscoefs !=0)[0]
            param_idx[non_zero,outidx]+=1 #add a count
    for idxout, out_key in enumerate(out_keys):
        print(f"For the forecasting of {out_key}, the following key were \nused in the following proportion of the folds:")
        idx_used = np.where(param_idx[:,idxout] != 0)[0]
        for idx in idx_used:
            print(f"    {in_keys[idx]}: {int(param_idx[idx,idxout])}/{len(res)}")