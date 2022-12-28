import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

from scipy.stats import beta

import matplotlib.pyplot as plt

class BetaDistribution_NaiveBayes(BaseEstimator):
    
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self,train_X:pd.DataFrame,train_y:pd.DataFrame):
        self.train_X=train_X
        self.train_y=train_y
        self.__param_estimation()
        
        return self
    
    def __param_estimation(self) -> None:
        
        self.parameter_per_class={}
        
        for n in range(10):
            
            #images of class n
            images_class_n=self.train_X[self.train_y["class"]==n]
            
            #mean and variance for each pixel of class n
            means_pixels_class_n=images_class_n.mean(axis=0)
            variances_pixels_class_n=images_class_n.var(axis=0)
            
            #alpha and beta estimation
            ks_pixels_class_n=((means_pixels_class_n*(1-means_pixels_class_n))/variances_pixels_class_n)-1
            
            alphas_pixels_class_n=ks_pixels_class_n*means_pixels_class_n
            betas_pixels_class_n=ks_pixels_class_n*(1-means_pixels_class_n)
            
            #class frequency    
            frequency=self.train_y[self.train_y["class"]==n].size/self.train_y["class"].size
            
            #negative alpha and beta
            alphas_pixels_class_n[alphas_pixels_class_n<=0]=alphas_pixels_class_n[alphas_pixels_class_n>0].min()
            betas_pixels_class_n[betas_pixels_class_n<=0]=betas_pixels_class_n[betas_pixels_class_n>0].min()
            
            #unique value pixel (for nan value alphas and betas)
            unique_counts=images_class_n.nunique(axis=0, dropna=True)
            
            unique_counts[unique_counts > 1] = -1
            
            unique_counts[unique_counts == 1] = means_pixels_class_n[unique_counts == 1]
            
            #Beta means
            beta_means_class_n=(alphas_pixels_class_n)/(alphas_pixels_class_n+betas_pixels_class_n)
            beta_means_class_n[unique_counts != -1]=unique_counts[unique_counts != -1]
            
            self.parameter_per_class[n]={'alphas':alphas_pixels_class_n.to_numpy(),
                                        'betas':betas_pixels_class_n.to_numpy(),
                                        'unique':unique_counts.to_numpy(),
                                        'Beta_means':beta_means_class_n.to_numpy(),
                                        'frequency':frequency}
    
    def predict(self,test_X:pd.DataFrame) -> pd.Series:
    
        epsilon=0.1
        
        output=[]
        indexes=[]
        
        for row_i,row in test_X.iterrows():
                
            _max=0
            _max_class=-1
            
            row=row.to_numpy()
             
            for n in range(10):
                
                #parameters
                class_parameters=self.parameter_per_class[n]
                
                _alpha=class_parameters['alphas']
                _beta=class_parameters['betas']
                _unique=class_parameters['unique']
                
                #integral
                beta_probabilities=beta.cdf(row+epsilon,_alpha,_beta)-beta.cdf(row-epsilon,_alpha,_beta)               
                
                #unique values
                beta_probabilities[ np.logical_and(_unique!=-1, _unique != row) ] = 0
                beta_probabilities[ np.logical_and(_unique!=-1, _unique == row) ] = 1
                
                probability=class_parameters['frequency']*np.product(beta_probabilities)
                
                if probability>_max:
                    _max=probability
                    _max_class=n
            
            output.append(_max_class)
            indexes.append(row_i)
            
        return pd.Series(data=output,index=indexes)
    
    def mean_plot(self)->None:
        
        _,axs=plt.subplots(3,4)
        
        axs = [item for sublist in axs for item in sublist]
        
        axs[10].axis('off')
        axs[11].axis('off')
        
        for digit in range(10):
            
            means_class_n=self.parameter_per_class[digit]["Beta_means"]
            
            axs[digit].imshow(means_class_n.reshape(28, 28))
            axs[digit].axis('off')