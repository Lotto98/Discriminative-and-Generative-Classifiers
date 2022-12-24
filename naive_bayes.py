import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

from scipy.stats import beta

class BetaDistribution_NaiveBayes(BaseEstimator):
    
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self,train_X:pd.DataFrame,train_y:pd.DataFrame):
        self.train_X=train_X
        self.train_y=train_y
        p=self.__param_estimation()
        
        return self,p
    
    def __param_estimation(self) -> None:
        
        self.parameter_per_class={}
        
        for n in range(10):
            
            images_class_n=self.train_X[self.train_y["class"]==n]
            
            #images_class_n=images_class_n.apply(lambda x: x + np.abs(np.random.normal(1,0.0001,1)) )
            
            means_pixels_class_n=images_class_n.mean(axis=0)
            variances_pixels_class_n=images_class_n.var(axis=0)
            
            means_squared_pixels_class_n=(images_class_n**2).mean(axis=0)
            
            unique_counts=images_class_n.nunique(axis=0, dropna=True)
            
            unique_counts[unique_counts > 1] = -1
            
            unique_counts[unique_counts == 1] = means_pixels_class_n[unique_counts == 1]
            
            ks_pixels_class_n=((means_pixels_class_n*(1-means_pixels_class_n))/variances_pixels_class_n)-1
            
            alphas_pixels_class_n=ks_pixels_class_n*means_pixels_class_n
            betas_pixels_class_n=ks_pixels_class_n*(1-means_pixels_class_n)
            
            alphas_pixels_class_n[alphas_pixels_class_n<=0]=alphas_pixels_class_n[alphas_pixels_class_n>0].min()
            betas_pixels_class_n[betas_pixels_class_n<=0]=betas_pixels_class_n[betas_pixels_class_n>0].min()
                
                
            frequency=self.train_y[self.train_y["class"]==n].size/self.train_y["class"].size
                
            self.parameter_per_class[n]={'alpha':alphas_pixels_class_n.to_numpy(),
                                        'beta':betas_pixels_class_n.to_numpy(),
                                        'unique':unique_counts.to_numpy(),
                                        'mean':means_pixels_class_n,
                                        'squared_mean':means_squared_pixels_class_n,
                                        
                                        #'var':variances_pixels_class_n,
                                        #'k':ks_pixels_class_n,
                                        'frequency':frequency}
            #print(self.parameter_per_class)
    
    def predict(self,test_X:pd.DataFrame) -> pd.Series:
    
        epsilon=0.1
        
        output=[]
        indexes=[]
        
        for row_i,row in test_X.iterrows():
                
            _max=0
            _max_class=-1
            
            row=row.to_numpy()
             
            for n in range(10):
                
                class_parameters=self.parameter_per_class[n]
                
                _alpha=class_parameters['alpha']
                _beta=class_parameters['beta']
                _unique=class_parameters['unique']
                
                _mean=class_parameters['mean']
                _squared_mean=class_parameters['squared_mean']
                
                
                beta_probabilities=beta.cdf(row+epsilon,_alpha,_beta)-beta.cdf(row-epsilon,_alpha,_beta)               
                
                #unique values
                
                beta_probabilities[ np.logical_and(_unique!=-1, _unique != row) ] = 0
                beta_probabilities[ np.logical_and(_unique!=-1, _unique == row) ] = 1
                
                #alpha and beta < 0
                
                #print("alpha",_alpha[ np.argwhere( np.isnan(beta_probabilities) ) ])
                #print("beta",_beta[ np.argwhere( np.isnan(beta_probabilities) ) ])
                
                """
                for e,p in enumerate(beta_probabilities):
                    if  np.isnan(p):
                        print(_mean[e]<=_squared_mean[e])
                """
                        
                np.nan_to_num(beta_probabilities, copy=False, nan=1.0)
                
                probability=class_parameters['frequency']*np.product(beta_probabilities)
                
                if probability>_max:
                    _max=probability
                    _max_class=n
                    #print(_max)
            
            output.append(_max_class)
            indexes.append(row_i)
            
        return pd.Series(data=output,index=indexes)
        