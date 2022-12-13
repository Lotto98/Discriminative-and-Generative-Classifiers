import pandas as pd
from sklearn.base import BaseEstimator
from scipy.integrate import quad

from decimal import *

import math

class BetaDistribution_NaiveBayes(BaseEstimator):
    
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self,train_X:pd.DataFrame,train_y:pd.DataFrame):
        self.train_X=train_X
        self.train_y=train_y
        self.__param_estimation()
    
    @staticmethod
    def __B(x, alpha, beta):
        return (x**(alpha-1))*((1-x)**beta-1)
    
    
    def __param_estimation(self):
        
        self.parameter_per_class={}
        
        for n in range(10):
            
            images_class_n=self.train_X[self.train_y["class"]==n]
            
            means_pixels_class_n=images_class_n.mean(axis=0)
            variances_pixels_class_n=images_class_n.var(axis=0)
        
            ks_pixels_class_n=((means_pixels_class_n*(1-means_pixels_class_n))/variances_pixels_class_n)-1
            
            ks_pixels_class_n=ks_pixels_class_n.fillna(0)
            
            alphas_pixels_class_n=ks_pixels_class_n*means_pixels_class_n
            betas_pixels_class_n=ks_pixels_class_n*(1-means_pixels_class_n)
            
            integrals_pixels_class_n=[]
            
            for item in zip(alphas_pixels_class_n, betas_pixels_class_n):
                integral=quad(BetaDistribution_NaiveBayes.__B,0,1,args=(item[0],item[1]))
                integrals_pixels_class_n.append(integral[0]+integral[1])
                
            frequency=self.train_y[self.train_y["class"]==n].size/self.train_y["class"].size
                
            self.parameter_per_class[n]={'alpha':alphas_pixels_class_n,
                                        'beta':betas_pixels_class_n,
                                        'integral':integrals_pixels_class_n,
                                        'frequency':frequency}
    def predict(self,test_X:pd.DataFrame):
        
        output=[]
        indexes=[]
        
        for row_i,row in test_X.iterrows():
                
            _max=0
            _max_class=None
            
            for n in range(10):
                
                class_parameters=self.parameter_per_class[n]
                
                product=1
                
                for i,x in enumerate(row):
                    try:
                        a=math.pow(x,class_parameters['alpha'][i]-1)                         
                    except:
                        a=1
                    
                    try:
                        b=math.pow(1-x,class_parameters['beta'][i]-1)
                    except:
                        b=1
                    
                    
                    c=class_parameters['integral'][i]
                    
                    product*=c*a*b
                    
                    if product==0:
                        break 
                        
                probability=product*class_parameters['frequency']
                
                if probability>_max:
                    _max=probability
                    _max_class=n
            
            output.append(_max_class)
            indexes.append(row_i)
        
        return pd.Series(data=output,index=indexes) 
     
    """
    def __param_estimation(self):
        
        mean=self.train_X.mean(axis=0)
        var=self.train_X.var(axis=0)
        
        k=( ( mean*( 1-mean ) ) / var )-1
        
        self.alpha=k*mean
        self.beta=k*(1-mean)
        
        self.integrals_pixels=[]
            
        for item in zip(self.alpha, self.beta):
            self.integrals_pixels.append(quad(BetaDistribution_NaiveBayes.__B,0,1,args=(item[0],item[1]))[0])
        
        self.frequencies=[]
        
        for n in range(10):
            self.frequencies.append(self.train_y[self.train_y["class"]==n].size/self.train_y["class"].size)
    
    def predict(self,test_X:pd.DataFrame):
        
        output=[]
        
        for _,row in test_X.iterrows():
                
            product=1
            
            for i,x in enumerate(row):
    
                try:
                    product*=self.integrals_pixels[i]*math.pow(x,self.alpha[i]-1)*math.pow(1-x,self.beta[i]-1)
                except:
                    product*=1
            
                _max=0
                _index=None
                for n in range(10):
                    if _max<(product*self.frequencies[n]):
                        _max=(product*self.frequencies[n])
                        _index=n
                
            output.append(_index)
            
        return output                   
    """
        