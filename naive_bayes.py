import pandas as pd
from sklearn.base import BaseEstimator
from scipy.integrate import quad

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
                integrals_pixels_class_n.append(quad(BetaDistribution_NaiveBayes.__B,0,1,args=(item[0],item[1]))[0])
                
            frequency=self.train_y[self.train_y["class"]==n].size/self.train_y["class"].size
                
            self.parameter_per_class[n]={'alpha':alphas_pixels_class_n,'beta':betas_pixels_class_n,'integral':integrals_pixels_class_n,'frequency':frequency}
            
    def predict(self,test_X:pd.DataFrame):
        
        output=[]
        
        for _,row in test_X.iterrows():
                
            _max=0
            _max_class=None
            
            for n in range(10):
                
                class_parameters=self.parameter_per_class[n]
                
                product=1
                
                for i,x in enumerate(row):
                    try:                  
                        product*=class_parameters['integral'][i]*math.pow(x,class_parameters['alpha'][i]-1)*math.pow(1-x,class_parameters['beta'][i]-1)        
                    except:
                        product=0
                        break
                    
                probability=product*class_parameters['frequency']
                
                if probability>_max:
                    _max=probability
                    _max_class=n
            
            output.append(_max_class)
        
        return pd.Series(output) 
                    
        
        
        