import pandas as pd
from sklearn.base import BaseEstimator
from scipy.stats import beta
import numpy as np

class BetaDistribution_NaiveBayes(BaseEstimator):
    
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self,train_X:pd.DataFrame,train_y:pd.DataFrame):
        self.train_X=train_X
        self.train_y=train_y
        self.__param_estimation()
    
    def __param_estimation(self):
        
        self.parameter_per_class={}
        
        for n in range(10):
            
            images_class_n=self.train_X[self.train_y["class"]==n]
            
            #images_class_n=images_class_n.apply(lambda x: x + np.abs(np.random.normal(1,0.0001,1)) )
            
            means_pixels_class_n=images_class_n.mean(axis=0)
            variances_pixels_class_n=images_class_n.var(axis=0)

            unique_counts=images_class_n.nunique(axis=0, dropna=True)
            
            ks_pixels_class_n=((means_pixels_class_n*(1-means_pixels_class_n))/variances_pixels_class_n)-1
            
            alphas_pixels_class_n=ks_pixels_class_n*means_pixels_class_n
            betas_pixels_class_n=ks_pixels_class_n*(1-means_pixels_class_n)
                
            frequency=self.train_y[self.train_y["class"]==n].size/self.train_y["class"].size
                
            self.parameter_per_class[n]={'alpha':alphas_pixels_class_n.to_numpy(),
                                        'beta':betas_pixels_class_n.to_numpy(),
                                        'unique_counts':unique_counts,
                                        'mean':means_pixels_class_n,
                                        'var':variances_pixels_class_n,
                                        'k':ks_pixels_class_n,
                                        'frequency':frequency}
            print(self.parameter_per_class)
            
    def predict(self,test_X:pd.DataFrame):
        
        epsilon=0.1
        
        output=[]
        indexes=[]
        
        for row_i,row in test_X.iterrows():
                
            _max=0
            _max_class=-1
            
            for n in range(10):
                
                class_parameters=self.parameter_per_class[n]
                
                product=1
                
                for i,x in enumerate(row):
                    
                    if class_parameters['unique_counts'][i]!=1:
                        
                        _alpha=class_parameters['alpha'][i]
                        _beta=class_parameters['beta'][i]
                        
                        _mean=class_parameters['mean'][i]
                        _var=class_parameters['var'][i]
                        _k=class_parameters['k'][i]
                        
                            
                        prob=beta.cdf(x+epsilon,_alpha,_beta)-beta.cdf(x-epsilon,_alpha,_beta)
                    
                        if np.isnan(prob) or prob==float("inf") or prob==float("-inf"):
                            """
                            print(prob)
                            print(n,i)
                            print(class_parameters['unique_counts'][i])
                            print(x,_alpha,_beta,_mean,_var,_k) 
                            """
                            pass
                        else:
                            #print(prob) 
                            product*=prob 
                    else:
                        if x==class_parameters['mean'][i]:
                            product*=1
                        else:
                            product*=0
                        
                    if product==0:
                        break
                    
                probability=product*class_parameters['frequency']
                
                if probability>_max:
                    _max=probability
                    _max_class=n
                    #print(_max)
            
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
        