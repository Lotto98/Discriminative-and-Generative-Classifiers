import numpy as np
import pandas as pd
import statistics as st

from sklearn.base import BaseEstimator

class KNN(BaseEstimator):
    
    def __init__(self,k:int=5):
        self.k=k
                    
    def fit(self, X_data:pd.DataFrame,y_data:pd.DataFrame | np.ndarray):
        self.X_train=X_data.to_numpy(copy=True)
        if isinstance(y_data,pd.DataFrame):
            self.y_train=y_data.to_numpy(copy=True)
        else:
            self.y_train=np.copy(y_data)
    
    def predict(self,test_X:pd.DataFrame):
        
        test_X=test_X.to_numpy(copy=True)
        
        output=[]
        indexes=[]
        
        for row_i,row in enumerate( (test_X) ):
            
            distances=np.linalg.norm(self.X_train - row, axis=1)
            
            mode=st.mode( ( self.y_train[ np.argsort(distances) ][:self.k] ).flatten()  )
            
            output.append(mode)
            indexes.append(row_i)    
            
        return pd.Series(data=output,index=indexes)      