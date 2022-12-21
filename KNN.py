import numpy as np
import pandas as pd
import statistics as st

from sklearn.base import BaseEstimator

class KNN(BaseEstimator):
    
    def __init__( self, k:int = 5 ) -> None:
        self.k=k
                    
    def fit( self, X_train:pd.DataFrame, y_train:pd.DataFrame | np.ndarray ):
        self.X_train=X_train
        if isinstance(y_train,pd.DataFrame):
            self.y_train=y_train.to_numpy()
        else:
            self.y_train=y_train
        
        return self
    
    def predict( self, test_X:pd.DataFrame ) -> pd.Series:
        
        test_X=test_X.to_numpy()
        
        output=[]
        indexes=[]
        
        for row_i,row in enumerate( (test_X) ):
            
            distances=np.linalg.norm(self.X_train - row, axis=1)
            
            mode=st.mode( ( self.y_train[ np.argsort(distances) ][:self.k] ).flatten()  )
            
            output.append(mode)
            indexes.append(row_i)    
            
        return pd.Series(data=output,index=indexes)      