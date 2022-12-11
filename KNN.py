import numpy as np
import statistics as st
from tqdm.notebook import tqdm

from sklearn.base import BaseEstimator

class KNN(BaseEstimator):
    
    def __init__(self,k:int=5):
        self.k=k
                    
    def fit(self, X_data,y_data):
        self.X_train=X_data
        self.y_train=y_data
    
    def __classify(self,point):
        
        distances=np.linalg.norm(self.X_train - point, axis=1)
        

        neighbors_y=np.array([])

            
        for _ in range(self.k):
            
            min_index=np.nanargmin(distances)
            
            neighbors_y=np.append(neighbors_y,self.y_train[min_index])
            
            distances[min_index]=np.nan

        return st.mode(neighbors_y)
    
    def predict(self,data):
        
        output=np.array([])
        
        for i,row in enumerate(data):
            output=np.append(output,self.__classify(row))
            print(i,'/',1000)
            
        return output        