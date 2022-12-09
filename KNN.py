from cmath import sqrt
import numpy as np
import statistics as st
from tqdm.notebook import tqdm,tnrange

class KNN:
    
    def __init__(self,k=1):
        self.k=k        
        
    def fit(self, X_data,y_data):
        self.X_train=X_data
        self.y_train=y_data
        
    def get_params(self,deep):
        
        return {'k':self.k}
    
    def set_params(self,**params):
        
        for key,value in params.items():
            setattr(self, key, value)
        return self
    
    def __classify(self,point):
        distances=np.linalg.norm(self.X_train - point, axis=1)
        
        neighbors_y=np.array([])
        for _ in range(self.k):
            
            max_index=np.nanargmax(distances)
            
            neighbors_y=np.append(neighbors_y,self.y_train[max_index])
            
            distances[max_index]=np.nan
        
        return st.mode(neighbors_y)
    
    def predict(self,data):
        output_y=np.array([])
        
        for row in data:
            output_y=np.append(output_y,self.__classify(row))
    
        return output_y            