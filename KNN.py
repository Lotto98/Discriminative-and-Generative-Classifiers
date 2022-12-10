import numpy as np
import statistics as st
from tqdm.notebook import tqdm
from typing import Union

class KNN:
    
    def __init__(self,k:Union[int,list[int]]=5):
        
        if isinstance(k,int):
            self.k=[k]
        else:
            self.k=sorted(k, reverse=False)
            
            prev=None
            for x in range(len(self.k)):
                temp=self.k[x]
                
                if prev is not None:
                    self.k[x]-=prev
                           
                prev=temp
                    
    def fit(self, X_data,y_data):
        self.X_train=X_data
        self.y_train=y_data
    
    def __classify(self,point):
        
        distances=np.linalg.norm(self.X_train - point, axis=1)
        
        outputs=[]
        neighbors_y=np.array([])
        
        for k in self.k:
            
            for _ in range(k):
                
                min_index=np.nanargmin(distances)
                
                neighbors_y=np.append(neighbors_y,self.y_train[min_index])
                
                distances[min_index]=np.nan
    
            outputs.append(st.mode(neighbors_y))
            
        return outputs
    
    def predict(self,data):
        
        outputs=[np.array([]) for _ in range(len(self.k))]
        
        for row in tqdm(data,desc='points'):
        
            for i,k in enumerate(self.__classify(row)):
                outputs[i]=np.append(outputs[i],k)
            
        return outputs         