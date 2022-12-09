from cmath import sqrt
import numpy as np
import statistics as st
from tqdm.notebook import tqdm

from multiprocessing import Pool
class KNN:
    
    def __init__(self,k,n_jobs):
        self.k=k
        self.n_jobs=n_jobs        
        
    def fit(self, X_data,y_data):
        self.X_train=X_data
        self.y_train=y_data
    
    def classify(self,point):
        
        distances=np.array([])
        for row in self.X_train:
            distances=np.append(distances,np.linalg.norm(point-row))
        
        neighbors_y=np.array([])
        for _ in range(self.k):
            
            max_index=np.nanargmax(distances)
            
            neighbors_y=np.append(neighbors_y,self.y_train[max_index])
            
            distances[max_index]=np.nan
        
        return st.mode(neighbors_y)
    
    def predict(self,data):
        
        pool=Pool()
        results=[]
        _from=0
        _step=int(len(data)/self.n_jobs)
        
        while _from<len(data):
            results+=[pool.apply_async(f,(self,data,_from,_from+_step,))]
            _from+=_step
        
        for x in (results):
            print(x.get(timeout=3600))

def f(knn,data,_from,_to):
    
        output_y=np.array([])
        
        for x in range(_from,_to):
            output_y=np.append(output_y,knn.classify(data[x]))
    
        return output_y