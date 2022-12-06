from cmath import sqrt
import numpy as np
import statistics as st
from tqdm.notebook import tqdm,tnrange

class KNN:
    
    def __init__(self,k):
        self.k=k        
        
    def fit(self, X_data,y_data):
        self.X_train=X_data
        self.y_train=y_data
    
    @staticmethod
    def __euclidean_distance(point1,point2):
        sum=0
        for p1 in point1:
            for p2 in point2:
                sum=sum+(p1-p2)**2
        return sqrt(sum)
    
    def __classify(self,point):
        distances=np.array([])
        for row in self.X_train:
            distances=np.append(distances,KNN.__euclidean_distance(point,row))
        
        neighbors_y=np.array([])
        for _ in range(self.k):
            
            max_index=np.nanargmax(distances)
            
            neighbors_y=np.append(neighbors_y,self.y_train[max_index])
            
            distances[max_index]=np.nan
        
        return st.mode(neighbors_y)
    
    def predict(self,data):
        output_y=np.array([])
        
        for row in tqdm(data,desc="points"):
            output_y=np.append(output_y,self.__classify(row))
    
        return output_y            