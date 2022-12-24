
from typing import Tuple
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

import pandas as pd
from sklearn.base import BaseEstimator

import pickle

from IPython.display import clear_output

def model_selector( model:BaseEstimator, properties:dict, X:pd.DataFrame, Y:pd.DataFrame, n_jobs:int = 1,y_as_numpy:bool = True ) ->Tuple[BaseEstimator,pd.DataFrame,float] :
    
    clf=GridSearchCV(model,properties,scoring="accuracy",cv=10,verbose=5,n_jobs=n_jobs)
    
    if y_as_numpy:
        clf.fit(X,Y.values.ravel())
    else:
        clf.fit(X,Y)
    
    clear_output(wait=True)
    
    result=pd.DataFrame(clf.cv_results_)
    
    print ("Best Score: ", clf.best_score_)
    print ("Best Params: ", clf.best_params_)
    
    return clf.best_estimator_,result

def save_model_to_file(model:BaseEstimator,model_filename:str):
    
    pickle.dump(model, open('models/'+model_filename, 'wb'))

    
def save_result_to_file(result:pd.DataFrame,result_filename:str):
    
    pickle.dump(result, open('results/'+result_filename, 'wb'))


def read_model_from_file(model_filename:str):
    
    with open( "models/"+model_filename, "rb" ) as f:
        model = pickle.load(f)
        
    return model


def read_result_from_file(result_filename:str):
    
    with open( "results/"+result_filename, "rb" ) as f:
        result = pickle.load(f)
        
    return result

def load(model_name:str)->Tuple[BaseEstimator,pd.DataFrame]:
    
    model_filename = model_name+'.sav'
    result_filename = 'result_'+model_name+'.sav'
        
    model=read_model_from_file(model_filename)
    result=read_result_from_file(result_filename)
        
    return model,result

def save(model:BaseEstimator,result:pd.DataFrame,model_name:str):
    
    model_filename = model_name+'.sav'
    result_filename = 'result_'+model_name+'.sav'
    
    save_model_to_file(model,model_filename)
    save_result_to_file(result,result_filename)


def plot_confusion_matrix(test_y:pd.DataFrame, pred_y:pd.DataFrame):
    cm = confusion_matrix(test_y, pred_y, labels=[x for x in range(10)])
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[x for x in range(10)]).plot()