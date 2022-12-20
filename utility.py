
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

import pandas as pd
from sklearn.base import BaseEstimator

import pickle

from IPython.display import clear_output

def model_selector(model:BaseEstimator,properties:dict,X:pd.DataFrame,Y:pd.DataFrame):
    
    grid=GridSearchCV(model,properties,scoring="accuracy",cv=10,return_train_score=True,verbose=5)
    grid.fit(X,Y.values.ravel())
    
    clear_output(wait=True)
    
    result=pd.DataFrame(grid.cv_results_)
    
    print ("Best Score: ", grid.best_score_)
    print ("Best Params: ", grid.best_params_)
    
    return grid.best_estimator_,result

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

def save_or_load(model:BaseEstimator,result:pd.DataFrame,model_name:str):
    
    model_filename = model_name+'.sav'
    result_filename = 'result_'+model_name+'.sav'
    
    if result is not None:
        save_model_to_file(model,model_filename)
        save_result_to_file(result,result_filename)
    else:
        
        model=read_model_from_file(model_filename)
        result=read_result_from_file(result_filename)
        
    return model,result



def confusion_matrix(test_y:pd.DataFrame, pred_y:pd.DataFrame):
    cm = confusion_matrix(test_y, pred_y, labels=[x for x in range(10)])
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[x for x in range(10)]).plot()