import os 
import sys 
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score as raccuracy

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        logging.info("Exception Occured")
        raise CustomException(e,sys)
    

def evaluate_model(x_train,y_train,x_test,y_test,models):
    report={}
    for i in range(len(list(models))):
        model= list(models.values())[i]
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        report[list(models.keys())[i]]=raccuracy(y_test,y_pred)

    return report



def calc_model_metrics(true,pred):
    mse=mean_squared_error(true,pred)
    r2=raccuracy(true,pred)
    mae=mean_absolute_error(true,pred)
    rmse=np.sqrt(mse)
    return (mse,rmse,mae,r2)
    
