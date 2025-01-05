import numpy as np 
import pickle
import os 
import sys
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object,evaluate_model
#Model trainer config

@dataclass
class ModelTrainerConfig:
    model_trainer_path=os.path.join("artifacts","model.pkl")



#Model trainer 

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1])
            models={
                'LinearRegression':LinearRegression(),
                'LassoRegression':Lasso(),
                'RidgeRegression':Ridge(),
                'ElasticNet':ElasticNet(),
                'DecisionTree':DecisionTreeRegressor()
            }

            report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            logging.info("===============================REPORT==============================\n")
            print(report)
            logging.info(f'Model Report is {report}')

            best_score=max(sorted(report.values()))
            best_model_name=list(report.keys())[list(report.values()).index(best_score
            )]

            logging.info(f"best model is {best_model_name}")
            save_object(
                self.model_trainer_config.model_trainer_path,models[best_model_name]
            )



        except Exception as e:
            raise CustomException(e,sys)

