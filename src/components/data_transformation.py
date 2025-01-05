import pandas as pd
import os
import sys
from sklearn.impute import SimpleImputer   ## handling missing values if any 
from sklearn.preprocessing import StandardScaler  ## for scaling down the numerical data
from sklearn.preprocessing import OrdinalEncoder  ## for encoding the ordinal values 
import numpy as np
from sklearn.pipeline import Pipeline     # for building pipelines 
from sklearn.compose import ColumnTransformer    # for combining the pipelines together 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
## Data transformation Config

@dataclass
class DataTransformationConfig:
    prepocessor_ob_file_path=os.path.join("artifacts","preprocessor.pkl")



## Dat Transformation Class 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_datatransformation_object(self):
        # entire pipeline will  be here 
        try:
            logging.info("Data Transformatoion started")

            cat_cols=['clarity', 'cut', 'color']
            num_cols=['carat','depth', 'table', 'x', 'y','z']

            cut_types=[ 'Good', 'Fair','Very Good','Premium', 'Ideal',]
            color_types=[ 'D','E', 'F', 'G', 'H', 'I', 'J']
            clarity_type=[ 'I1', 'SI2', 'SI1', 'VS2','VS1',  'VVS2', 'VVS1','IF']
            num_pipeline=Pipeline(steps=
                [
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[clarity_type,cut_types,color_types])),
                ('scaler',StandardScaler())
                ]
            )
            preprocessor=ColumnTransformer(
                [ 
                    ('numerical_pipeline',num_pipeline,num_cols),
                 ('categorical_pipeline',cat_pipeline,cat_cols)]
                )
            logging.info("Pipeines Created for  Transformation")
            return preprocessor

        except Exception as e:
            logging.info("Exception occured")
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_data_path,test_data_path):

        train_df=pd.read_csv(train_data_path)
        test_df=pd.read_csv(test_data_path)
        logging.info("Training and Testing data reading completed....")
        preprocessor= self.get_datatransformation_object()

        target_column='price'
        drop_columns=['id',target_column]
        # splitting the data into independent and dependent features both train and test data 
        
        input_train_features=train_df.drop(columns=drop_columns,axis=1)
        target_train_feature=train_df[target_column]

        input_test_features=test_df.drop(columns=drop_columns,axis=1)
        target_test_feature=test_df[target_column]

        # fit and transform the input train features and test features 
        input_train_features_arr=preprocessor.fit_transform(input_train_features)
        input_test_features_arr=preprocessor.transform(input_test_features)
        logging.info('preprocessing completed...')
        #now combine the target columns 
        train_arr=np.c_[input_train_features_arr,np.array(target_train_feature)]
        test_arr=np.c_[input_test_features_arr,np.array(target_test_feature)]

        save_object(self.data_transformation_config.prepocessor_ob_file_path,preprocessor)
        logging.info("Preprocessor pickle is created and saved")

        return (train_arr,test_arr,self.data_transformation_config.prepocessor_ob_file_path)









