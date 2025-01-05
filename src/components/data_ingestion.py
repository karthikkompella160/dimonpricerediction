import pandas as pd
import numpy as np
import os 
import sys
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


# initilize the data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','rawdata.csv')



# create a data ingestion class

class DataIngestion:

    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
    
    def initiate_data_ingestion(self):

        logging.info("Data Ingestion Started ")
        
        try:
            df= pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))

            logging.info("Dataset reading completed into pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Splitting the Dataset for training and testing....")

            trainset,testset=train_test_split(df,test_size=0.25,random_state=22)

            trainset.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            testset.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('data ingestion completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error occured in Data Ingestion")
            raise CustomException(e,sys)
