import os
import sys
#from mlproject.exception import CustomException
#from mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.data_transformation import DataTransformationConfig


@dataclass 
class DataIngestionConfig:
    train_data_path:str = os.path.join('Artifacts','train.csv')
    test_data_path:str = os.path.join('Artifacts','test.csv')
    raw_data_path:str = os.path.join('Artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered into data ingestion componment')
        try:
            df = pd.read_csv('notebooks\data\stud.csv')
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('tran test split initiated')
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=40)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)
            logging.info('Ingestion of the data completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured during data ingeston stage')
            raise CustomException(e,sys)
        

if __name__ =='__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_tranformation = DataTransformation()
    data_tranformation.initiate_data_tranformation(train_data,test_data)
