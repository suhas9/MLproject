import sys
from dataclasses import dataclass
import numpy as np
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
from src.mlproject.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Artifacts','preprocessor.pkl')
    logging.info('preprocessor pickel file created ')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='median')),
                    ('Scaler',StandardScaler(with_mean=False)),
                    

                ]
            )
            logging.info('numerical columns scaling completed')
            cat_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequent')),
                    ('OHE',OneHotEncoder()),
                    ('Scaling',StandardScaler(with_mean=False)),
                ]
            )

            logging.info('Categorical encoding completed')

            logging.info(f"Categorical column:{categorical_columns}")
            logging.info(f"Numerical column:{numerical_columns}")

            preprocessor = ColumnTransformer([
                ('Num_pipeline',num_pipeline,numerical_columns),
                ('Cat_pipelne',cat_pipeline,categorical_columns)]
            )

            return preprocessor


        except Exception as e:
            logging.info("Error has occured in get_data_transformer_obj ")
            raise CustomException(e,sys)
        
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessor object')

            preprocessor_obj = self.get_data_transformer_obj()
            target_columns = 'math_score'
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_columns],axis=1)
            target_feature_train_df = train_df[target_columns]

            input_feature_test_df = test_df.drop(columns=[target_columns],axis=1)
            target_feature_test_df = test_df[target_columns]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(input_feature_train_arr)]
            test_arr = np.c_[input_feature_test_arr,np.array(input_feature_test_arr)]
            logging.info(f'saved preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Error has occured in initiate_data_tranformation ")
            raise CustomException(e,sys)