import os,sys
from dataclasses import dataclass

from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('Artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_path = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting training data and testing input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test = X_test,y_test = y_test,models=models)

            #to get the best model score
            best_model_score = max(sorted(model_report.values()))

            #to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            if best_model_score <0.6:
                raise CustomException('No best model found')
            logging.info('Best found model on both taining and testing dataset')

            save_object(
                file_path=self.model_trainer_path.trained_model_path,
                obj = best_model
            )

            predcited = best_model.predict(X_test)
            r2_square = r2_score(y_test,predcited)

            return r2_square



        except Exception as e:
            logging.info('Error occured in initiate model trainer')
            raise CustomException(e,sys)