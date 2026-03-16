import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
# all models
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


# This class is responsible for storing the configuration of the model trainer component, where we will specify the path to store the trained model (pickle file)
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_confiq = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        '''This function is responsible for training the model. It will return the r2 score and model.pkl of the trained model on the test data'''
        
        try:
            logging.info("Entered the model trainer method")
            
            # splitting the train and test data into X and y
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
        
            logging.info("Splitting the train and test data into X and y is completed")
            
            # defining the models to train
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            # evaluating the models and getting the report of the models
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)
            
            logging.info(f"All Model report: {model_report}")
            
            # selecting the best model based on the r2 score
            best_model_score = max(sorted(model_report.values()))
            # best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            logging.info(f"Best model: {best_model_name}, Score: {best_model_score}")
            
            # getting the best model object
            best_model = models[best_model_name]
            
            # checking if the best model score is greater than 0.6, if not then we will raise an exception
            if best_model_score < 0.6:
                logging.info("No best model found with score greater than 0.6")
                raise CustomException("No best model found with score greater than 0.6")
            
            # saving the best model to the specified path
            save_object(
                file_path=self.model_trainer_confiq.trained_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Best model saved successfully to the specified path")
            
            # predicting the test data using the best model
            predicted_data = best_model.predict(X_test)
            
            # calculating the r2 score for the best model on the test data
            best_model_r2_score = r2_score(y_test, predicted_data)
            logging.info(f"Best model R2 score on test data: {best_model_r2_score}")
            
            return best_model_r2_score
        
        

        except Exception as e:
            logging.info("Exception occurred at model training stage")
            raise CustomException(e, sys)