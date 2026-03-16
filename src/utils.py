import sys
import os
from src.logger import logging
from src.exception import CustomException
import pickle
import dill
from sklearn.metrics import r2_score

# utility function to save the preprocessor object (ml model object after training -> pickle file) to the specified path
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.info("Exception occurred while saving object")
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    
    try:
        report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            
            # Train model
            model.fit(X_train, y_train)
            # predicting train and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # calculating r2 score for train and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
            
    except Exception as e:
        logging.info("Exception occurred while evaluating models")
        raise CustomException(e, sys)
    