import sys
import os
from src.logger import logging
from src.exception import CustomException
import pickle
import dill

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
    