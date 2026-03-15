import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.utils import save_object

# class for data ingestion configuration, where we will specify the path to store the train, test and raw data
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
# class for data ingestion, where we will read the data from the source, split it into train and test and then save it to the specified path
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method for collecting the data")
        try:
            # reading the data from the source
            df = pd.read_csv('notebooks\data\stud.csv')
            logging.info("Read the dataset as dataframe")
            
            # making the directory to store the data if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            # saving the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split part initiated")
            # splitting the data into train and test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # saving the train and test data to the specified path
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data and train test split is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            
            
        except Exception as e:
            logging.info("Exception occurred at data ingestion stage")
            raise CustomException(e, sys)
        

# test the data ingestion component and transformation component by running this file
if __name__ == "__main__":
    obj = DataIngestion()
    # obj.initiate_data_ingestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    data_transformation_obj = DataTransformation()
    data_transformation_obj.make_data_transformation(
    train_data_path=train_data_path, test_data_path=test_data_path)
    