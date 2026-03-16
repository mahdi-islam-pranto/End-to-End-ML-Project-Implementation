import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.utils import save_object

# class for data transformation configuration, where we will specify the path to store the preprocessor object (ml model object after training)
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def define_data_transformation(self):
        '''This function is responsible for data transformation. It will return the preprocessor object which will be used for transforming the data'''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                ]
            
            # pipeline for numerical columns -> imputation(fill null values) and scaling
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            # pipeline for categorical columns -> imputation(fill null values), encoding and scaling
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            # combining the numerical and categorical pipeline using column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
           
        except Exception as e:
            logging.info("Exception occurred at define data transformation stage")
            raise CustomException(e, sys)
        
    def make_data_transformation(self, train_data_path, test_data_path):
        '''This function is responsible for performing data transformation on the train and test data and saving the preprocessor object to the specified path'''
        try:
            # reading the train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Read train and test data completed")
            
            # getting the preprocessor object (data transformation object/pipeline)
            preprocessor = self.define_data_transformation()
            
            # separating the input features and target feature from the train and test data
            input_feature_train_df = train_df.drop(['math_score'], axis=1)
            target_feature_train_df = train_df['math_score']
            
            input_feature_test_df = test_df.drop(['math_score'], axis=1)
            target_feature_test_df = test_df['math_score']
            
            # performing data transformation on the input features of train and test data
            input_feature_train_df_transformed = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_df_transformed = preprocessor.transform(input_feature_test_df)
            
            logging.info("Applying the preprocessing object/pipeline on the training and testing data completed")
            
            # concatenating the transformed input features and target feature for train and test data to get the final train and test array which will be used for model training and evaluation
            train_array = np.c_[input_feature_train_df_transformed, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_df_transformed, np.array(target_feature_test_df)]
            
            # save the preprocessor object to the specified path
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor)
            
            logging.info("Saved the preprocessor object to the specified path")
            
            # save train and test array to the specified path
            np.save(os.path.join('artifacts', 'train_array.npy'), train_array)
            np.save(os.path.join('artifacts', 'test_array.npy'), test_array)
            logging.info("Data transformation is completed")
            
            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception occurred at make data transformation stage")
            raise CustomException(e, sys)
