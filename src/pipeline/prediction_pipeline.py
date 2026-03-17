import pandas as pd
import sys
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

# This class is responsible for making predictions using the trained model. It will load the model and preprocessor objects from the specified path, transform the input features using the preprocessor object and then make predictions using the model object. It will return the predicted values.
class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            # loading the model and preprocessor objects from the specified path
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Model and preprocessor objects are loaded for prediction")
            
            # transforming the input features using the preprocessor object
            scaled_input_features = preprocessor.transform(features)
            logging.info("Input features are transformed using the preprocessor object")
            
            # making predictions using the model object
            predicted_values = model.predict(scaled_input_features)
            logging.info("Predictions are made using the model object")
            
            return predicted_values
        
        except Exception as e:
            logging.info("Exception occurred in prediction pipeline")
            raise CustomException(e, sys)
        
        
        
# This class is responsible for transforming the input data to a dataframe, which will be used for prediction. It will take the input data as arguments and return the dataframe.
class TransformFeatureData:
    def __init__(self,
                 gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
                 ):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def transform_data_to_dataframe(self):
        '''This function is responsible for transforming the input data to a dataframe, which will be used for prediction'''
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            
            }
            
            input_df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe is created from the input data")
            
            return input_df
        
        except Exception as e:
            logging.info("Exception occurred in transform_data_to_dataframe method of TransformFeatureData class")
            raise CustomException(e, sys)
        
        
# test the prediction pipeline
# if __name__ == "__main__":
#     try:
#         # creating an object of the TransformFeatureData class with the input data
#         transform_feature_data = TransformFeatureData(
#             gender= 'male',
#             race_ethnicity= "group B",
#             parental_level_of_education= "some college",
#             lunch= "standard",
#             test_preparation_course= "none",
#             reading_score= 72,
#             writing_score= 74
#         )
#         input_dataframe = transform_feature_data.transform_data_to_dataframe()
#         # creating an object of the PredictPipeline class and making predictions
#         predict_pipeline = PredictPipeline()
#         predicted_values = predict_pipeline.predict(input_dataframe)
        
#         print(f"Predicted math score: {predicted_values[0]}")
#         logging.info(f"Predicted values: {predicted_values}")
#     except Exception as e:
#         logging.info("Exception occurred while testing the prediction pipeline")
#         raise CustomException(e, sys)