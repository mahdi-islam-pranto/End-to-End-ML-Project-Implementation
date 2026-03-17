from src.pipeline.prediction_pipeline import PredictPipeline, TransformFeatureData
from fastapi import FastAPI
from pydantic import BaseModel

# class for the input data/features for prediction
class InputData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

app = FastAPI(
    title="Student Math Performance Prediction API",
    description="API for predicting student math performance based on various features",
    version="1.0.0"
)

@app.post("/predict")
def predict(input_data: InputData):
    
    try:
        # Initialize the TransformFeatureData class with the input data
        transform_feature_data = TransformFeatureData(
            gender=input_data.gender,
            race_ethnicity= input_data.race_ethnicity,
            parental_level_of_education= input_data.parental_level_of_education,
            lunch= input_data.lunch,
            test_preparation_course= input_data.test_preparation_course,
            reading_score= input_data.reading_score,
            writing_score= input_data.writing_score
        )
        
        # transform the input data to a dataframe
        input_dataframe = transform_feature_data.transform_data_to_dataframe()
        
        # predicting the math score using the PredictPipeline class
        predict_math_score = PredictPipeline()
        
        predicted_math_score = predict_math_score.predict(input_dataframe)
        
        return {
            "status": "success",
            "predicted_math_score": predicted_math_score[0]}
    
    
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
    


