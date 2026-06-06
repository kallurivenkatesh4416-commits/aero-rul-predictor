"""
FastAPI application for aircraft engine RUL prediction.

This API exposes the trained ML model through HTTP endpoints.
Users can send engine sensor readings and receive:
1. Predicted Remaining Useful Life
2. Risk category
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.inference import predict_rul


app = FastAPI(
    title = 'Aircraft engine RUL prediction API',
    description = (
        "API for predicting Remaining Useful Life of aircraft turbofan engines "
        "using NASA C-MAPSS sensor data."
    ),
    version = '1.0.0',
)


class EngineSensorInput(BaseModel):
    """
    Input schema for one engine sensor reading.

    These fields match the 17 features used by the trained model.
    """
    
    operational_setting_1: float = Field(..., example=0.0023)
    operational_setting_2: float = Field(..., example=0.0003)

    sensor_2: float = Field(..., example=642.15)
    sensor_3: float = Field(..., example=1589.70)
    sensor_4: float = Field(..., example=1400.60)
    sensor_6: float = Field(..., example=21.61)
    sensor_7: float = Field(..., example=554.36)
    sensor_8: float = Field(..., example=2388.06)
    sensor_9: float = Field(..., example=9046.19)
    sensor_11: float = Field(..., example=47.47)
    sensor_12: float = Field(..., example=521.66)
    sensor_13: float = Field(..., example=2388.02)
    sensor_14: float = Field(..., example=8138.62)
    sensor_15: float = Field(..., example=8.4195)
    sensor_17: float = Field(..., example=392.0)
    sensor_20: float = Field(..., example=39.06)
    sensor_21: float = Field(..., example=23.4190)
    
    
class PredictionResponse(BaseModel):
    """
    Response schema returned by the prediction API
    """
    
    predicted_rul: float
    risk_category: str
    
@app.get("/")
def health_check() -> dict[str,str]:
    """
    Health check endpoint.

    Returns:
        dict[str, str]: API status message.
    """
    return {
        'status': 'ok',
        'message': "Aircraft engine RUL prediction API is running",
    
    }
    
    
@app.post('/predict', response_model = PredictionResponse)
def predict_engine_rul(engine_data: EngineSensorInput) -> PredictionResponse:
    """
    Predict Remaining Useful Life for one engine sensor reading.

    Args:
        engine_data (EngineSensorInput): Engine operational settings and sensor values.

    Returns:
        PredictionResponse: Predicted RUL and risk category.

    Raises:
        HTTPException: If prediction fails.
    """
    
    try:
        input_df = pd.DataFrame([engine_data.model_dump()])
        
        prediction_result = predict_rul(input_df)
        
        risk_category = str(prediction_result.loc[0, 'risk_category'])
        predicted_rul = float(prediction_result.loc[0,'predicted_rul'])
        
        return PredictionResponse(
            predicted_rul = round(predicted_rul, 2),
            risk_category = risk_category,
        )
        
    except Exception as error:
        raise HTTPException(
            status_code = 400,
            detail = f'Prediction failed: {str(error)}',
        ) from error