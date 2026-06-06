"""
Inference utilities for the aircraft engine RUL prediction model.

This module:
1. Loads the trained model artifact.
2. Validates incoming feature data.
3. Generates Remaining Useful Life predictions.
4. Converts predicted RUL into business risk categories.
"""

import joblib
import pandas as pd
from src.config import MODEL_PATH

def load_model_artifact() -> dict:
    """
    Load the trained model artifact from disk.

    Returns:
        dict: Saved model artifact containing the trained model,
                feature columns, model name, and validation metrics.

    Raises:
        FileNotFoundError: If the model artifact file does not exist.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Model artifact not found at: {MODEL_PATH}'
            "please run the training pipeline first using: python -m src.train"
        )
        
    model_artifact = joblib.load(MODEL_PATH)
    return model_artifact


def validate_input_features(
    input_data: pd.DataFrame,
    feature_columns: list[str],
    
) -> None:
    
    """"
    Validate whether incoming data contains all required feature columns.

    Args:
        input_data (pd.DataFrame): Input data for prediction.
        feature_columns (list[str]): Feature columns expected by the model.

    Raises:
        ValueError: If any required feature column is missing.
    """
    
    missing_columns = [
        cols for cols in feature_columns
        if cols not in input_data.columns
    ]
    
    if missing_columns:
        raise ValueError(
            f'Input data is missing required feature column: {missing_columns}'
        )
        
        


def assign_risk_category(
    predicted_rul: float
) -> str:
    """
    Convert predicted RUL into a business-friendly risk category.

    Args:
        predicted_rul (float): Predicted remaining useful life.

    Returns:
        str: Risk category.
    """
    
    if predicted_rul <= 30:
        return 'Critical'
    
    if predicted_rul <= 70:
        return 'Warning'
    
    return 'Healthy'


def predict_rul(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict Remaining Useful Life for input engine sensor data.

    Args:
        input_data (pd.DataFrame): Input data containing required feature columns.

    Returns:
        pd.DataFrame: Prediction results with predicted RUL and risk category.
    """
    
    model_artifact = load_model_artifact()
    
    model = model_artifact['model']
    feature_columns = model_artifact['feature_columns']
    
    validate_input_features(
        input_data = input_data,
        feature_columns=feature_columns,
    )
    
    prediction_input = input_data[feature_columns]
    
    predicted_rul_values = model.predict(prediction_input)
    
    results = input_data.copy()
    results['predicted_rul'] = predicted_rul_values
    results['risk_category'] = results['predicted_rul'].apply(assign_risk_category)
    
    return results
    