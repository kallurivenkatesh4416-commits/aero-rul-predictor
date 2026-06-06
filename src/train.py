"""
Training pipeline for the aircraft engine RUL prediction model.

This script:
1. Loads NASA C-MAPSS training data.
2. Creates the RUL target.
3. Selects useful feature columns.
4. Splits data by engine ID to avoid data leakage.
5. Trains the final tuned XGBoost model.
6. Saves the model artifact for inference.
"""

import joblib
import pandas as pd
import numpy as np


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import MODEL_DIR, MODEL_PATH, RANDOM_STATE, XGBOOST_PARAMS
from src.data_processing import add_rul_target, load_train_data
from src.features import select_feature_columns, split_features_and_target

def split_data_by_engine(
    data: pd.DataFrame,
    test_size: float = 0.2,
    
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets by engine ID.

    This prevents data leakage because rows from the same engine
    will not appear in both training and validation sets.

    Args:
        data (pd.DataFrame): Training dataset with unit_number column.
        test_size (float): Fraction of engines to use for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train data and validation data.
    """
    
    if "unit_number" not in data.columns:
        raise ValueError('Input data must have a unit_number column')
    
    unique_engine_ids = data['unit_number'].unique()
    
    train_engine_ids, validation_engine_ids = train_test_split(
        unique_engine_ids,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    
    train_data = data[data['unit_number'].isin(train_engine_ids)].copy()
    validation_data = data[data['unit_number'].isin(validation_engine_ids)].copy()
    
    return train_data, validation_data

def train_xgboost_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> XGBRegressor:
    """
    Train the final tuned XGBoost model.

    These hyperparameters were selected from the experimentation notebook
    using GroupKFold cross-validation to avoid engine-level data leakage.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target values.

    Returns:
        XGBRegressor: Trained XGBoost regression model.
    """
    
    model = XGBRegressor(**XGBOOST_PARAMS)
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(
    model: XGBRegressor,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    
) -> dict[str, float]:
    """
    Evaluate the trained model on validation data.

    Args:
        model (XGBRegressor): Trained XGBoost model.
        X_validation (pd.DataFrame): Validation feature matrix.
        y_validation (pd.Series): True validation target values.

    Returns:
        dict[str, float]: Dictionary containing MAE, RMSE, and R2 score.
    """
    
    predictions = model.predict(X_validation)
    mae = mean_absolute_error(y_validation, predictions)
    rmse = np.sqrt(mean_squared_error(y_validation, predictions))
    r2 = r2_score(y_validation, predictions)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2 score': r2,
    }
    
def save_model_artifact(
    model: XGBRegressor,
    feature_columns: list[str],
    validation_metrics: dict[str, float],
)-> None:
    """
    Save the trained model and important metadata.

    Args:
        model (XGBRegressor): Trained model.
        feature_columns (list[str]): Feature columns used during training.
        validation_metrics (dict[str, float]): Validation performance metrics.
    """
    
    MODEL_DIR.mkdir(exist_ok = True)
    
    model_artifact = {
        'model': model,
        'feature_columns': feature_columns,
        'model_name': 'Tuned XGBoost',
        'validation_metrics': validation_metrics,
    }
    
    joblib.dump(model_artifact, MODEL_PATH)
    
    print(f'Model saved in path: {MODEL_PATH}')


def run_training_pipeline() -> None:
    """
    Run the full model training pipeline.

    Steps:
    1. Load raw training data.
    2. Create RUL target.
    3. Split data by engine ID.
    4. Select feature columns.
    5. Train final XGBoost model.
    6. Evaluate on validation engines.
    7. Save model artifact.
    """
    
    print('Loading training data............')
    raw_train_data = load_train_data()
    
    print('Creating RUL target...............')
    train_data_with_rul = add_rul_target(raw_train_data)
    
    print('Splitting data by engine ID...........')
    train_data, validation_data = split_data_by_engine(train_data_with_rul)
    
    print('Selecting feature columns..............')
    feature_columns = select_feature_columns(train_data)
    
    X_train, y_train = split_features_and_target(
        train_data,
        feature_columns
    )
    
    X_validation, y_validation = split_features_and_target(
        validation_data,
        feature_columns,
    )
    
    print('Training XGBoost model................')
    model = train_xgboost_model(X_train, y_train)
    
    
    print('Evaluating model on validation data.............')
    validation_metrics = evaluate_model(
        model, 
        X_validation, 
        y_validation,
    )
    
    print('Validation metrics')
    for metric_name, metric_value in validation_metrics.items():
        print(f'{metric_name} : {metric_value:.4f}')
        
    print('Saving model artifact')
    save_model_artifact(
        model,
        feature_columns,
        validation_metrics,
    )
    
    print('Training pipeline completed successfully')
    
    
if __name__ == "__main__":
    run_training_pipeline()

