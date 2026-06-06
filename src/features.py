"""
Feature engineering utilities for the NASA C-MAPSS dataset.

This module contains functions for:
1. Detecting constant columns.
2. Selecting model input features.
3. Separating features and target.
"""

import pandas as pd

def get_constant_columns(data: pd.DataFrame) -> list[str]:
    """
    Identify columns that have only one unique value.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        list[str]: List of constant column names.
    """
    unique_value_counts = data.nunique()
    constant_columns = unique_value_counts[unique_value_counts == 1].index.tolist()

    return constant_columns

def select_feature_columns(
    data: pd.DataFrame,
    target_column: str = 'rul',
) -> list[str]:
    """
    Select useful feature columns for model training.

    This removes:
    1. Constant columns
    2. Engine ID column
    3. Cycle/time column
    4. Target column

    Args:
        data (pd.DataFrame): Input dataset.
        target_column (str): Name of the target column.

    Returns:
        list[str]: Selected feature column names.
    """
    
    constant_columns=get_constant_columns(data)
    
    columns_to_exclude = constant_columns + [
        'unit_number',
        'time_in_cycles',
        target_column
    ]
    
    feature_columns = [
        cols for cols in data.columns
        if cols not in columns_to_exclude
        ]
    
    return feature_columns

def split_features_and_target(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_columns: str = 'rul',
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features and target variable.

    Args:
        data (pd.DataFrame): Dataset containing features and target.
        feature_columns (list[str]): Columns to use as model inputs.
        target_column (str): Name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """
    missing_features = [
        cols for cols in feature_columns
        if cols not in data.columns
    ]
    
    if missing_features:
        raise ValueError(
            f"The following feature columns is missing from the data:"
            f"{missing_features}"
        )
        
    if target_columns not in data.columns:
        raise ValueError(
            f'Target column: {target_columns} missing from the dataset '
        )
        
    X = data[feature_columns]
    y = data[target_columns]
    return X, y