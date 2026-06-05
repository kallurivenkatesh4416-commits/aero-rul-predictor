"""
Data loading utilities for the NASA C-MAPSS dataset.

This file will help us:
1. Define column names.
2. Load raw training data.
3. Load raw test data.
4. Create the RUL target.
"""

from pathlib import Path
import pandas as pd
from src.config import MAX_RUL, RUL_FILE_PATH, TEST_FILE_PATH, TRAIN_FILE_PATH

"""
Data loading utilities for the NASA C-MAPSS dataset.
...
"""



def get_cmapss_column_names() -> list[str]:
    """
    Return standard column names for the NASA C-MAPSS dataset.

    Each row contains:
    - engine/unit id
    - cycle number
    - 3 operational settings
    - 21 sensor readings

    Returns:
        list[str]: List of column names.
    """
    
    index_columns = ['unit_number', 'time_in_cycles']
    
    operational_setting_columns = [
        'operational_setting_1',
        'operational_setting_2',
        'operational_setting_3',
    ]
    
    sensor_columns = [f'sensor_{sensor_num}' for sensor_num in range(1,22)]
    
    return index_columns + operational_setting_columns + sensor_columns

def validate_file_exists(file_path: Path) -> None:
    """
    Validate whether a required file exists.

    Args:
        file_path (Path): Path to the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}."
            "Please check whether the NASA CMAPSS data is downloaded"
            "and extracted correctly"
        )
        
def load_cmapss_file(file_path: Path) -> pd.DataFrame:
    """
    Load a C-MAPSS text file into a pandas DataFrame.

    The raw NASA files are whitespace-separated text files without headers.

    Args:
        file_path (Path): Path to the C-MAPSS text file.

    Returns:
        pd.DataFrame: Loaded dataset with proper column names.
    """
    validate_file_exists(file_path)
    
    column_names = get_cmapss_column_names()
    
    data = pd.read_csv(
        file_path,
        sep = r"\s+",
        header = None,
        names = column_names,
    )
    return data




def load_train_data() -> pd.DataFrame:
    """
    Load FD001 training data.

    Returns:
        pd.DataFrame: Raw training data.
    """
    return load_cmapss_file(TRAIN_FILE_PATH)




def load_test_data() -> pd.DataFrame:
    """
    Load FD001 test data.

    Returns:
        pd.DataFrame: Raw test data.
    """
    return load_cmapss_file(TEST_FILE_PATH)



def load_test_rul() -> pd.DataFrame:
    """
    Load true Remaining Useful Life values for FD001 test engines.

    The RUL file has one value per test engine.

    Returns:
        pd.DataFrame: Test RUL values with unit numbers.
    """
    
    validate_file_exists(RUL_FILE_PATH)
    
    rul_data = pd.read_csv(
        RUL_FILE_PATH,
        sep = r"\s+",
        header = None,
        names = ['true_rul'],
    )
    
    rul_data['unit_number'] = range(1, len(rul_data) + 1)
    
    return rul_data[['unit_number', 'true_rul']]


def add_rul_target(data: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """
    Add Remaining Useful Life target to the training data.

    For each engine:
        RUL = final cycle of that engine - current cycle

    Args:
        data (pd.DataFrame): Training data.
        max_rul (int): Maximum RUL value after capping.

    Returns:
        pd.DataFrame: Training data with a new 'rul' column.
    """
    required_columns = {'unit_number', 'time_in_cycles'}
    
    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Input data must contain 'unit_number' and 'time_in_cycles' columns."
        )
        
    data_with_rul = data.copy()
    
    max_cycle_per_unit = data_with_rul.groupby('unit_number')[
        'time_in_cycles'
    ].transform('max')
    
    data_with_rul['rul'] = max_cycle_per_unit - data_with_rul['time_in_cycles']
    data_with_rul['rul'] = data_with_rul['rul'].clip(upper = max_rul)
    
    return data_with_rul



