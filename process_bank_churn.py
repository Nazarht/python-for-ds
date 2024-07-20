import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple, List

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and validation sets, stratified by the target column.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the data to include in the validation set.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training data, validation data,
        training targets, validation targets.
    """
    features = df.drop(columns=target_col)
    targets = df[target_col]
    
    train_data, val_data, train_targets, val_targets = train_test_split(
        features, targets, test_size=test_size, random_state=random_state, stratify=targets
    )
    
    return train_data, val_data, train_targets, val_targets

def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drops unnecessary columns from the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        columns_to_drop (list): List of column names to be dropped.
        
    Returns:
        pd.DataFrame: The dataframe with the specified columns dropped.
    """
    return df.drop(columns=columns_to_drop)

def preprocess_numeric_and_categorical_data(
    df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool, preprocessor: ColumnTransformer = None
) -> Tuple[np.ndarray, List[str], ColumnTransformer]:
    """
    Preprocesses numeric and categorical data using MinMaxScaler and OneHotEncoder.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        numeric_cols (list): List of numeric column names.
        categorical_cols (list): List of categorical column names.
        scale_numeric (bool): Flag indicating whether to scale numeric columns.
        preprocessor (ColumnTransformer, optional): Pre-trained preprocessor for transforming data.
        
    Returns:
        Tuple[np.ndarray, list, ColumnTransformer]: Transformed data, feature names, and preprocessor.
    """
    if preprocessor is None:
        numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())]) if scale_numeric else 'passthrough'
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        transformed_data = preprocessor.fit_transform(df)
    else:
        transformed_data = preprocessor.transform(df)
    
    one_hot_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = numeric_cols + list(one_hot_names)
    
    return transformed_data, all_feature_names, preprocessor

def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], MinMaxScaler, OneHotEncoder]:
    """
    Preprocesses the data, including splitting, dropping unnecessary columns, and scaling/encoding features.
    
    Args:
        raw_df (pd.DataFrame): The raw input dataframe.
        scale_numeric (bool): Flag indicating whether to scale numeric columns.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list, MinMaxScaler, OneHotEncoder]: 
        Processed training inputs, validation inputs, training targets, validation targets, 
        input columns, scaler, and encoder.
    """
    target_col = 'Exited'
    columns_to_drop = ["Surname", "CustomerId"]
    
    # Split the data
    train_data, val_data, train_targets, val_targets = split_data(raw_df, target_col)
    
    # Concatenate targets back to features for dropping unnecessary columns
    train_data_df = pd.concat([train_data, train_targets], axis=1)
    val_data_df = pd.concat([val_data, val_targets], axis=1)
    
    # Drop unnecessary columns
    train_data_df = drop_unnecessary_columns(train_data_df, columns_to_drop)
    val_data_df = drop_unnecessary_columns(val_data_df, columns_to_drop)
    
    # Separate inputs and targets
    input_cols = list(train_data_df.columns)[1:-1]
    train_inputs, train_targets = train_data_df[input_cols], train_data_df[target_col]
    val_inputs, val_targets = val_data_df[input_cols], val_data_df[target_col]
    
    # Identify numeric and categorical columns
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()
    
    # Preprocess training data
    train_inputs_transformed, all_feature_names, preprocessor = preprocess_numeric_and_categorical_data(
        train_inputs, numeric_cols, categorical_cols, scale_numeric
    )
    
    # Preprocess validation data using the pre-trained preprocessor
    val_inputs_transformed, _, _ = preprocess_numeric_and_categorical_data(
        val_inputs, numeric_cols, categorical_cols, scale_numeric, preprocessor
    )
    
    # Extract the scaler and encoder
    scaler = preprocessor.named_transformers_['num']['scaler'] if scale_numeric else None
    encoder = preprocessor.named_transformers_['cat']['onehot']
    
    # Convert transformed data back to DataFrame
    train_inputs_df = pd.DataFrame(train_inputs_transformed, columns=all_feature_names)
    val_inputs_df = pd.DataFrame(val_inputs_transformed, columns=all_feature_names)
    
    train_targets_df = pd.DataFrame(train_targets).reset_index(drop=True)
    val_targets_df = pd.DataFrame(val_targets).reset_index(drop=True)
    
    return train_inputs_df, train_targets_df, val_inputs_df, val_targets_df, input_cols, scaler, encoder

def preprocess_new_data(new_df: pd.DataFrame, input_cols: List[str], scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Preprocesses new data using the existing scaler, encoder, and input columns.
    
    Args:
        new_df (pd.DataFrame): The new input dataframe.
        input_cols (list): List of input column names.
        scaler (MinMaxScaler): The pre-trained scaler for numeric columns.
        encoder (OneHotEncoder): The pre-trained encoder for categorical columns.
        
    Returns:
        pd.DataFrame: The preprocessed new data.
    """
    new_df = drop_unnecessary_columns(new_df, ["Surname", "CustomerId"])
    new_inputs = new_df[input_cols]
    
    numeric_cols = new_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = new_inputs.select_dtypes(include='object').columns.tolist()
    
    numeric_transformer = Pipeline(steps=[('scaler', scaler)]) if scaler else 'passthrough'
    categorical_transformer = Pipeline(steps=[('onehot', encoder)])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    new_inputs_transformed = preprocessor.fit_transform(new_inputs)
    one_hot_names = encoder.get_feature_names_out(categorical_cols)
    all_feature_names = numeric_cols + list(one_hot_names)
    
    new_inputs_df = pd.DataFrame(new_inputs_transformed, columns=all_feature_names)
    
    return new_inputs_df