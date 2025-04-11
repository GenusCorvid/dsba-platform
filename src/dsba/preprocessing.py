from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from sklearn.model_selection import train_test_split


def split_features_and_target(
    df: DataFrame, target_column: str
) -> tuple[DataFrame, Series]:
    """
    Splits a DataFrame into features and target, which is a common format used by machine learning libraries such as scikit-learn.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def split_dataframe(
    df: DataFrame, test_size: float = 0.2
) -> tuple[DataFrame, DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=42)


def preprocess_dataframe(df):
    """
    Preprocess DataFrame by encoding categorical columns.
    ML algorithms typically can't only handle numbers, so there may be quite a lot of feature engineering and preprocessing with other types of data.
    Here, we take a very simplistic approach of applying the same treatment to all non-numeric columns.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Impute missing values in numeric columns with median
    if len(numeric_cols) > 0 and df[numeric_cols].isna().any().any():
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    
    # Impute missing values in categorical columns with most frequent value
    # and encode categorical columns
    for column in categorical_cols:
        # First impute missing values with most frequent value
        if df[column].isna().any():
            most_frequent = df[column].mode()[0]
            df[column] = df[column].fillna(most_frequent)
        
        # Then encode the categorical values
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
    
    return df
