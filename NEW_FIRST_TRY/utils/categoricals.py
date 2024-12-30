from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class CategoricalToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting necessary, return self
        return self

    def transform(self, X):
        # Check if input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Create a copy of the DataFrame to avoid modifying the original data
        X_transformed = X.copy()
        
        # Iterate over each column in the DataFrame
        for col in X_transformed.columns:
            # Check if the column is of categorical data type
            if pd.api.types.is_categorical_dtype(X_transformed[col]):
                # Convert the categorical column to string
                X_transformed[col] = X_transformed[col].astype(str)
        
        return X_transformed
    
class CatogoricalUnknownTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        X_transformed = X.copy()
        # Iterate over each column in the DataFrame
        for col in X_transformed.columns:
            # Check if the column is of categorical data type
            if pd.api.types.is_categorical_dtype(X_transformed[col]):
                # Convert the categorical column to string
                X_transformed[col] = X_transformed[col].cat.add_categories("__Unknown__")
                X_transformed[col] = X_transformed[col].fillna("__Unknown__")
        
        return X_transformed
    
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None, drop_org=False):
        self.cat_cols = cat_cols
        self.drop_org = drop_org
        self.freq_encoding_dict = {}

    def fit(self, X, y=None):
        """
        Learn frequency encoding for categorical columns.
        
        Parameters:
        X : DataFrame
            Input data to fit the encoder.
        y : Ignored
        """
        X = pd.DataFrame(X)  # Ensure input is a DataFrame
        
        # Use specified categorical columns or infer them
        self.cat_cols = self.cat_cols or X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in self.cat_cols:
            self.freq_encoding_dict[col] = X[col].value_counts().to_dict()

        return self

    def transform(self, X):
        """
        Apply frequency encoding to the categorical columns.

        Parameters:
        X : DataFrame
            Input data to transform.

        Returns:
        DataFrame
            Transformed DataFrame with frequency-encoded columns.
        """
        X = pd.DataFrame(X).copy()  # Ensure input is a DataFrame

        for col in self.cat_cols:
            if col in X.columns:
                # Apply frequency encoding
                X[f"{col}_freq"] = X[col].map(self.freq_encoding_dict.get(col, {})).astype('category')
                if self.drop_org:
                    X.drop(columns=[col], inplace=True)

        return X