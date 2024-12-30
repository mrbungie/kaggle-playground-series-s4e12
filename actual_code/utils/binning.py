from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import ContinuousOptimalBinning
import pandas as pd
import numpy as np

class ContinuousOptimalBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, metric: Literal["indices", "mean", "bins"] = "indices"):
        """
        Transformer to apply ContinuousOptimalBinning column by column.

        Parameters:
        binning_params (dict): Parameters to pass to ContinuousOptimalBinning for each column.
        """
        self.binning_models = {}
        self.metric = metric

    def fit(self, X, y):
        """
        Fit the ContinuousOptimalBinning models column by column.

        Parameters:
        X (pd.DataFrame): The input dataframe with continuous features.
        y (array-like): Target variable used for optimal binning.

        Returns:
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        for column in X.columns:
            binning_model = ContinuousOptimalBinning(name=column, dtype="numerical")
            binning_model.fit(X[column], y)
            self.binning_models[column] = binning_model

        return self

    def transform(self, X):
        """
        Transform the dataset using the fitted binning models.

        Parameters:
        X (pd.DataFrame): The input dataframe to transform.

        Returns:
        pd.DataFrame: Transformed dataframe with columns replaced by bin means.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        X_transformed = pd.DataFrame(index=X.index)

        for column in X.columns:
            if column not in self.binning_models:
                raise ValueError(f"Column {column} was not fitted.")

            binning_model = self.binning_models[column]
            transformed_column = binning_model.transform(X[column], metric=self.metric)

            X_transformed[column] = transformed_column

        return X_transformed