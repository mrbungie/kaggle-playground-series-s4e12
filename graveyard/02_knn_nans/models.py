import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import Pipeline

class BinnedXGBRegressionModel():
    def __init__(self, 
                 n_bins=5, 
                 classifier_base_model=XGBClassifier(), 
                 regressor_base_model=XGBRegressor(), 
                 random_state=None):
        """
        A model that first bins the target using qcut, uses an XGBClassifier 
        to predict the bin, and then uses separate XGBRegressor models for 
        each bin to predict the final continuous values.

        Parameters
        ----------
        n_bins : int, default=5
            Number of quantile-based bins to discretize the target into.

        classifier_params : dict or None
            Parameters passed to XGBClassifier.

        regressor_params : dict or None
            Parameters passed to XGBRegressor.

        random_state : int or None
            Random state for reproducibility.
        """
        self.n_bins = n_bins
        self.classifier_base_model = classifier_base_model
        self.regressor_base_model = regressor_base_model
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        y_series = pd.Series(y)

        # Discretize the target into bins
        self.bin_labels_ = pd.qcut(
            y_series, 
            q=self.n_bins, 
            labels=False, 
            duplicates='drop'
        ).astype(int).values
        
        # Fit XGBClassifier to predict the bin
        self.bin_classifier_ = clone(self.classifier_base_model)
        self.bin_classifier_.fit(X, self.bin_labels_)

        # Determine the actual number of bins used (in case duplicates='drop' reduced it)
        self.actual_bins_ = len(np.unique(self.bin_labels_))

        # Fit a separate XGBRegressor for each bin
        self.models_ = []
        for bin_idx in range(self.actual_bins_):
            indices = (self.bin_labels_ == bin_idx)
            reg = clone(self.regressor_base_model)
            reg.fit(X[indices], y[indices])
            self.models_.append(reg)

        return self

    def predict(self, X):
        check_is_fitted(self, ["bin_classifier_", "models_"])
        X = check_array(X)
        
        # Predict the probability distribution over the bins
        bin_proba = self.bin_classifier_.predict_proba(X)
        
        # Get predictions from each bin's regressor for all samples
        # This will result in a matrix of shape (n_samples, actual_bins_)
        bin_preds = np.column_stack([m.predict(X) for m in self.models_])
        
        # Compute the weighted average across bins for each sample
        # using the predicted probabilities as weights
        y_pred = np.sum(bin_proba * bin_preds, axis=1)
        
        return y_pred
    def predict_proba(self, X):
        """Predict the probability distribution over the bins."""
        check_is_fitted(self, ["bin_classifier_"])
        X = check_array(X)
        return self.bin_classifier_.predict_proba(X)