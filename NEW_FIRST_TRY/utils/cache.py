import hashlib
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

class CachedPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor, cache=None):
        """
        Wrapper for caching preprocessors.

        :param preprocessor: The preprocessor to be wrapped (e.g., StandardScaler, PCA).
        :param cache: Dictionary or external cache for storing fitted preprocessors.
        """
        self.preprocessor = preprocessor
        self.cache = cache if cache is not None else {}
    
    def _calculate_hash(self, X, y=None, **kwargs):
        """
        Calculate a unique hash for the dataset, parameters, and kwargs.
        """
        data_to_hash = pickle.dumps({
            "X": X, 
            "y": y, 
            "kwargs": kwargs
        })
        return hashlib.sha256(data_to_hash).hexdigest()

    def fit(self, X, y=None, **kwargs):
        """
        Fit the preprocessor, using cache if available.
        """
        dataset_hash = self._calculate_hash(X, y, **kwargs)
        if dataset_hash in self.cache:
            print("Using cached preprocessor")
            self.preprocessor = self.cache[dataset_hash]
        else:
            print("Fitting and caching preprocessor")
            self.preprocessor.fit(X, y, **kwargs)
            self.cache[dataset_hash] = pickle.loads(pickle.dumps(self.preprocessor))
        return self

    def transform(self, X):
        """
        Transform the data using the fitted preprocessor.
        """
        if not hasattr(self.preprocessor, "transform"):
            raise NotImplementedError("The preprocessor must implement transform.")
        return self.preprocessor.transform(X)

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the preprocessor and transform the data.
        """
        self.fit(X, y, **kwargs)
        return self.transform(X)