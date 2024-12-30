class AutogluonRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, eval_metric=None, time_limit=60, hyperparameters=None, verbosity=2):
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.hyperparameters = hyperparameters
        self.verbosity = verbosity
        self.tabular_predictor = None

    def fit(self, X, y):
        X = pd.DataFrame(X)
        X['target'] = y
        self.tabular_predictor = TabularPredictor(problem_type='regression', label='target', eval_metric=self.eval_metric)
        self.tabular_predictor.fit(train_data=X, time_limit=self.time_limit, hyperparameters=self.hyperparameters, verbosity=self.verbosity)

    def predict(self, X):
        if self.model is None:
            raise NotFittedError("Model has not been fitted. Call 'fit' before 'predict'.")
        return self.tabular_predictor.predict(X)
