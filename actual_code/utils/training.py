import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_log_error

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.base import clone

def train_with_cross_validation(X, y, splitter, pipeline_template, preprocess_y=lambda y: y, post_process_y=lambda y: y, X_test=None, scorer=root_mean_squared_log_error):
    # Initialize placeholders
    oof_predictions = pd.Series(index=y.index, dtype=float)  # Out-of-training fold predictions as a pandas Series
    val_scores = []  # Validation scores for each fold
    train_scores = []
    fold_test_predictions = pd.DataFrame(index=X_test.index) if X_test is not None else None  # DataFrame for test predictions per fold

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X)):
        print(f"Starting fold {fold_idx + 1}")

        # Split data into training and validation folds
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Apply preprocessing to target
        y_train_fold_transformed = preprocess_y(y_train_fold)
        y_val_fold_transformed = preprocess_y(y_val_fold)

        # Clone the pipeline to ensure independence between folds
        pipeline = clone(pipeline_template)

        # Fit the pipeline to the training data
        pipeline.fit(X_train_fold, y_train_fold_transformed)

        # Train predictions
        train_predictions = post_process_y(pipeline.predict(X_train_fold))

        # Generate predictions for the validation set
        val_predictions = post_process_y(pipeline.predict(X_val_fold))

        # Calculate validation score and store predictions
        val_score = scorer(val_predictions, y_val_fold)
        train_score = scorer(train_predictions, y_train_fold)
        val_scores.append(val_score)
        train_scores.append(train_score)

        # Accumulate out-of-fold predictions
        oof_predictions.iloc[val_idx] = val_predictions

        # Generate and store test predictions for this fold
        if X_test is not None:
            fold_test_predictions[f'fold_{fold_idx + 1}'] = post_process_y(pipeline.predict(X_test))

        # Print fold results
        print(f"Fold {fold_idx + 1} - Train: {train_score}, Validation: {val_score}")

    print("Cross-validation complete.")
    mean_val_score = np.mean(val_scores)
    print(f"Mean validation RMSLE: {mean_val_score}")
    mean_train_score = np.mean(train_scores)
    print(f"Mean train RMSLE: {mean_train_score}")

    # Fit the pipeline on the full data
    final_pipeline = clone(pipeline_template)
    final_pipeline = final_pipeline.fit(X, preprocess_y(y))

    # Generate test predictions if X_test is provided
    test_predictions = None
    if X_test is not None:
        test_predictions = post_process_y(final_pipeline.predict(X_test))

    # Create summary
    summary = {
        "mean_validation_score": mean_val_score,
        "fold_validation_scores": val_scores,
        "mean_train_score": mean_train_score,
        "fold_train_scores": train_scores
    }

    return {
        "trained_pipeline": final_pipeline,
        "oof_predictions": oof_predictions,
        "validation_scores": val_scores,
        "summary": summary,
        "test_predictions": test_predictions,  # Return test predictions
        "fold_test_predictions": fold_test_predictions  # Return fold test predictions
    }