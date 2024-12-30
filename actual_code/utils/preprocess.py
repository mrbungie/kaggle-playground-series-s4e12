import pandas as pd
import numpy as np

from typing import Literal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin
import miceforest as mf

import hashlib
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import ContinuousOptimalBinning
from .binning import ContinuousOptimalBinningTransformer
from .cache import CachedPreprocessor
from .categoricals import CategoricalToStringTransformer

# Define feature columns
INIT_NUMERICAL_COLS = ['id', 'Age', 'Annual Income', 'Number of Dependents', 'Health Score',
       'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']
INIT_CATEGORICAL_COLS = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
       'Policy Type', 'Policy Start Date', 'Customer Feedback',
       'Smoking Status', 'Exercise Frequency', 'Property Type']

categorical_one_hot = ['Gender', 'Marital Status', 'Occupation', 'Location', 'Smoking Status', 'Property Type']
categorical_ordinal = ['Education Level', 'Policy Type', 'Customer Feedback', 'Exercise Frequency']
numerical_features = [
    'Age', 'Annual Income', 'Number of Dependents', 'Health Score',
    'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration',
    'Customer Tenure', 'Claim Frequency', 'Policy Start Year', 'Policy Start Month',
    'Policy Start Day', 'Policy Start Hour', 'Policy Start Minute', 'Policy Start Second',
    'Annual Income log 10', 'Previous Claims log'
]
target = 'Premium Amount'

# Define ordinal encoding mappings
ordinal_mappings = {
    'Education Level': ['High School', "Bachelor's", "Master's", 'PhD'],
    'Policy Type': ['Basic', 'Comprehensive', 'Premium'],
    'Customer Feedback': ['Poor', 'Average', 'Good'],
    'Exercise Frequency': ['Rarely', 'Monthly', 'Weekly', 'Daily']
}

def treat_dataset_pandas_init(dataset, keep_null_counts=True, target_transform="log10p", process_as_category=True):
    dataset = dataset.copy()
    dataset['Policy Start Date'] = pd.to_datetime(dataset['Policy Start Date'])
    
    last_date = dataset['Policy Start Date'].max()
    reference_date = last_date + pd.Timedelta(days=1)
    
    dataset['Customer Tenure'] = (reference_date - dataset['Policy Start Date']).dt.days / 365
    dataset['Claim Frequency'] = dataset['Previous Claims'] / dataset['Insurance Duration']
    dataset['Policy Start Year'] = dataset['Policy Start Date'].dt.year
    dataset['Policy Start Month'] = dataset['Policy Start Date'].dt.month
    dataset['Policy Start Day'] = dataset['Policy Start Date'].dt.day
    
    dataset['Month_name'] = dataset['Policy Start Date'].dt.month_name().astype('category')
    dataset['Day_of_week'] = dataset['Policy Start Date'].dt.day_name().astype('category')
    dataset['Week'] = dataset['Policy Start Date'].dt.isocalendar().week.astype(float)
    dataset['Year_sin'] = np.sin(2 * np.pi * dataset['Policy Start Year'].astype(float))
    dataset['Month_sin'] = np.sin(2 * np.pi * dataset['Policy Start Month'].astype(float) / 12) 
    dataset['Month_cos'] = np.cos(2 * np.pi * dataset['Policy Start Month'].astype(float) / 12)
    dataset['Day_sin'] = np.sin(2 * np.pi * dataset['Policy Start Day'].astype(float) / 31)  
    dataset['Day_cos'] = np.cos(2 * np.pi * dataset['Policy Start Day'].astype(float) / 31)
    dataset['Group']=(dataset['Policy Start Year'].astype(float)-2020)*48+dataset['Policy Start Month'].astype(float)*4+dataset['Policy Start Day'].astype(float)//7 # https://www.kaggle.com/code/backpaker/rid-train-h2o
    
    del dataset["id"]
    del dataset["Policy Start Date"]

    if target_transform == "log1p":
        dataset["Premium Amount"] = np.log1p(dataset["Premium Amount"])
    elif target_transform == "sqrt":
        dataset["Premium Amount"] = np.sqrt(dataset["Premium Amount"])
    elif target_transform == "identity":
        pass

    dataset["Annual Income log 10"] = np.log10(dataset["Annual Income"])
    dataset["Previous Claims log"] = np.log1p(dataset["Previous Claims"])

    for col in [col for col in dataset.columns if col not in ["target","id","Gender","Year","Year_sin","Year_cos","Month_sin","Month_cos","Day_sin","Day_cos","Week","Group","Policy Start Year","Policy Start Month","Policy Start Day","Policy Start Hour","Policy Start Minute","Policy Start Second","Premium Amount"]]:
        dataset[f"{col}_null"] = dataset[col].isnull()
    

    if keep_null_counts:
        dataset['Null columns'] = dataset.isnull().sum(axis=1)

    dataset['contract length'] = pd.cut(
        dataset["Insurance Duration"].fillna(99),  
        bins=[-float('inf'), 1, 3, float('inf')],  
        labels=[0, 1, 2]
    ).astype(int)
    dataset['Income to Dependents Ratio'] = dataset['Annual Income'] / (dataset['Number of Dependents'].fillna(0) + 1)
    dataset['Income_per_Dependent'] = dataset['Annual Income'] / (dataset['Number of Dependents'] + 1)
    dataset['CreditScore_InsuranceDuration'] = dataset['Credit Score'] * dataset['Insurance Duration']
    dataset['Health_Risk_Score'] = dataset['Smoking Status'].apply(lambda x: 1 if x == 'Smoker' else 0) + \
                                dataset['Exercise Frequency'].apply(lambda x: 1 if x == 'Low' else (0.5 if x == 'Medium' else 0)) + \
                                (100 - dataset['Health Score']) / 20
    dataset['Credit_Health_Score'] = dataset['Credit Score'] * dataset['Health Score']
    dataset['Health_Age_Interaction'] = dataset['Health Score'] * dataset['Age']
    # Creating Ratio variables, as well as categorizing continuous ones
    dataset['Claims v Duration'] = dataset['Previous Claims'] / dataset['Insurance Duration']
    dataset['Health vs Claims'] = dataset['Health Score'] / (dataset['Previous Claims'] + 1)
    dataset['Cat Credit Score'] = dataset['Credit Score'].astype('string').astype('category')
    dataset['Int Credit Score'] = dataset['Credit Score'].apply(lambda x: int(x) if pd.notna(x) else x)

    if process_as_category:
        for col in INIT_CATEGORICAL_COLS:
            if col in dataset.columns:
                dataset[col] =  dataset[col].astype("category")
    
    return dataset

def build_preprocessing_pipeline(imputation_method='simple', exclude_cols=None, train_data=None, binning_method: Literal["opti-ordinal","opti-ohe","opti-means", None] = "opti-ordinal", categorical_to_string=True):
    if exclude_cols is None:
        exclude_cols = []
        
    if imputation_method == 'mice':
        init_steps = [('mice', mf.ImputationKernel(train_data, num_datasets=1, random_state=1))]
    else:
        init_steps = []
        
    if categorical_to_string:
        init_steps.append(('categorical_to_string', CategoricalToStringTransformer()))
        
    if imputation_method == 'simple':
        ohe_init_steps = [('ohe_imputer', SimpleImputer(strategy='constant', fill_value="Unknown", missing_values=pd.NA))]
        ordinal_init_steps = [('ordinal_imputer', SimpleImputer(strategy='constant', fill_value="Unknown", missing_values=pd.NA))]
        numerical_init_steps = [('num_imputer', SimpleImputer(strategy='mean', missing_values=pd.NA))]
    else:
        ohe_init_steps = []
        ordinal_init_steps = []
        numerical_init_steps = []

    # Separate numeric and categorical columns
    numeric_cols = train_data.select_dtypes(include=['float', 'int']).columns
    categorical_cols = [col for col in train_data.select_dtypes(include=['object', 'category', 'bool']).columns if col not in exclude_cols and col not in ordinal_mappings.keys()] 

    # Create column transformers
    one_hot_transformer = Pipeline(steps=ohe_init_steps + [
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    ordinal_transformers = {
        col: Pipeline(steps=ordinal_init_steps + [
            (f'ordinal_{col}', OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        for col, categories in ordinal_mappings.items()
    }
    
    if binning_method == "opti-ordinal":
        binning_column_transformer = ContinuousOptimalBinningTransformer(metric="indices")
        steps = [
            ('column_transformer_binning', binning_column_transformer),
            ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]
    elif binning_method == "opti-ohe":
        binning_column_transformer = ContinuousOptimalBinningTransformer(metric="indices")
        steps = [
            ('column_transformer_binning', binning_column_transformer),
            ('opti_numerical_ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ]
    elif binning_method == "opti-means":
        binning_column_transformer = ContinuousOptimalBinningTransformer(metric="mean")
        steps = [
            ('column_transformer_binning', binning_column_transformer)
        ]
    else:
        steps = numerical_init_steps + [
            ('scaler', StandardScaler())
        ]
        
    numerical_transformer = Pipeline(steps=steps)

    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', one_hot_transformer, categorical_cols),
            ('education', ordinal_transformers['Education Level'], ['Education Level']),
            ('policy_type', ordinal_transformers['Policy Type'], ['Policy Type']),
            ('customer_feedback', ordinal_transformers['Customer Feedback'], ['Customer Feedback']),
            ('exercise_frequency', ordinal_transformers['Exercise Frequency'], ['Exercise Frequency']),
            ('num', numerical_transformer, numeric_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    final_pipeline = Pipeline(steps=init_steps + [
        ('preprocessor', preprocessor),
    ])

    return CachedPreprocessor(final_pipeline)

## 
