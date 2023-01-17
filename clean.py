import numpy as np
import pandas as pd
from IPython.display import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (IterativeImputer, KNNImputer, MissingIndicator,
                            SimpleImputer)
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

final_test = pd.read_csv('data/test.csv')
train_init = pd.read_csv('data/train.csv')
train_init = train_init.drop(columns=["Id"])


labels = train_init["SalePrice"]
features = train_init.drop(columns=["SalePrice"])

# Set up training and testing data
x_train,x_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=6)



numerical = x_train.select_dtypes(include='number')
numerical_cols = numerical.columns.to_list()

numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])


categorical = x_train.select_dtypes(exclude=["number"])
categorical_cols = categorical.columns.to_list()

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


full_preprocessor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_cols),
    ('category', categorical_pipeline, categorical_cols)
])


x_train_clean = full_preprocessor.fit_transform(x_train)
x_test_clean = full_preprocessor.fit_transform(x_test)

