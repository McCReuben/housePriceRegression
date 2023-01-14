import numpy as np
import pandas as pd
from IPython.display import Image
from pandas_profiling import ProfileReport
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import (IterativeImputer, KNNImputer, MissingIndicator,
                            SimpleImputer)
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from xgboost import XGBRegressor

final_test = pd.read_csv('data/test.csv')
train_init = pd.read_csv('data/train.csv')
train_init = train_init.drop(columns=["Id"])


x_train = ""

numerical = x_train.select_dtypes(include='number')
numerical_cols = numerical.columns.to_list()

numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])


categorical = x_train.select_dtypes(exclude=["number"]).columns.to_list()
categorical_cols = categorical.columns.to_list()

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


full_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_cols),
    ('category', categorical_pipeline, categorical_cols)
])

full_processor.fit_transform(x_train)


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

lasso = Lasso(alpha=0.1)

lasso_pipeline = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', lasso)
])

_ = lasso_pipeline.fit(x_train, y_train)

preds = lasso_pipeline.predict(X_test)


mean_absolute_error(y_test, preds)
lasso_pipeline.score(X_test, y_test)
