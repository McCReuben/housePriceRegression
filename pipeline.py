import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
