import os
from operator import index

import numpy as np
import pandas as pd
import pandas_profiling
import plotly.express as px
import streamlit as st
from IPython.display import Image
from pycaret.regression import (compare_models, load_model, pull, save_model,
                                setup)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from streamlit_pandas_profiling import st_profile_report
from xgboost import XGBRegressor

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("ML End-to-End Pipeline")
    choice = st.radio("Navigation", ["Upload Data","Exploratory Anaysis","ML Modelling", "Results"])


if choice == "Upload Data":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Exploratory Anaysis": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report(title="Data Profiling Report", minimal=True, pool_size=8)
    profile_df.to_file("results/EDA.html")
    st_profile_report(profile_df)

if choice == "ML Modelling": 
    st.title("Choose a Model")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        
        save_model(best_model, 'best_model')

if choice == "Results": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")