import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    final_test = pd.read_csv('data/test.csv')
    train_init = pd.read_csv('data/train.csv')

    labels = train_init["SalePrice"]
    features = train_init.drop(columns=["SalePrice"])

    # Set up training and testing data
    raw_train, final_test, x_train,x_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=7)
    return x_train,x_test,y_train,y_test
