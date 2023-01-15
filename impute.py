from sklearn.impute import (IterativeImputer, KNNImputer, MissingIndicator,
                            SimpleImputer)


# Imputation
def give_missing_summary(df, feature):
    pass


def get_missing_values(df, quiet=False):
    # Create a boolean mask indicating the presence of missing values in each column
    na_vals = df.isna()
    # Get average number of missing values in each column
    na_percent = na_vals.mean()
    sorted_na_percent = na_percent.sort_values(ascending=False)
    # Print the column names and percentage of missing values for columns with missing values
    if (not quiet): print(sorted_na_percent[sorted_na_percent > 0])

    return sorted_na_percent.keys()

def missing_values_onebyone(df):
    features_missing_data = get_missing_values(df)
    # Go through features with missing values
    for feature in features_missing_data:
        give_missing_summary(df, feature)
        print("Options: dc (drop column), dr(drop missing rows), sI <strategy> (Simple Imputer by 'mean', 'median', or 'most_frequent'), iI <max_iter> <estimator> (Iterative Imputer, default estimator is Bayesian Ridge), kNN, f <flag> (mark missing values with <flag>)")
        ui = input().split()
        if ui[0] == "dc":
            df.drop[feature]
        elif ui[0] == "dr":
            df.dropna(subset=feature)
        elif ui[0] == "sI":
            strategy = "mean" if len(ui) == 1 else ui[1]
            SimpleImputer(strategy=strategy)
        elif ui[0] == "iI":
            max_iter = 10 if len(ui) < 2 else ui[1]
            imp = IterativeImputer(strategy=strategy)
        elif ui[0] == "kNN":
            max_iter = 10 if len(ui) < 2 else ui[1]
            imp = IterativeImputer(strategy=strategy)
        elif ui[0] == "f":
            pass
