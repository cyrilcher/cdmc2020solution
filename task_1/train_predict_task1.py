import random
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle


def load_data():
    train_df = pd.read_csv('data/AndroidMalware_Training_set.csv', low_memory=False, header=None)
    train_df = train_df[~train_df[2495].isna()]
    test_df = pd.read_csv('data/AndroidMalware_Testing_set.csv',low_memory=False,header=None)
    test_df = test_df.fillna(0)
    return train_df, test_df


def get_nonunique(df):
    columns_to_drop = []
    for i in df.iloc[:,:-4].columns:
        if len(df[i].unique()) == 1:
            columns_to_drop.append(i)
    columns_to_drop = list(set(columns_to_drop) & set(df.columns))
    return columns_to_drop


def get_corr(df):
    independent = []
    correlated = []
    random.seed(23)
    corr_df = df.loc[random.sample(list(df.index.values), 2000)].copy()
    corr_df = shuffle(corr_df)
    corr_df = corr_df.corr()
    for i in tqdm(corr_df.index):
        for j in corr_df.columns:
            if i != j and abs(corr_df.at[i, j]) == 1:
                if i not in independent and i not in correlated:
                    independent.append(i)
                if j in independent:
                    continue
                else:
                    correlated.append(j)
    correlated = list(set(correlated))
    return correlated


def feature_selection(train_df, test_df):
    # drop nonunique columns
    nonunique_cols = get_nonunique(train_df)
    train_df = train_df.drop(nonunique_cols, axis=1)
    test_df = test_df.drop(nonunique_cols, axis=1)
    # drop correlated columns
    correlated_cols = get_corr(train_df)
    train_df = train_df.drop(correlated_cols, axis=1)
    test_df = test_df.drop(correlated_cols, axis=1)
    return train_df, test_df


def fit_predict_rf(train_df, test_df):
    X_train = train_df.iloc[:,:-4]
    X_test = test_df.iloc[:,:-1]
    y_train = train_df[2495].copy()
    params = {'n_estimators':250, 'max_depth':None, 'class_weight':None, 'n_jobs':-1} #123
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_df['labels'] = y_pred
    test_df['index'] = test_df.index + 1

    test_df[['index', 'labels']].to_csv('task1_submission.csv', index=None, header=None)


if __name__ == '__main__':
    print('preparing data')
    train_df, test_df = load_data()
    train_df, test_df = feature_selection(train_df, test_df)
    print('training model and generating submission')
    fit_predict_rf(train_df, test_df)