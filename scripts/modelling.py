
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import joblib
import pandas as pd
from math import sqrt, floor

import sys
from importlib import reload
reload(sys.modules['settings'])
from settings import features, target, numerical_features, categorical_features, random_grid, problem_type, SEED

if problem_type == 'classification':
    clfs = [RandomForestClassifier(random_state=SEED)]#, LogisticRegression(random_state=SEED) , LinearSVC()]
if problem_type == 'regression':
    clfs = [RandomForestRegressor(random_state=SEED)]#, Ridge(random_state=SEED) , LinearSVR()]

def split(df, by = 'random', col = None):
    '''For results validation, apply specific strategies
    
    Args
        col (str): time series or area column
        mode (str): 'random', 'date', 'area'
    Returns
        CVSplitter/indices you can pass to sklearn.model_selection.cross_validate
    '''
    if by == 'random':
        return KFold(shuffle=True, n_splits=5)
    if by == 'date':
        return _ts_split(df, col)
    if by == 'area':
        return _geo_split(df, col)

def sk_pipeline(clf):
    '''Adds step that does scaling and one hot encoding
    
    Args
        clf (sklearn classifier/Pipeline): classifier to build preprocessors for
    Returns
        (sklearn Pipeline): Pipeline which includes scaling and one hot encoding
    '''
    global numerical_features, categorical_features
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop=None))])
    numerical_transformer = Pipeline(steps=[('scaler', RobustScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)],
        remainder='passthrough')
    return Pipeline(steps=[('preprocessor', preprocessor),('classifier', clf)])

def get_best_model(df, split_by = 'random', based_on = 'accuracy_score', model_path = 'model.pkl'):
    '''Get best model based on classifiers defined in modelling.py

    Args
        df (pd.DataFrame): data to train model on
        split_by (str): validation strategy, passed to split() function
        based_on (str): name of scorer to use
        model_path (str): where to save model
    Returns
        (sklearn pipeline): best model
    '''
    print(f'Cross validation by {split_by}')
    cv = split(df, by = split_by)

    X, y = df[features], df[target]

    clf_cv = RandomizedSearchCV(estimator=sk_pipeline(clfs[0]),
                                param_distributions=random_grid,
                                n_iter=25,
                                cv=cv,
                                n_jobs=-1,
                                verbose=1)
    clf_cv.fit(X,y)
    print('Best model cross-validated score:')
    print(clf_cv.best_score_)
    joblib.dump(clf_cv.best_estimator_, model_path)
    print(f'Saved model file to {model_path}')
    return clf_cv.best_estimator_

def train(clf, df, train_size = 0.8, model_path = 'model.pkl'):
    '''Run training using one train-test split

    Args
        clf (sklearn classifier): model to use for training
        df (pd.DataFrame): data to use for fitting model
        train_size (int): percentage of data to use for fitting model
        model_path (str): path where model will be saved
    Returns
        (sklearn pipeline): trained model
    
    '''
    df = df.sample(frac = 1).reset_index(drop = True)
    split = int(floor(len(df)*train_size))
    X_train, y_train = df.loc[:split, features], df.loc[:split, target]
    X_val, y_val = df.loc[split:, features], df.loc[split:, target]

    pipe = sk_pipeline(clf) # add onehot encoding and scaling
    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_val)
    metrics = evaluate(y_val, y_pred)
    print('Model results on validation set:')
    print(metrics)

    joblib.dump(pipe, model_path)
    print(f'Saved model file to {model_path}')
    return pipe

def predict(df, clf = None, model_path = 'model.pkl'):
    '''Predict on data using classifier

    Args
        df (pd.DataFrame): data to predict on
        clf (sklearn classifier): model to use for predicting
        model_path (str): path where model is saved
    Returns
        (pd.DataFrame): predictions for df
    '''
    X = df[features]
    if clf is None:
        clf = joblib.load(model_path)
    return clf.predict(X)

def evaluate(y_true, y_pred):
    '''Calculates metrics of accuracy between actual values and model predicted values.
    
    Args
        y_true (list of int or str): actual labels
        y_pred (list of int or str): predicted labels
    Returns
        (dict): accuracy metrics based on problem type
    '''
    if problem_type == 'classification':
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average = 'weighted'),
        }
    if problem_type == 'regression':
        return {
            'correlation': pearsonr(y_true, y_pred)[0],
            'r2': pearsonr(y_true, y_pred)[0]**2,#r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': sqrt(mean_squared_error(y_true, y_pred)),
        }

def error_by_subgroup(input_df, col, n = 3):
    '''Calculates error by subgroup for analysis

    Args
        input_df (pd.DataFrame): dataframe to calculate errors for
        col (str): column to divide into subgroups for error calculation
        n (int): number of rows to limit output
    Returns
        df (pd.DataFrame): table of error results
    
    '''
    print('Acceptance for column: {}'.format(col))
    records = []

    unq_val = list(input_df[col].dropna().unique())

    for val in unq_val:
        subset = input_df[input_df[col] == val]
        accpt = calculate_metrics(list(subset['model_preds']), list(subset['price']))
        records.append((val, subset.shape[0], accpt))

    df = pd.DataFrame(records)
    df.columns = ['unq_val', 'cnt', 'accpt']
    df = df.sort_values('accpt', ascending = False)
    print(df.head(n))
    print(df.tail(n))

    return df