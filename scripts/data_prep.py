'Wrappers for data processing steps. To edit, modify lines in preprocessing_workflow()'

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import os

def _set(
    target,
    data,
    CAT_FILL = 'Missing',
    NUM_FILL = 0,
    SEED = 42
):
    '''Creates settings.py from settings_template.py, to be used by eda.py and modelling.py
    
    Args
        target (str): column to use as label for the prediction problem
        data (pd.DataFrame): data to create settings for
        CAT_FILL (str): value to set for blanks in string data
        NUM_FILL (int): value to set for blanks in numerical data
        SEED (int): number used to generate random numbers
    Returns
        None
    '''

    features = list(set(data.columns)-{target})
    categorical_features = list(data[features].select_dtypes(include=['object']).columns)
    numerical_features = list(data[features].select_dtypes(include=['number']).columns)

    try:
        if str(data[target].dtype) in ['object']:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
        print(f'Set problem_type to {problem_type}.')
    except KeyError:
        # Temporarily set to classification, overwritten after preprocessing_workflow is finished
        problem_type = 'classification'

    text = f'''
# Data prep parameters ----
target = '{target}'
categorical_features = {categorical_features}
numerical_features = {numerical_features}
features = categorical_features + numerical_features
problem_type = '{problem_type}'

CAT_FILL = "{CAT_FILL}"
NUM_FILL = {NUM_FILL}

SEED = {SEED}
'''
    path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    fname = os.path.join(path, 'settings_template.py')
    f0 = open(fname, 'r')
    orig_text = f0.readlines()
    fname2 = os.path.join(path, 'settings.py')
    f = open(fname2, 'w')
    f.writelines(text)
    f.writelines(orig_text)
    f.close()

def filter(df_, by_ = ["col1 > 5"]):
    '''Runs list of filter strings sequentially as df.querys

    Args
        df_ (pd.DataFrame): data to filter on
        by_ (list of str): filters to pass to df.query(), ordered by filtering order
    Returns
        df (pd.DataFrame): filtered df
    '''
    df = df_.copy()
    for f in by_:
        df = df.query(f)
    return df

def clip_outliers(data, p = 0.05):
    '''Remove outliers by setting a floor and ceiling for numerical values based on top p% and bottom (1-p)%

    Args
        data (pd.Series): set of numbers to clip outliers on
        p (float): percentile on which to clip
    Returns
        data (pd.Series): clipped series
    '''
    p_lower = data.quantile(p)
    p_upper = data.quantile(1-p)
    data[data < p_lower] = p_lower
    data[data > p_upper] = p_upper
    #return data[np.logical_and(data < p_upper, data > p_lower)]
    return data

def preprocessing_workflow(df_):
    '''Transforms raw df to sklearn pipeline-ready df
    
    Args
        df_ (pd.DataFrame): raw dataset
    Returns
        df (pd.DataFrame): preprocessed data
    '''

    import sys
    from importlib import reload
    reload(sys.modules['settings'])
    from settings import target, categorical_features, numerical_features, CAT_FILL, NUM_FILL, SEED

    df = df_.copy()
    print('Raw data:')
    print(df.shape)

    print('1. Set datatypes')
    # df = df.astype({
    #     'col1': float, 'col2': str
    # })

    print('2. Merge dfs')
    # df = pd.merge(df1, df2)
    print(df.shape)

    print('3. Rename columns')
    # df.rename(columns = {
    #     'col_1': 'col1',
    #     'col_2': 'col2'
    # }, inplace = True)

    print('4. Create new columns')
    spellcasters = ['Bard','Cleric','Druid','Sorcerer','Warlock','Wizard']
    df = df.assign(
        is_spellcaster = df['primary_class'].isin(spellcasters).astype(int).astype(str),
        is_human = (df['processedRace']=='Human').astype(int),
        #'col5': df['col4'].apply(winsorize, args = (0.05, 0.95))
    )
    print(df.shape)

    print('5. Impute missing values')
    impute_categorical = {f:CAT_FILL for f in categorical_features}
    impute_numerical = {f:NUM_FILL for f in numerical_features}
    df = df.fillna(value = {
        **impute_numerical,
        **impute_categorical,
    }) # count of imputed rows

    print('6. Filter rows')
    # filter(df, by_ = [
    #     "col > 5", 
    #     "col2 == 'Terrace House'"
    # ]) # print out row count
    print(df.shape)

    print('7. Drop unnecessary columns')
    df = df.drop(labels = [
        'countryCode','primary_class', 'hash'
    ], axis = 1)
    print(df.shape)

    _set(
        target = target,
        data = df,
        CAT_FILL = CAT_FILL,
        NUM_FILL = NUM_FILL,
        SEED = SEED
    )

    return df

class Preprocess( BaseEstimator, TransformerMixin ):
    '''Sklearn transformer that applies preprocessing_workflow()
    
    No need to edit, just add custom code to preprocessing_workflow
    '''
    def __init__(self):
        pass
    
    def fit( self, X, y = None ):
        return self #nothing

    def transform( self, X, y = None ):
        X = X.copy()
        X = preprocessing_workflow(X)

        return X

def preprocess(
    df,
    target,
    as_is = False,
    CAT_FILL = 'Missing',
    NUM_FILL = 0,
    SEED = 42
):
    '''Runs preprocessing_workflow() over input df and generates settings.py
    
    Args
        df (pd.DataFrame): raw data
        target (str): column to use as label for the prediction problem
        as_is (bool): if False, apply preprocessing steps and settings creation. if True, don't apply any preprocessing, just create settings.py
        CAT_FILL (str): value to set for blanks in string data
        NUM_FILL (int): value to set for blanks in numerical data
        SEED (int): number used to generate random numbers
    Returns
        (pd.DataFrame): preprocessed data
    '''

    _set(
        target = target,
        data = df,
        CAT_FILL = CAT_FILL,
        NUM_FILL = NUM_FILL,
        SEED = SEED
    )

    if as_is:
        return df
    
    pipe = Pipeline(steps=[
        ('preprocess', Preprocess()),
    ])
    return pipe.fit_transform(df)
