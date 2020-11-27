'Plotting and sense checking functions'
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr

from settings import categorical_features, numerical_features

# import utils

# def quick_check(df):
#     'Check supplied columns for nulls or duplicates'
#     return utils.quick_checker_all(df.fillna(""))

def check_nulls_and_outliers(df, columns = None):
    '''
    1. Find rows missing some values
    2. Find outliers

    Args
        df (pd.DataFrame): data to check
        columns (list of str): subset of columns to check, can be None.
    Returns
        None
    '''
    if columns is None:
        columns = df.columns
    
    sub = df[columns]
    rows, cols = df.shape
    print(f'Total rows: {rows}')

    print('Variables with missing values:')
    print(sub.describe().transpose().query(f"count < {rows}")[['count', 'mean', 'std', 'min', 'max']])
    
    print('Variables with outliers:')
    numerical_features = df.select_dtypes(include=['number']).columns
    sub_num = sub[numerical_features]
    scaler = MinMaxScaler()
    scaler.fit(sub_num)
    scaler.transform(sub_num)
    sub_num.boxplot(figsize=(12,4))

def correlations_to_label(df, label = None, cols = None, show_above = 0.7):
    '''Show correlations for `cols` in `df` with values above `show_above`
    
    Args
        df (pd.DataFrame): data to check correlations on
        label (str): column used as label for prediction problem
        cols (list of str): columns to check correlation with label
        show_above (float): filter shown correlations to above this threshold
    Returns
        (pd.DataFrame): table of correlations
    '''

    if cols is None:
        cols = df.columns

    corr_table = []

    for column in numerical_features:
        spearman_r = spearmanr(df[column],df[label])[0]
        pearson_r = pearsonr(df[column],df[label])[0]
        
        corr_dict = {'feature': column,
                    'spearman_r': spearman_r,
                    'pearson_r': pearson_r,
                    'spearman_r_abs': np.abs(spearman_r),
                    'pearson_r_abs': np.abs(pearson_r)}
        
        corr_table.append(corr_dict)

    return (pd.DataFrame(corr_table)
            .sort_values(by='pearson_r_abs',ascending=False)
            .query(f"pearson_r_abs > {show_above}"))

def overlapping_histograms(df, feature = 'area', upper_bound = None, bins_ = 100):
    '''Plot a graph of overlapping histograms for particular column in a dataframe

    Args
        df (pd.DataFrame): data to check
        feature (str): column in df to plot
        upper_bound (int): maximum value for plotting axis
        bins_ (int): number of histogram splits
    Returns:
        None

    '''
    # upper_bound = 1*1e-7
    types = list(df['type'].unique())
    if upper_bound == None: upper_bound = df[feature].quantile(0.95);
    bins = np.linspace(0, upper_bound, bins_)
    for type_ in types:
        data = clip_outliers(df.loc[df['type'] == type_, feature])
        plt.hist(data, bins = bins, weights=np.ones(len(data)) / len(data), 
                 alpha=0.5, label=type_,
                 histtype = 'step',linewidth=2)
    plt.title(feature.capitalize())
    plt.legend(loc='upper right')
    plt.show()

    # # distribution of train vs test
    # sns.distplot(train_df2017['price'], bins=20, kde=False, rug=False, color="r", norm_hist = True, label="train")
    # sns.distplot(test_df2017['price'], bins=20, kde=False, rug=False, color="b", norm_hist = True, label="test")
    # # sns.kdeplot(train_df2017['price'], bw=.2, label="train")
    # # sns.kdeplot(test_df2017['price'], bw=.2, label="test")
    # plt.legend();

def pairplot(df, category, cols = None):
    '''Compares columns pairwise, adds a color based on category
    
    Args
        df (pd.DataFrame): data used to plot
        category (str): name of column to use for coloring data
    Returns
        None
    '''
    sns.pairplot(df, hue = category)

def scatter_wline(df, x, y):
    '''Creates a scatterplot with fitted line
    
    Args
        df (pd.DataFrame): data used to plot
        x, y (str): columns used to plot
    Returns
        None
    '''
    # positive correlation between price and avg postal price
    sns.lmplot(x=x, y=y, data=df)