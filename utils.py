import time
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import math
from sklearn.preprocessing import LabelEncoder


def save_submission(ids, prices):
    submission=pd.DataFrame()
    submission['Id']=ids
    submission['SalePrice']=prices
    
    dirPath = "submissions"
    os.makedirs(dirPath, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filePath = f"{dirPath}/submission-{timestr}.csv"

    submission.to_csv(filePath, index=False)
    return filePath

def draw_correlation_matrix(dataset):
    corrmat = dataset.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)

def draw_scatter_plots(dataset, cols):
    sns.set()
    sns.pairplot(dataset[cols], size = 2.5)
    plt.show();

# histogram and normal probability plot
def draw_hist_normal_prob_plot(dataset, features=[]):
    if features != None:
        for f in features:
            sns.distplot(dataset[f], fit=norm);
            fig = plt.figure()
            res = stats.probplot(dataset[f], plot=plt)
            plt.show()
    else:
        sns.distplot(dataset, fit=norm);
        fig = plt.figure()
        res = stats.probplot(dataset, plot=plt)
        plt.show()
        
def drop_features(dataset, features=[]):
    return dataset.copy(deep=True).drop(features, axis = 1)

def summary_of_missing_data(dataset, draw_histogram=False):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    result = missing_data[total>0]
    
    if draw_histogram:
        f, ax = plt.subplots(figsize=(15, 12))
        plt.xticks(rotation='90')
        sns.barplot(x=result.index, y=result['Percent'])
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        plt.show()
    
    return result

def add_new_feature_is_exist_or_not(dataset, features=[]):
    ds = dataset.copy(deep=True)
    for f in features:
        ds[f"Has{f}"] = pd.Series(len(ds[f]), index=ds.index)
        ds[f"Has{f}"] = 1 
        ds.loc[ds[f].isnull(), f"Has{f}"] = 0
    return ds

def encode_labels(dataset, features=[]):
    ds = dataset.copy(deep=True)
    for f in features:
        lbl = LabelEncoder() 
        lbl.fit(list(ds[f].values)) 
        ds[f] = lbl.transform(list(ds[f].values))
    return ds
