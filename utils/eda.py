import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List

from sklearn.ensemble import IsolationForest

def summary(df: pd.DataFrame) -> pd.DataFrame:
    '''Returns a summary table with stats, missing values etc.'''
    print(f'data shape: {df.shape}')
    duplicates = df[df.duplicated()]
    print(f"Number of duplicates found and removed: {len(duplicates)}")
    df = df.drop_duplicates()
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values 
    summ['%missing'] = df.isnull().sum().values / len(df) * 100
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['average'] = desc['mean'].values
    summ['standard_deviation'] = desc['std'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ

def detect_outliers(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    '''Detect the outlier for each row'''
    # Subset the dataframe to only the specified features
    df_subset = df[features]

    # Initialize the Isolation Forest model
    clf = IsolationForest(contamination='auto')

    # Fit the model on the subset
    predictions = clf.fit_predict(df_subset)

    # Create a DataFrame to store the outlier count for each row
    outlier_count_df = pd.DataFrame({
        'Outlier_Count': [(pred == -1) for pred in predictions]
    })
    
    # Attach the outlier count to the original dataframe
    df['Outlier_Count'] = outlier_count_df
    
    # Return the dataframe with the added outlier count column
    return df


def plot_correlation_heatmap(df: pd.DataFrame, title_name: str='Train correlation') -> None:
    '''Plot correlation triangular heatmap.'''
    corr = df.corr()
    fig, axes = plt.subplots(figsize=(12, 8))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='RdBu_r', annot=True,annot_kws={"size": 8})
    plt.title(title_name)
    plt.show()
