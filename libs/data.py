"""
-*-coding: utf-8-*- 
Author: Yann Cherdo 
Creation date: 2020-07-01 09:18:29
"""

import pandas as pd
import numpy as np


def generate_XY_continuous_TS(df:pd.DataFrame, X_columns:list, Y_columns:list, w_x:int, w_y:int)->tuple:
    """
    Generate to X and Y lists of sliding time series.
    X beeing a list of time series with w_x points
    Y beeing a list of X following time series with w_y points
    One pair x and y will be dropped if one of those or both do not present all
    same temporal deltas of 1h, meaning that time series without uniform sample rate are dropped.

    Args:
        df (pd.DataFrame): [description]
        w_x (int): [description]
        w_y (int): [description]

    Returns:
        tuple: [description]
    """

    X = []
    Y = []

    df.dropna(inplace=True)
    df['is_good_delta'] = df['Datetime'].diff() == pd.to_timedelta('1h')
    df.set_index('is_good_delta', inplace=True)
    del df['Datetime']
    df = df.astype(float)
    
    for i in range(len(df) - w_x - w_y):
        df_slice = df.iloc[i: i + w_x + w_y]
        if all(df_slice.index): # check if all time deltas are 1h for that w_x+w_y time series
            X.append([row[X_columns].to_list() for _, row in df_slice.iloc[: w_x].iterrows()])
            Y.append([row[Y_columns].to_list() for _, row in df_slice.iloc[w_x: ].iterrows()])

    return X, Y