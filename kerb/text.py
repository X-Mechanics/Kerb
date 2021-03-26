import pandas as pd
import numpy as np
import os

def twitter():
    data_dir = './data'
    data_complaint = pd.read_csv(os.path.join(data_dir, 'complaint1700.csv'))
    data_non_complaint = pd.read_csv(os.path.join(data_dir, 'noncomplaint1700.csv'))
    data_complaint['label'] = 0
    data_non_complaint['label'] = 1

    # Concatenate complaining and non-complaining data
    data = pd.concat([data_complaint, data_non_complaint], axis=0).reset_index(drop=True)
    data.drop(['airline'], inplace=True, axis=1)
    return data