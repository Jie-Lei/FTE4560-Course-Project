# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:42:23 2021

@author: 24620
"""

#%%
def train_test_split(data,n):
    '''
    data: DataFrame.shape(N,D+1)
    n: train for 11-n, test for n
    '''
    import pandas as pd
    import random
    
    samples = data.shape[0]/11

    count = 0
    test = []
    data_1 = data.copy()
    for i in list(range(int(samples))):
        l = list(range(count,11*(i+1)))
        count += 11
        x = random.sample(l,n)
        test = test + x 
    data_test = pd.DataFrame(data_1.iloc[test])
    data_train = pd.DataFrame(data_1.drop(index = data.index[test]))
    
    return data_train, data_test
