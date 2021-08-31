#!/usr/bin/env python3

# Get necessary packages ready and let's set sail!!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################################### Detect extreme samples based on outliers #######################################
# %%

def outliers_sample(X):
    X.replace('?', 0, inplace=True)

    def fliers_index(n, outliers):
        index_list = []
        indices = [X[np.array(X['Attr%s' % (n)], dtype='float64') == outlier].index.to_list()[0] \
                   for outlier in outliers]
        for index in indices:
            index_list.append(index)
        return index_list

    fliers_list = []
    fliers_list_2 = []
    for i in range(1, X.shape[1] + 1):
        # The Box Plot Method:
        dict_ = plt.boxplot(X['Attr%s' % (i)].astype(np.float64))
        outliers = [i.get_ydata() for i in dict_['fliers']][0]
        fliers_list = fliers_list + fliers_index(i, outliers)
        if i % 8 == 0: print('The Box Plot Method:', 'Attr%s |' % (i), ' completed.')

        # The 3 Sigma Method:
        col = X['Attr%s' % (i)].astype(np.float64)
        mean = col.mean()
        sigma = np.sqrt(sum((col - mean) ** 2) / (len(col) - 1))
        fliers_list_2 = fliers_list_2 + np.where((col > 3 * sigma) | (col < -3 * sigma))[0].tolist()
        if i % 8 == 0: print('The 3 Sigma Method:', 'Attr%s |' % (i), ' completed.')

        # The Box Plot Method:
    bincount = np.bincount(np.array(fliers_list))  # 949个sample在fliers中的出现频率
    len(bincount)
    fliers_indices = np.argsort(-bincount)  # 出现频率的index降序
    ten_fliers1 = fliers_indices[:10]  # 选取10个最严重的samples (降序)

    # The 3 Sigma Method:
    bincount_2 = np.bincount(np.array(fliers_list_2))  # 949个sample在fliers中的出现频率
    fliers_indices_2 = np.argsort(-bincount_2)  # 出现频率的index降序
    ten_fliers2 = fliers_indices_2[:10]  # 选取10个最严重的samples （降序）

    common_samples = []
    for fliers in ten_fliers1:
        if fliers in ten_fliers2: common_samples.append(fliers)

    return common_samples


##################################### Fill in the missing values based on data type ################################
# %%

def impute(X, method='median', data_type='train', substitutes=None):
    col_names = X.columns
    X = np.array(X.replace('?', np.NaN).astype(np.float64))
    print('Before:', np.isnan(X).sum().sum())

    if data_type == 'train':
        if method == 'median':
            substitutes = np.nanmedian(X, axis=0).tolist()
        elif method == 'mean':
            substitutes = np.nanmean(X, axis=0).tolist()

        values = dict(zip([i for i in range(X.shape[1])], substitutes))  # values need to be a dictionary
        X = pd.DataFrame(X)
        X.fillna(value=values, inplace=True)
        print('After', X.isna().sum().sum())
        X.columns = col_names
        return substitutes, X

    elif data_type == 'test':
        values = dict(zip([i for i in range(X.shape[1])], substitutes))  # values need to be a dictionary
        X = pd.DataFrame(X)
        X.fillna(value=values, inplace=True)
        print('After', X.isna().sum().sum())
        X.columns = col_names
        return X

    else:
        print('Error: The param \'data_type\' need to be \'train\' or \'test\'.')

    return None



##################################### Scale the feature values before modeling ##########################################
#%%

def scaling(X, method='normal'):
    mean = X.mean()
    if method == 'normal':
        X = (X - mean) / np.std(X, ddof=1)
    elif method == 'length':
        X = (X - mean) / np.sqrt(np.sum((X - mean) ** 2))
    return X



##################################### Scale the feature values before modeling ##########################################
# %%
train = pd.read_csv('./data/training.csv')
test = pd.read_csv('./data/testing.csv')
train_X = train.iloc[:, :train.shape[1] - 1]
test_X = test.iloc[:, :test.shape[1] - 1]

# delete samples
train_X_copy = train_X.copy()
common_samples = outliers_sample(train_X_copy)
train_X.drop(common_samples, inplace=True)

# impute
col_mean, train_X = impute(train_X)
test_X = impute(test_X, data_type='test', substitutes=col_mean)

# scale
train_X = scaling(train_X)
test_X = scaling(test_X)

train_Y = train.iloc[:,train.shape[1]-1]
test_Y = test.iloc[:,test.shape[1]-1]

print(train_X.head())

