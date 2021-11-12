# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:03:47 2021

@author: 24620
"""
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def normalization(data):
    col = list(data)
    for i in col:
        x = data[i].values
        x = (x-x.min())/(x.max()-x.min())
        data[i] = x
    return data

def supervised_transform(data,n):
    df = pd.DataFrame(np.zeros((n+1,)))
    for i in range(data.shape[0]):
        if i+n+1 >= close.shape[0]:
            break
        x = data.iloc[i:i+n+1,:].values
        df[str(i)] = x#.reset_index(drop=True)
    df = df.iloc[:,1:].reset_index(drop=True).T
    date = data.index[n:-1]
    df['Date'] = date
    df = df.set_index('Date')
    col = []
    for i in range(-n,1):
        col.append(str(i))
    df.columns = col
    return df,date

def dtw(x,y):
    from dtw import accelerated_dtw
    x = in_norm(x)
    y = in_norm(y)
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(x,y, dist='euclidean')
    return d
def corre(x,y):
    j = np.corrcoef(x.flatten(),y.flatten())
    return j[0,1]
def rmse(x,y):
    x = np.array(x)
    y = np.array(y)
    j = np.mean(((in_norm(x)-in_norm(y))**2))**(0.5)
    return j

def in_norm(x):
    global min_c
    global max_c
    return (x+min_c)*(max_c-min_c)

def index(y_train,y_test):
    global y_trian_1
    global y_test_1
    cor_train = corre(y_train_1,y_train)
    rmse_train = rmse(y_train_1,y_train)
    
    train = np.array([cor_train,rmse_train])

    cor_test = corre(y_test_1,y_test)
    rmse_test = rmse(y_test_1,y_test)
    
    test = np.array([cor_test,rmse_test])
    
    df = pd.DataFrame([train,test])
    df.columns = ['Correlation','RMSE']
    df = df.T
    df.columns = ['Train','Test']
    df.to_excel(path_1+'CNN_5_V'+str(version)+'.xlsx')
    return df

#%%
version = 3
#%%
path = "D://OneDrive - CUHK-Shenzhen//桌面//FTE4560//Project 2 of FTE4560//china stock market//"
path_1 = 'D://OneDrive - CUHK-Shenzhen//桌面//FTE4560//Project 2 of FTE4560//Results//CNN_5//'
data = pd.read_csv(path+"399001.SZ.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index("Date")
close = data[['Close']]
min_c = np.min(close.values)
max_c = np.max(close.values)
data = normalization(data)

t0 = data.index[433]
#%%
close = data[['Close']]
#%%
def adf_test(data):
    import statsmodels.tsa.stattools as ts
    adftest = ts.adfuller(data)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res
#%%
c = pd.DataFrame(adf_test(close.values))
#c.to_excel(path_1+'adf.xlsx')
#%%
diff = np.diff(close.values.flatten())
d = pd.DataFrame(adf_test(diff))
#d.to_excel(path_1+'adf_diff.xlsx')
#%%
n = 5
df,date = supervised_transform(close,n)

x = df.iloc[:,:n]
y = df.iloc[:,[n]]

x_train_1 = x.iloc[:434-n-1,:].values
y_train_1 = y.iloc[:434-n-1,:].values
x_test_1 = x.iloc[434-n-1:,:].values
y_test_1 = y.iloc[434-n-1:,:].values
#%%
x_train_1 = x_train_1.reshape(x_train_1.shape[0],x_train_1.shape[1],1)
x_test_1 = x_test_1.reshape(x_test_1.shape[0],x_train_1.shape[1],1)

CNN_1 = keras.Sequential([
    layers.Conv1D(32,3,activation='relu',input_shape=(x_train_1.shape[1],1)),
    layers.Conv1D(16,3,activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(100,activation='linear'),
    layers.Dense(1,activation='linear'),
    ])
print(CNN_1.summary())
#%%
def rmse_2(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

CNN_1.compile(optimizer='adam', loss=[rmse_2])
 
BATCH_SIZE = 8
EPOCHS = 200
 
history = CNN_1.fit(x_train_1,
                  y_train_1,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)
#%%
x_train_2 = np.diff(x_train_1,axis=0)
y_train_2 = np.diff(y_train_1,axis=0)
x_test_2 = np.diff(x_test_1,axis=0)
y_test_2 = np.diff(y_test_1,axis=0)
#%%
x_train_2 = x_train_2.reshape(x_train_2.shape[0],x_train_2.shape[1],1)
x_test_2 = x_test_2.reshape(x_test_2.shape[0],x_test_2.shape[1],1)

CNN_2 = keras.Sequential([
    layers.Conv1D(32,3,activation='relu',input_shape=(x_train_2.shape[1],1)),
    layers.Conv1D(16,3,activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(100,activation='linear'),
    layers.Dense(1,activation='linear'),
    ])
print(CNN_2.summary())
#%%
def rmse_2(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

CNN_2.compile(optimizer='adam', loss=[rmse_2])
 
BATCH_SIZE = 8
EPOCHS = 200
 
history = CNN_2.fit(x_train_2,
                  y_train_2,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)
#%%
y_pre_train_1 = CNN_1.predict(x_train_1).flatten()
y_pre_train_2 = np.insert(CNN_2.predict(x_train_2),0,0)
y_pre_train = y_pre_train_1 + y_pre_train_2

plt.figure(figsize=(12,5))
date_train = date[:434-n-1].values
plt.plot(date_train,in_norm(y_pre_train),label='Predicted')
plt.plot(date_train,in_norm(y_train_1),label='GroundTruth')
#plt.axvline(t0,ls='--',c="#338844")
plt.legend()
plt.savefig(path_1+'CNN_5_train_V'+str(version)+'.jpg',dpi=400)
#%%
y_pre_test_1 = CNN_1.predict(x_test_1).flatten()
y_pre_test_2 = np.insert(CNN_2.predict(x_test_2),0,0)
y_pre_test = y_pre_test_1 + y_pre_test_2

plt.figure(figsize=(12,5))
date_test = date[434-n-1:].values
plt.plot(date_test,in_norm(y_pre_test),label='Predicted')
plt.plot(date_test,in_norm(y_test_1),label='GroundTruth')
#plt.axvline(t0,ls='--',c="#338844")
plt.legend()
plt.savefig(path_1+'CNN_5_test_V'+str(version)+'.jpg',dpi=400)
#%%
y_pre = np.concatenate((y_pre_train,y_pre_test),axis=0)
plt.figure(figsize=(12,5))
plt.plot(date,in_norm(y_pre),label='Predicted')
plt.plot(date,in_norm(y.values),label='GroundTruth')
plt.axvline(t0,ls='--',c="#338844")
plt.legend()
plt.savefig(path_1+'CNN_5_V'+str(version)+'.jpg',dpi=400)
#%%
df = index(y_pre_train,y_pre_test)
#%%
plt.savefig(path_1+'CNN_5_V'+str(version)+'.jpg',dpi=400)
#%%
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
