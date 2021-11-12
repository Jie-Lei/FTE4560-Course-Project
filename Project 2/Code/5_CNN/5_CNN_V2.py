# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:37:33 2021

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

def data_transfer(data,n):
    N = data.shape[0]
    df,date = supervised_transform(data[['Close']],n)
    y = df.iloc[:,[n]].values
    temp = np.zeros((1,n,5))
    for i in range(N):
        if i+n+1 >= N:
            break
        sample = data.iloc[i:i+n+1,:]
        x = sample.iloc[:n,:].values
        x = x.reshape(1,n,5)  
        temp = np.concatenate((temp,x),axis=0)
    temp = temp[1:,:,:]             
    return temp,y,date

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
    j = np.mean(((in_norm(x)-in_norm(y))**2))**(0.5)
    return j

def in_norm(x):
    global min_c
    global max_c
    return (x+min_c)*(max_c-min_c)

def index(x_train,x_test):
    y_pre_train = CNN.predict(x_train)
    
    #dtw_train = dtw(y_pre_train,y_train)
    cor_train = corre(y_pre_train,y_train)
    rmse_train = rmse(y_pre_train,y_train)
    
    train = np.array([cor_train,rmse_train])
    
    y_pre_test = CNN.predict(x_test)
    
    #dtw_test = dtw(y_pre_test,y_test)
    cor_test = corre(y_pre_test,y_test)
    rmse_test = rmse(y_pre_test,y_test)
    
    test = np.array([cor_test,rmse_test])
    
    df = pd.DataFrame([train,test])
    df.columns = ['Correlation','RMSE']
    df = df.T
    df.columns = ['Train','Test']
    df.to_excel(path_1+'CNN_5_V'+str(version)+'.xlsx')
    return df
#%%
version = 2
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
n = 5
x,y,date = data_transfer(data,n)

x_train = x[:434-n-1,:,:]
y_train = y[:434-n-1,:]
x_test = x[434-n-1:,:,:]
y_test = y[434-n-1:,]

CNN = keras.Sequential([
    layers.Conv1D(32,3,activation='relu',input_shape=(x.shape[1],5)),
    layers.Conv1D(16,3,activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(100,activation='linear'),
    layers.Dense(1,activation='linear'),
    ])
print(CNN.summary())

CNN.compile(optimizer='adam', loss="mse")
 
BATCH_SIZE = 16
EPOCHS = 200

history = CNN.fit(x_train,
                  y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)
#%%
y_pre_train = CNN.predict(x_train)
y_pre_test = CNN.predict(x_test)

y_pre = np.concatenate((y_pre_train,y_pre_test),axis=0)

plt.figure(figsize=(12,5))
plt.plot(date,in_norm(y_pre),label='Predicted')
plt.plot(date,in_norm(y),label='GroundTruth')
plt.axvline(t0,ls='--',c="#338844")
plt.legend()
plt.savefig(path_1+'CNN_5_V'+str(version)+'.jpg',dpi=400)
#%%
date_test = date[434-n-1:].values
plt.figure(figsize=(12,5))
plt.plot(date_test,in_norm(y_pre_test),label='Predicted')
plt.plot(date_test,in_norm(y_test),label='GroundTruth')
plt.legend()
plt.savefig(path_1+'CNN_5_test_V'+str(version)+'.jpg',dpi=400)
#%%
date_train = date[:434-n-1].values
plt.figure(figsize=(12,5))
plt.plot(date_train,in_norm(y_pre_train),label='Predicted')
plt.plot(date_train,in_norm(y_train),label='GroundTruth')
plt.legend()
plt.savefig(path_1+'CNN_5_train_V'+str(version)+'.jpg',dpi=400)

#%%
df = index(x_train,x_test)
#%%
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%%
