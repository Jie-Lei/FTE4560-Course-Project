# -*- coding: utf-8 -*-
"""
Created on Sat May  1 03:25:42 2021

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
        if i+n+1 >= data.shape[0]:
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

def index(y_test,y_train):
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
    #df.to_excel(path_1+'CNN_5_V'+str(version)+'.xlsx')
    return df

def adf_test(data):
    import statsmodels.tsa.stattools as ts
    adftest = ts.adfuller(data)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res

def plot_func(y_pre,y_true,date):
    plt.figure(figsize=(12,5))
    plt.plot(date,in_norm(y_pre),label='Predicted')
    plt.plot(date,in_norm(y_true),label='GroundTruth')
    plt.axvline(t0,ls='--',c="#338844")
    plt.legend()
    
def loss(history): 
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
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
#Model 1: CNN (First strategy)
close = data[['Close']]
n = 5
df,date = supervised_transform(close,n)

x = df.iloc[:,:n]
y = df.iloc[:,[n]]

x_train = x.iloc[:434-n-1,:].values
y_train = y.iloc[:434-n-1,:].values
x_test = x.iloc[434-n-1:,:].values
y_test = y.iloc[434-n-1:,:].values

x_train = x_train.reshape(x_train.shape[0],x.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x.shape[1],1)

CNN_1 = keras.Sequential([
    layers.Conv1D(32,3,activation='relu',input_shape=(x.shape[1],1)),
    layers.Conv1D(16,3,activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(100,activation='linear'),
    layers.Dense(1,activation='linear'),
    ])
#print(CNN.summary())

CNN_1.compile(optimizer='adam', loss="mse")
 
BATCH_SIZE = 8
EPOCHS = 200
 
history_1 = CNN_1.fit(x_train,
                  y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)

y_pre_train = CNN_1.predict(x_train)
y_pre_test = CNN_1.predict(x_test)
y_pre = in_norm(np.concatenate((y_pre_train,y_pre_test),axis=0))
plot_func(y_pre,y.values,date)

date_test = date[434-n-1:].values
plot_func(y_pre_test,y_test,date_test)
index_test = index(y_pre_test,y_test)

date_train = date[:434-n-1].values
plot_func(y_pre_train,y_train,date_train)
index_train = index(y_pre_train,y_train)

loss(history_1)
#%%
#Model 2: CNN (Second strategy)
n = 5
x,y,date = data_transfer(data,n)

x_train = x[:434-n-1,:,:]
y_train = y[:434-n-1,:]
x_test = x[434-n-1:,:,:]
y_test = y[434-n-1:,]

CNN_2 = keras.Sequential([
    layers.Conv1D(32,3,activation='relu',input_shape=(x.shape[1],5)),
    layers.Conv1D(16,3,activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(100,activation='linear'),
    layers.Dense(1,activation='linear'),
    ])
#print(CNN_2.summary())

CNN_2.compile(optimizer='adam', loss="mse")
 
BATCH_SIZE = 16
EPOCHS = 200

history_2 = CNN_2.fit(x_train,
                  y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)
#%%
y_pre_train = CNN_2.predict(x_train)
y_pre_test = CNN_2.predict(x_test)
y_pre = in_norm(np.concatenate((y_pre_train,y_pre_test),axis=0))
plot_func(y_pre,y.values,date)

date_test = date[434-n-1:].values
plot_func(y_pre_test,y_test,date_test)
index_test = index(y_pre_test,y_test)

date_train = date[:434-n-1].values
plot_func(y_pre_train,y_train,date_train)
index_train = index(y_pre_train,y_train)

loss(history_2)
#%%
#Model 3: CNN (Third strategy)
close = data[['Close']]
n = 15
df,date = supervised_transform(close,n)

x = df.iloc[:,:n]
y = df.iloc[:,[n]]

x_train = x.iloc[:434-n-1,:].values
y_train = y.iloc[:434-n-1,:].values
x_test = x.iloc[434-n-1:,:].values
y_test = y.iloc[434-n-1:,:].values

x_train = x_train.reshape(x_train.shape[0],x.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x.shape[1],1)

CNN_3 = keras.Sequential([
    layers.Conv1D(32,3,activation='relu',input_shape=(x.shape[1],1)),
    layers.Conv1D(16,3,activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(100,activation='linear'),
    layers.Dense(1,activation='linear'),
    ])

CNN_3.compile(optimizer='adam', loss='mse')
 
BATCH_SIZE = 8
EPOCHS = 500
 
history_3 = CNN_3.fit(x_train,
                  y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)

y_pre_train = CNN_3.predict(x_train)
y_pre_test = CNN_3.predict(x_test)
y_pre = in_norm(np.concatenate((y_pre_train,y_pre_test),axis=0))
plot_func(y_pre,y.values,date)

date_test = date[434-n-1:].values
plot_func(y_pre_test,y_test,date_test)
index_test = index(y_pre_test,y_test)

date_train = date[:434-n-1].values
plot_func(y_pre_train,y_train,date_train)
index_train = index(y_pre_train,y_train)

loss(history_3)
#%%
#Model 4: CNN (Fourth strategy)
n = 15
x,y,date = data_transfer(data,n)

x_train = x[:434-n-1,:,:]
y_train = y[:434-n-1,:]
x_test = x[434-n-1:,:,:]
y_test = y[434-n-1:,]

CNN_4 = keras.Sequential([
    layers.Conv1D(64,3,activation='relu',input_shape=(x.shape[1],5)),
    layers.Conv1D(32,3,activation='relu'),
    layers.Conv1D(64,2,activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64,activation='linear'),
    layers.Dense(1,activation='linear'),
    ])

CNN_4.compile(optimizer='adam', loss="mse")
 
BATCH_SIZE = 16
EPOCHS = 300
 
history_4 = CNN_4.fit(x_train,
                  y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)

y_pre_train = CNN_4.predict(x_train)
y_pre_test = CNN_4.predict(x_test)
y_pre = in_norm(np.concatenate((y_pre_train,y_pre_test),axis=0))
plot_func(y_pre,y.values,date)

date_test = date[434-n-1:].values
plot_func(y_pre_test,y_test,date_test)
index_test = index(y_pre_test,y_test)

date_train = date[:434-n-1].values
plot_func(y_pre_train,y_train,date_train)
index_train = index(y_pre_train,y_train)

loss(history_4)
#%%
#Model 5: CNN_1+CNN_2 (First strategy)
adf = pd.DataFrame(adf_test(close.values))
diff = np.diff(close.values.flatten())
adf_diff = pd.DataFrame(adf_test(diff))

n = 5
df,date = supervised_transform(close,n)

x = df.iloc[:,:n]
y = df.iloc[:,[n]]

x_train_1 = x.iloc[:434-n-1,:].values
y_train_1 = y.iloc[:434-n-1,:].values
x_test_1 = x.iloc[434-n-1:,:].values
y_test_1 = y.iloc[434-n-1:,:].values

x_train_1 = x_train_1.reshape(x_train_1.shape[0],x_train_1.shape[1],1)
x_test_1 = x_test_1.reshape(x_test_1.shape[0],x_train_1.shape[1],1)

CNN_5_1 = keras.Sequential([
    layers.Conv1D(32,3,activation='relu',input_shape=(x_train_1.shape[1],1)),
    layers.Conv1D(16,3,activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(100,activation='linear'),
    layers.Dense(1,activation='linear'),
    ])

CNN_5_1.compile(optimizer='adam', loss='mse')
 
BATCH_SIZE = 8
EPOCHS = 200
 
history_5_1 = CNN_5_1.fit(x_train_1,
                  y_train_1,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)

x_train_2 = np.diff(x_train_1,axis=0)
y_train_2 = np.diff(y_train_1,axis=0)
x_test_2 = np.diff(x_test_1,axis=0)
y_test_2 = np.diff(y_test_1,axis=0)

x_train_2 = x_train_2.reshape(x_train_2.shape[0],x_train_2.shape[1],1)
x_test_2 = x_test_2.reshape(x_test_2.shape[0],x_test_2.shape[1],1)

CNN_5_2 = keras.Sequential([
    layers.Conv1D(32,3,activation='relu',input_shape=(x_train_2.shape[1],1)),
    layers.Conv1D(16,3,activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(100,activation='linear'),
    layers.Dense(1,activation='linear'),
    ])

CNN_5_2.compile(optimizer='adam', loss='mse')
 
BATCH_SIZE = 8
EPOCHS = 200
 
history_5_2 = CNN_5_2.fit(x_train_2,
                  y_train_2,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)

y_pre_train_1 = CNN_5_1.predict(x_train_1).flatten()
y_pre_train_2 = np.insert(CNN_5_2.predict(x_train_2),0,0)
y_pre_train = y_pre_train_1 + y_pre_train_2

y_pre_test_1 = CNN_5_1.predict(x_test_1).flatten()
y_pre_test_2 = np.insert(CNN_5_2.predict(x_test_2),0,0)
y_pre_test = y_pre_test_1 + y_pre_test_2

y_pre = np.concatenate((y_pre_train,y_pre_test),axis=0)
plot_func(y_pre,y.values,date)

date_test = date[434-n-1:].values
plot_func(y_pre_test,y_test,date_test)
index_test = index(y_pre_test,y_test)

date_train = date[:434-n-1].values
plot_func(y_pre_train,y_train,date_train)
index_train = index(y_pre_train,y_train)

loss(history_5_1)
loss(history_5_2)
#%%
#Model 6: CNN+LSTM (Second strategy)
n = 5
x,y,date = data_transfer(data,n)

x_train = x[:434-n-1,:,:]
y_train = y[:434-n-1,:]
x_test = x[434-n-1:,:,:]
y_test = y[434-n-1:,]

CNN_6 = keras.Sequential([
    layers.Conv1D(64,2,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])),
    layers.Conv1D(128,2,activation='relu'),
    layers.MaxPooling1D(3),
    layers.Dropout(0.25),
    layers.LSTM(128),
    layers.Dense(1,activation='linear')
    ])

CNN_6.compile(optimizer='adam', loss='mse')
 
BATCH_SIZE = 4
EPOCHS = 300

history_6 = CNN_6.fit(x_train,
                  y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS)

y_pre_train = CNN_6.predict(x_train)
y_pre_test = CNN_6.predict(x_test)
y_pre = in_norm(np.concatenate((y_pre_train,y_pre_test),axis=0))
plot_func(y_pre,y.values,date)

date_test = date[434-n-1:].values
plot_func(y_pre_test,y_test,date_test)
index_test = index(y_pre_test,y_test)

date_train = date[:434-n-1].values
plot_func(y_pre_train,y_train,date_train)
index_train = index(y_pre_train,y_train)

loss(history_6)
#%%