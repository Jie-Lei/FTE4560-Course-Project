# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:00:14 2021

@author: 24620
"""
import pandas as pd
import numpy as np
import sklearn as sk

path = "D://OneDrive - CUHK-Shenzhen//桌面//FTE4560//Project 2 of FTE4560//china stock market//"

data_1 = pd.read_csv(path+"000001.SS.csv")
data_2 = pd.read_csv(path+"399001.SZ.csv")

#%%
data_1['Date'] = pd.to_datetime(data_1['Date'])
data_2['Date'] = pd.to_datetime(data_2['Date'])

data_1 = data_1.set_index('Date')
data_2 = data_2.set_index("Date")
#%%
from sklearn.decomposition import FastICA

transformer = FastICA(n_components=5,random_state=0)

ss_ica = transformer.fit_transform(data_1.values)
sz_ica = transformer.fit_transform(data_2.values)
#%%
def plot_func(data,ica,title):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    plt.style.use('seaborn-white')
    plt.figure(figsize=(15,16))
    plt.suptitle(title,fontsize=18)
    plt.grid()  
    col = list(data)
    for i in range(ica.shape[1]):
        ax = plt.subplot(3,2,i+1)
        ax.plot(data.index,ica[:,i],c="#338844")
        ax.set_title("Component"+str(i),fontsize=14)
        ax.set_xlabel("Date",fontsize=14)
        ax.set_ylabel("Price",fontsize=14)
    plt.tight_layout()   
    plt.savefig(path+title+".jpg",dpi=400)
    plt.close()
#%%
plot_func(data_1,data_1.values,'000001.SS')
plot_func(data_1,ss_ica,'FastICA_000001.SS')
plot_func(data_2,data_2.values,'399001.SZ')
plot_func(data_2,sz_ica,'FastICA_399001.SZ')
#%%
data = pd.read_excel(path+"stack.xlsx")   
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index("Date")

col = list(data)

for i in col:
    x = data[i].values
    x = (x-x.min())/(x.max()-x.min())
    data[i] = x
#%%
transformer = FastICA(n_components=5,random_state=0)
ica_1 = transformer.fit_transform(data.values)

plot_func(data,ica_1,'Fast_ICA_5')
#%%
transformer = FastICA(n_components=3,random_state=0)
ica_2 = transformer.fit_transform(data.values)

plot_func(data,ica_2,'Fast_ICA_3')








