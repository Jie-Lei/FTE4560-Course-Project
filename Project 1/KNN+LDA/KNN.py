# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:07:22 2021

@author: 24620
"""
import numpy as np
import pandas as pd
import sys
path = "D://OneDrive - CUHK-Shenzhen//桌面//FTE4560_Code//"
sys.path.append(path)
import train_test_split
#%%
#import data
#shape (N,D) or (N,D+1) 
#trian, test may not be divided
#%%
class K_Nearest_Neighbour:
    def __init__(self,k=2):
        self.k=k
        
    def k_nearest(self,square,train_y,k=1):
        distance = pd.DataFrame(square,columns = ['distance'])
        #Creat a dataframe (distance,train_y),then sort it
        distance['label'] = train_y
        #From small to big distance
    
        distance = distance.sort_values(by='distance', ascending=True)
    
        #The k smallest distance
        k_distance = distance[:k]
        #Count the number of these k smallest distance
        count_class = pd.DataFrame(k_distance['label'].value_counts()).T
        return list(count_class)[0]
    
    def KNN(self,train_x,train_y,test_x,k):
        """
        train_x.array.shape(N,D)
        train_y.DataFrame.shape(N,)
        test.array.shape(N,D)
        """
        from tqdm import tqdm
        print('K = ',k)
        test = len(test_x) #get the column name list
        test_outcome = []
        test_x = pd.DataFrame(test_x)            #shape(N,D)
        train_x = train_x.T                 
        for i in tqdm(range(test)):
            data = test_x.iloc[i].values.reshape(-1,1)         #one test sample shape(D,1)
            square = ((train_x-data)**2).sum(axis=0).T         #Calculating shape(N,)
            c= self.k_nearest(square,train_y,k)     #The outcome of classification
            test_outcome.append(c)
        return test_outcome
#%%
x = np.array(list(range(9))).reshape(3,3)

t = np.array(list(range(6))).reshape(2,3)

#%%
t2 = t.repeat(x.shape[0])
#%%
t3 = t.reshape(2,1,3)













