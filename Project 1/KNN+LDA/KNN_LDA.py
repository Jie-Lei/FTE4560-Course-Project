import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def covariance(data):           #Subfunction of solve_eigenvalue
    '''
    data: dataframe with (N,D+1)
    '''
    D = data.shape[1]-1
    S_w,S_b = np.zeros((D,D)),np.zeros((D,D))    
    m = (data.iloc[:,:-1].values.T).mean(axis=1)
    labels = list(set(data.iloc[:,-1].tolist()))
    for i in labels:
        subclass = data.loc[data.iloc[:,-1]==i] 
        subclass = subclass.iloc[:,:-1].values.T        #array.shape(D,n)
        S_w += S_within_k(subclass)
        S_b += S_between(subclass,m)
    S_w = pd.DataFrame(S_w).fillna(0).values
    S_b = pd.DataFrame(S_b).fillna(0).values
    return S_w,S_b

def S_within_k(subclass):       #Subfunction of covariance
    '''
    DataFrame.shape(D,n)
    '''
    x_1 = (subclass - subclass.mean(axis=1).reshape(-1,1))
    return np.matmul(x_1,x_1.T)

def S_between(subclass,m):      #Subfunction of covariance
    '''
    DataFrame.shape(D,n)
    m.shape(D,n)
    '''
    n = subclass.shape[1]
    x_2 = subclass.mean(axis=1) - m
    return n*np.matmul(x_2,x_2.T)

def solve_eigenvalue(data,d):   #Getting the W
    '''
    data: dataframe with (N,D+1)
    d: the target dimension
    '''
    S_w,S_b = covariance(data)
    '''
    Note that: for numpy.linalg. there are two functions to find the eigenvalue and eigenvector
        - np.linalg.eigh() used to solve Hermetian matrix,you can always get real number
        - np.linalg.eig() used to solve nonsymetric matirx
    eigh is more stable, better for LDA and PCA
    '''
    val,vec=np.linalg.eigh(np.dot(np.mat(S_w).I ,S_b))
    print(len(np.extract(val>0,val)))

    #K_largest eigenvector
    index_vec = np.argsort(-val)
    largest_index = index_vec[:d] 
    W = vec[:,largest_index]
    return W

def linear_DA(data,w):
    '''
    Data: array shape(N,D)
    w: array shape(D,d)
    '''
    x = np.dot(data,w)
    return np.array(x)  #array.shape(N,d)

def k_nearest(square,train_y,k=1):
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

def KNN(train_x,train_y,test_x,k):
    """
    data_train_x.array.shape(N,D)
    data_train_y.DataFrame.shape(N,)
    data_test.array.shape(N,D)
    """
    print('K = ',k)
    test = len(test_x) #get the column name list
    test_outcome = []
    test_x = pd.DataFrame(test_x)            #shape(N,D)
    train_x = train_x.T                 
    for i in tqdm(range(test)):
        data = test_x.iloc[i].values.reshape(-1,1)         #one test sample shape(D,1)
        square = ((train_x-data)**2).sum(axis=0).T         #Calculating shape(N,)
        c= k_nearest(square,train_y,k)     #The outcome of classification
        test_outcome.append(c)
    return test_outcome

def accuracy(outcome,y):
    y = y.tolist()
    rate = 0
    l = len(outcome)
    for i,j in zip(outcome,y):
        if i==j:
            rate += 1
    return str((rate/l)*100) + '%'

def train_test_split(data,n):
    '''
    data: DataFrame.shape(N,D+1)
    n: train for 11-n, test for n
    '''
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

def x_y_split(data):
    '''
    data: DataFrame.shape(N,D+1)
    '''
    return data.iloc[:,:-1].values,data.iloc[:,-1].values

def lda_knn(train_data,test_data,k,d,save):
    '''
    train_data: DataFrame.shape(N,D+1)
    test_data: Dataframe.shape(n,D+1)
    k: K argument for KNN
    lda: whether do lda befor KNN, default is False 
    '''
    train_x,train_y = x_y_split(train_data)   #x.array.shape(N,D) y.array.shape(N,)
    test_x,test_y = x_y_split(test_data)      #x.array.shape(n,D) y.array.shape(n,)

    w = solve_eigenvalue(train_data,d)                #Weight array shape(D,d)
    if save == True:
        np.save('LDA_face_weight.npy',w)
    train_x = linear_DA(train_x,w)                    #array.shape(N,d)
    test_x =  linear_DA(test_x,w)                     #array.shape(n,d)

    outcome_test = KNN(train_x,train_y,test_x,k)
    a = accuracy(outcome_test,test_y)
    return outcome_test,a

def knn(train_data,test_data,k):
    '''
    train_data: DataFrame.shape(N,D+1)
    test_data: Dataframe.shape(n,D+1)
    k: K argument for KNN
    lda: whether do lda befor KNN, default is False 
    '''
    train_x,train_y = x_y_split(train_data)   #x.array.shape(N,D) y.array.shape(N,)
    test_x,test_y = x_y_split(test_data)      #x.array.shape(n,D) y.array.shape(n,)

    outcome_test = KNN(train_x,train_y,test_x,k)
    a = accuracy(outcome_test,test_y)
    return outcome_test,a

if __name__ == "__main__":

    bank = False
    if bank == True:
        path = 'C://Users//24620//Desktop//FTE4560//Project 1 of FTE4560//data_knn_lda_JL//bankruptcy//data//'
        train_data = pd.read_csv(path+"bankruptcy_train.csv") #DataFrame.shape(N,D+1)
        test_data = pd.read_csv(path+"bankruptcy_test.csv")   #Dataframe.shape(n,D+1)
        #LDA+KNN
        outcome_bank_lda, bank_lda = lda_knn(train_data,test_data,k=19,d=1,save=True) #k=19,78%
        print("Bankruptcy LDA+KNN: ",bank_lda)
        #KNN
        outcome_bank_knn, bank_knn= knn(train_data,test_data,k=9)    #K=9,69%
        print("Bankruptcy KNN: ",bank_knn)

    else:
        path = 'C://Users//24620//Desktop//FTE4560//Project 1 of FTE4560//data_knn_lda_JL//yaleface//data//'
        data_x = pd.read_csv(path+"X.csv").T                  #DataFrame.shape(N,D)
        data_y = pd.read_excel(path+"Y.xlsx",sheet_name='Y')  #Dataframe.shape(N,1)
        data_x['y'] = data_y.values                           #DataFrame.shape(N,D+1)
        data = data_x.sort_values(by='y',ascending=True)      #DataFrame.shape(N,D+1)
        train_data, test_data = train_test_split(data,3)      #Split by 7-3 (N,D+1)
        #LDA+KNN
        outcome_face_lda, face_lda = lda_knn(train_data,test_data,k=1,d=14,save=True) #K=7,62.2%

        print("Yaleface LDA+KNN: ",face_lda)
        #KNN
        outcome_face_knn, face_knn = knn(train_data,test_data,k=5) #K=5, 68.9%
        print("Yaleface KNN: ",face_knn)
    
        

