import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
'''
def train_test_split(data,n):
    ''''''
    data: DataFrame.shape(N,D+1)
    n: train for 11-n, test for n
    ''''''
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
#Transfer the label into the vector form
def one_hot(y_data, n_samples, n_classes):
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y_data.T] = 1
    return one_hot
  
def plot_loss(all_loss):
    #fig = plt.figure(figsize=(8,5))
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(all_loss)), all_loss)
    plt.title("Development of loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()
'''
import sys
func_path = "D://OneDrive - CUHK-Shenzhen//桌面//FTE4560_Code//"
sys.path.append(func_path)
import one_hot
import train_test_split
import plot_loss


class Neural_Network:

    def __init__(self, alpha = 0.01, lam = 0.01,loss_func='MSE'):
        self.arch = list()
        self.n_hidden = 0       # 默认取 0 个隐藏层
        self.alpha = alpha      # 学习率
        self.lam = lam          # 正则化项系数
        self.weights = list()   # 初始化权值矩阵
        self.gradients = list() # 初始化梯度矩阵
        self.bias = 1           # 偏置项
        self.loss_func = loss_func

    def add_input_layer(self,input_shape):
        '''
         - tuple(input_shape): train_data.shape(n,D)
        '''
        self.arch.append(input_shape)

    def add_hidden_layer(self,func,neurons):
        '''
         - str(func): activation func
         - int(neurons): num of neurons
         - int(layer): rank in hidden layer
        '''
        tup = tuple([func,neurons])  
        self.arch.append(tup)
    
    def add_output_layer(self,func,y_class):
        '''
         - str(func): activation func
         - int(neurons): num of neurons
        '''
        tup = tuple([func,y_class])
        self.arch.append(tup)
        self.n_hidden = len(self.arch)-2

    def init_weights(self):
        '''
        把参数存在列表当中，每一层参数都存成数组！加入了偏执
        ''' 
        # Every layer's weight matrix
        '''
        for i in range(self.n_hidden + 1):
            self.weights.append(
                np.random.normal(0,0.1,size = (self.arch[i+1][1], self.arch[i][1] + 1)) 
                )#shape(num_neurons,input+1)
        '''
        for i in range(self.n_hidden + 1):
            self.weights.append(
                np.random.rand(self.arch[i+1][1], self.arch[i][1] + 1) *np.sqrt(1/(self.arch[i+1][1]-1)) 
                )#shape(num_neurons,input+1)
                 
    def init_gradients(self):
        '''
        初始化梯度，每一个参数都有一个初始梯度值为0
        '''
        # 每一层的梯度矩阵
        for i in range(self.n_hidden + 1):
            self.gradients.append(np.zeros((self.arch[i+1][1], self.arch[i][1] + 1)))

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def deriv_sigmoid(self,sigmoid):
        return sigmoid*(1-sigmoid)

    def tanh(self,x):
        return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

    def deriv_tanh(self,tanh):

        return 1-tanh**2

    def softmax(self,x):
        return np.exp(x)/(np.exp(x).sum(axis=1).reshape(-1,1))

    def MSE_loss(self,y,y_pre):
        #Mean of square error
        return ((y-y_pre)**2).sum()

    def cross_entropy(self,y,y_pre):
        return -(y * np.log(y_pre+0.00001) + (1-y) * np.log(1-y_pre+0.00001)).sum()

    def activating(self,act_func,activation):
        if act_func =='sigmoid':
            return self.sigmoid(activation)
        elif act_func == 'tanh':
            return self.tanh(activation)
        elif act_func == 'softmax':
            return self.softmax(activation)

    def deriv_acivation(self,act_func,activation):
        if act_func =='sigmoid':
            return self.deriv_sigmoid(activation)
        elif act_func == 'tanh':
            return self.deriv_tanh(activation)

    def train(self, x_data, y_data,test_x,test_y,epochs):
        '''
        x_data: np.array(N,D)
        y_data: np.array(N,1).reshape(-1,1)
        '''
        num_samples  = self.arch[0][0]
        #num_features = self.arch[0][1]
        #Get the classes number of y_data
        num_classes = self.arch[-1][1] 
        #Rewrite y_data as a vector  
        y_one_hot = one_hot(y_data, num_samples, num_classes)
        #Saving the loss of every epoch
        all_loss = list()
        #Initialization
        self.init_weights()
        self.init_gradients()
        
        for epoch in range(epochs):
            loss = 0
            layer_output = self.forward_propagation(x_data) #list with arrays (N,1+input)
            layer_error = self.cal_layer_error(layer_output, y_one_hot)
            self.cal_gradients(layer_output, layer_error)
            self.update_weights(num_samples)
            
            loss = (1 / num_samples) * self.cal_loss(layer_output, y_one_hot)
            all_loss.append(loss)
            
            test_pre = self.predict(test_x,test_y)
            test_acc = self.accuracy(test_pre,test_y)
            
            if epoch % 10 == 0:
                predict = np.argmax(layer_output[-1],axis=1).reshape(-1,1)
                acc = self.accuracy(predict,y_data)
                print('Epoch %d: train loss = %.4f, train = %.4f, test = %.4f'%(epoch,loss,acc,test_acc))
            
            if acc >= 90 and test_acc >= 90:
                predict = np.argmax(layer_output[-1],axis=1).reshape(-1,1)
                print('-'*15+'Early stop!'+'-'*15)
                acc = self.accuracy(predict,y_data)
                print('Epoch %d: train loss = %.4f, train = %.4f, test = %.4f'%(epoch,loss,acc,test_acc))
                break
            
        return self.weights, all_loss
    
    def forward_propagation(self, data):
        '''
        data.shape(N,D)
        '''
        layer_out = list()
        #插入偏执在数组的第0位
        N = data.shape[0]
        bias = np.ones(N)*(self.bias) # bias.shape(N,1)
        a = np.insert(data, 0, values = bias,axis=1) #a.shape(N,1+D)
        #存入输出列表作为输入层的输出
        layer_out.append(a)
        for i in range(self.n_hidden + 1):
            #self.weights[i].shape(num_neurons,1 + input)
            z = np.dot(a,self.weights[i].T)
            act = self.arch[i+1][0]
            a = self.activating(act,z)    
            #a.shape(N,num_neurons)
            if i != self.n_hidden:
                bias = np.ones(N)*(self.bias) # bias.shape(N,1)
                a = np.insert(a, 0, values = bias,axis=1) #a.shape(N,1+D)
            #a.shape(N,1 + num_neurons)
            layer_out.append(a)
        
        return layer_out

    def cal_layer_error(self, layer_output, y):
        # 只有第 2 →n 层有误差，输入层没有误差
        layer_error = list()
        # 计算输出层的误差
        error = layer_output[-1] - y
        layer_error.append(error)
        # 反向传播计算误差
        for i in range(self.n_hidden, 0, -1):
            act = self.arch[i][0]
            #self.weights[i] shape(K, 1+input) 
            #error.shape(N,K)
            error = np.dot(error, self.weights[i]) * self.deriv_acivation(act,layer_output[i])
            # 删除第一列，偏置项没有误差
            error = np.delete(error, 0,axis=1)
            layer_error.append(error)
        #翻转误差列表
        #return np.array(layer_error[::-1])  
        return layer_error[::-1]

    def cal_gradients(self, layer_output, layer_error):
        
        #self.gradents[l].shape(num_, 1 + input)
        for l in range(self.n_hidden + 1):
            '''
            for i in range(self.gradients[l].shape[0]):
                for j in range(self.gradients[l].shape[1]):
                    self.gradients[l][i][j] += layer_error[l][i] * layer_output[l][j]
            '''
            self.gradients[l] += np.dot(layer_error[l].T, layer_output[l])

    def update_weights(self, n_samples):
        for l in range(self.n_hidden + 1):
            #正则化
            gradient = 1.0 / n_samples * self.gradients[l] + self.lam * self.weights[l]
            #偏置项无需正则化
            gradient[:,0] -= self.lam * self.weights[l][:,0]
            #参数更新
            self.weights[l] -= self.alpha * gradient
   
    def cal_loss(self,output,y_one_hot):
        if self.loss_func == 'MSE':
            loss = self.MSE_loss(y_one_hot,output[-1])
        else:
            loss = self.cross_entropy(y_one_hot,output[-1])
        return loss

    def predict(self, test_x,test_y):
        n_samples = test_x.shape[0]

        output = self.forward_propagation(test_x)[-1]
        predict = np.argmax(output,axis=1)
        return predict.reshape(-1,1)    
    
    def accuracy(self,y_predict, test_y):
        if y_predict.shape != test_y.shape:
            y_predict = y_predict.reshape(-1,1)
        return np.sum(y_predict == test_y) / test_y.shape[0] *100
        
if __name__ == "__main__":
    random.seed(2021)
    bank = True
    if bank == True:
        path = 'C://Users//24620//Desktop//FTE4560//Project 1 of FTE4560//NN//bankruptcy//'

        train = pd.read_csv(path+'bankruptcy_train.csv') #shape(N,D)
        test = pd.read_csv(path+'bankruptcy_test.csv') #shape(N,D)
        
        x_train = np.array(train.iloc[:,:-1])
        y_train = np.array(train.iloc[:,-1]).reshape(-1,1)

        x_test = np.array(test.iloc[:,:-1])
        y_test = np.array(test.iloc[:,-1]).reshape(-1,1)
        #Num_classes
        K = len(set(y_test.flatten()))

        model = Neural_Network(alpha = 0.00001, lam = 0.01,loss_func='cross_entropy')
        #Bulid the Network
        model.add_input_layer(x_train.shape)
        model.add_hidden_layer('sigmoid',10)
        model.add_output_layer('softmax',K)
        #Train
        weights ,all_loss = model.train(x_train, y_train, x_test, y_test,epochs=6000)

        y_predict = model.predict(x_test,y_test)
        a = model.accuracy(y_predict,y_test)
        
        print('-'*41)
        print('Test accuracy: %.4f'%(a))
        #Plot loss function
        plot_loss(all_loss)

    else:
        path_1 = 'C://Users//24620//Desktop//FTE4560//Project 1 of FTE4560//data_knn_lda_JL//yaleface//data//'
        face_x = pd.read_csv(path_1+"X.csv").T          #(N,D)
        face_y = data_y = pd.read_excel(path_1+"Y.xlsx",sheet_name='Y')

        face_x['y'] = face_y.values                           #(N,D+1)
        face = face_x.sort_values(by='y',ascending=True)      #(N,D+1)
        train,test = train_test_split(face,3)      #Split by 4 as  return 11-n and n

        x_train = np.array(train.iloc[:,:-1])
        y_train = np.array(train.iloc[:,-1]).reshape(-1,1)

        x_test = np.array(test.iloc[:,:-1])
        y_test = np.array(test.iloc[:,-1]).reshape(-1,1)

        #Num_classes
        K = len(set(y_test.flatten()))

        model = Neural_Network(alpha = 0.01, lam = 0.1,loss_func='cross_entropy')
        model.add_input_layer(x_train.shape)
        model.add_hidden_layer('sigmoid',100)
        model.add_output_layer('softmax',K)

        weights ,all_loss = model.train(x_train, y_train, x_test, y_test,epochs=1000)

        y_predict = model.predict(x_test,y_test)
        a = model.accuracy(y_predict,y_test)
        print('-'*41)
        print('Test accuracy: %.4f'%(a))
        #Plot loss function
        plot_loss(all_loss)
