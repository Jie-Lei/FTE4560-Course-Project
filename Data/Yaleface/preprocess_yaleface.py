#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os 
import sys
import shutil
from PIL import Image

###################################### Organize all the preprocessing steps together ###################################
#%%
def preprocess(path_source, path_target, resize_pct = 0.2):
    '''
    param: {
        path_source (string): the path where images are originally placed
        path_target (string): the path where preprocessed images are stored
        resize_pct (float) : the percentage change of image size
    }
    
    output:{
        X (DataFrame): rescaled image matrix with size of (# of img size, # of samples)
        Y (DataFrame): '1-of-K' response matrix with size of (K, # of samples)
    }
    '''
    
    X_list = []
    Y_list = []
    K = 15 # there are 15 candidates
    i = 0 # column index of output matrix
    subject_index = {} # store the column indices for each individual
    
    # Create the folders if not already exists
    try:
        os.makedirs(path_target)
    except:
        pass
    

    for filename in os.listdir(path_source):
        if filename == '.DS_Store':  continue # hidden file for mac
    
        # Create an image copy to the new file
        source = os.path.join(path_source, filename)
        target = os.path.join(path_target, filename)
        try:
           shutil.copy(source, target)
        except IOError as e:
           print("Unable to copy file %s. %s" % (filename, e))
        except:
           print("Unexpected error:", sys.exc_info())
        
        # Rename the image
        info_list = filename.split('.')
        subject = info_list[0][-2:] # get the ID of the member (the last two numbers)
        if subject == '': continue # some tricks needed for special filenames
        if len(info_list) == 2 and filename.endswith('gif'): info_list[1] = 'centerlight'
        new_name = subject + info_list[1] + ".gif"
        file_path = path_target + new_name
        os.rename(path_target + filename, path_target + new_name)
        
        # Resize the images
        img = Image.open(file_path)
        wsize = int((float(img.size[0])*float(resize_pct)))
        hsize = int((float(img.size[1])*float(resize_pct)))
        img = img.resize((wsize,hsize), Image.ANTIALIAS)
        img.save(file_path)
        size = wsize*hsize
        
        # Flatten the image matrix into a vector
        im = np.array(img)
        im = np.ndarray.flatten(im)
        im_list = im.tolist()
        X_list.append(im_list)
        
        # Encode the response label by 1-of-K scheme
        subject = int(subject)
        y = [0]*K
        y[subject-1] = 1
        Y_list.append(y)
        
        # Store the file path into a dictionary
        if subject not in subject_index.keys():
            subject_index[subject] = []
        subject_index[subject] = subject_index[subject]+[i]
        
        i += 1
    
    # The shape of flattened data: (# of img size, # of samples)
    X_flatten = np.array(X_list).T
    Y_flatten = np.array(Y_list).T
    
    # Rescale the pixel value
    X_flatten = X_flatten / 255
    
    # Report some information
    print(f'SUCCESS: All images are resized into {img.size[0]}*{img.size[1]}')
         
    X = pd.DataFrame(X_flatten)
    Y = pd.DataFrame(Y_flatten)
    Y.index = range(1, len(Y)+1) # the index of Y matrix denotes the corresponding label
    
    return X, Y, subject_index


##################################### Randomly split data under specific requirement ###################################
#%%
def random_split(X, Y, index_dict, train_num = 4):
    '''
    X: (3-dimension array)
    Y: (3-dimension array)
    '''
    # statement
    print(f'We randomly select {train_num} images per individual to form a training set.')
    
    subject = X.shape[0]
    expression = X.shape[1]
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for i in range(subject):
        index = np.arange(expression)
        np.random.shuffle(index)
        shuffled_list = index.tolist()
        train = shuffled_list[:train_num]
        test = shuffled_list[train_num:]
        
        X_train.append(X[i,train,:].tolist())
        X_test.append(X[i,test,:].tolist())
        Y_train.append(Y[i,train,:].tolist())
        Y_test.append(Y[i,test,:].tolist())
        
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
        
    return X_train, Y_train, X_test, Y_test

##################################### Transform the image matrix into a 3-dimensional array ###################################
#%%
def two2three_dim(X, subject_index):
    X_ = []
    for i in range(15):
        index = subject_index[i+1]
        xi = np.array(X.iloc[:,subject_index[1]].T).tolist()
        X_.append(xi)
    X_ = np.array(X_)
    return X_


####################################### Load data from scratch or read .csv files ######################################
#%%
# When preprocessing the data, we can adjust the 'resize_pct'.
X, Y, subject_index  = preprocess('./yaleface/data/raw_img/', './yaleface/data/img/', resize_pct = 0.2)
X = two2three_dim(X, subject_index)
Y = two2three_dim(Y, subject_index)

# By random_split(), the data are splitted one times:
# When splitting the data, we decide 'train_num' to be 4 or 8.
X_train, Y_train, X_test, Y_test = random_split(X, Y, subject_index, train_num = 8)


print(X_train[:2]) # sample output (3-dimension array)

