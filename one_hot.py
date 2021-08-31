# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:54:04 2021

@author: 24620
"""
def one_hot(y_data, n_samples, n_classes):
    import numpy as np
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y_data.T] = 1
    return one_hot