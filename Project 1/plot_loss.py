# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:54:40 2021

@author: 24620
""" 
def plot_loss(all_loss):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(all_loss)), all_loss)
    plt.title("Development of loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()