# -*- coding: utf-8 -*-
"""Boltzmann Machine.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JxIEe_TAcz-utfLKTqjbo3foOtqDGD0s

#Boltzmann Machine

##Downloading the dataset

###ML-100K
"""


"""##Importing the libraries"""

import numpy as np
import pandas as pd
import torch
## nn to implement neral networks
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
#for stochastic range and  descent
from torch.autograd import Variable

"""## Importing the dataset"""
movies = pd.read_csv('ml-1m/movies.dat',sep = '::',header = None, engine='python',
                     encoding='latin-1')
users  =  pd.read_csv('ml-1m/users.dat',sep = '::',header = None, engine='python',
                     encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::',header = None, engine='python',
                     encoding='latin-1')
"""## Preparing the training set and the test set"""
training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
test_set = np.array(test_set, dtype='int')

"""## Getting the number of users and movies"""
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies =  int(max(max(training_set[:,1]),max(test_set[:,1])))

"""## Converting the data into an array with users in lines and movies in columns"""
#users in lines (observations) and movies in columns (features)
#structure contain the observations wich will go to network, feature going to the input node
#torch expect a list
def convert(data):
    #every list contain the lines wich are the users and  
    #where for each ther is the ratings of the movies
    new_data = []
    for id_users in range(1,nb_users+1):
        id_movies = data[:,1][data[:,0]==id_users]
        id_ratings = data[:,2][data[:,0]==id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings 
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)       
    

"""## Converting the data into Torch tensors instead of Numpy array"""
#tensorflow expect a list of list 
#tensors are one class here they are floats
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

"""## Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)"""
# -1 the movie dosn't been rated
training_set[training_set == 0] = -1
# or dosn't work with pytorch
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1


test_set[test_set == 0] = -1
# or dosn't work with pytorch
test_set[test_set== 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1






"""## Creating the architecture of the Neural Network"""
# Bernouil RBM because we are predicting y/n movie
class RBM():
    #nv number of visible nodes = nobr of movies
    def __init__(self,nv,nh):
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
    #x is the visible neurns v in the p(h/v)
    def sample_h(self,x):
        #prodects of the weights and the neurons
        #torch.nm produit de deux matrice
        #expand_as(wx) : ensure that every bais is added to the mini batch
        wx = torch.mm(x,self.W.t())
        #activation function  : prob that the h  n will be activated
        activation = wx + self.a.expand_as(wx)
        #A vector of len(element) = hidden nodes = prob for each to be activatd
        p_h_given_v = torch.sigmoid(activation)
        #bernoulli sampling : sampels nerones activated if the random value > prob 
        return p_h_given_v, torch.bernoulli(p_h_given_v)
         #y is the hidden nodes
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
         # V0 : rating of all the movies by one user
          #vk : visible nodes after k samlings
           #ph0 : vector of probabalitieds that the first iteration the hidden nodes = 1 given the v0
           #phk : vector of probabalitieds after k iteration the hidden nodes = 1 given the vk
    def train(self,v0,vk,ph0,phk):
         self.W += (torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)).t()
         # ((v0 - vk),0)kieeping v as a tensor of 2 disminsions
         self.b += torch.sum((v0 - vk),0) 
         self.a += torch.sum((ph0 - phk),0) 
# training_set[0] first line 
# nv = nb_movies
#safer way : 
nv = len(training_set[0]) 
nh = 100 #nbr of features to detect
#updating weights by a batch of observation
batch_size = 100
rbm = RBM(nv,nh)
        

"""## Training the RBM"""
nb_epoch = 10
# in each epoch the observation input are trained where in each batch the weights are adusted
#the we will got the ratings for the movies unrated
for epoch in range(1,nb_epoch+1):
    train_loss = 0
    #normilize the train loss by the counter s (train_loss/s)
    s = 0.
    #function of the class are only for one user
    #applicate the fucntions of the class to all the users in a batch
    #rang(0,nb_users - batch_size,step = batch_size)
    for id_user in range(0,nb_users - batch_size,batch_size):
     #id_user = traget , id_user will enter in the gibbs and change and target is fix
     #target to get the error
     #vk is the input
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        #ph0,_ take only the first element of the function 
        ph0,_ = rbm.sample_h(v0)
        #for loop for the k steps of the contrastive divergence
        #take the batch of observation in gibbs loop
        for k in range(10):
            #get the first sampling of the hidden,visibles nodes
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
            _,phk = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss+=torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
            s +=1. 
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))

        


"""## Testing the RBM"""


test_loss = 0
s = 0. 
for id_user in range(nb_users):
    #we're using the training set to activate the neurons and to predict the unrated of the test set
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss+=torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s +=1. 
print('test loss: ' + str(test_loss/s))


