# -*- coding: utf-8 -*-
"""AutoEncoders.ipynb


"""##Importing the libraries"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

"""## Importing the dataset"""

# We won't be using this dataset.
movies = pd.read_csv('ml-1m/movies.dat',sep ='::', header=None,
                     engine='python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat',sep ='::', header=None,
                     engine='python', encoding = 'latin-1') 
ratings = pd.read_csv('ml-1m/ratings.dat',sep ='::', header=None,
                     engine='python', encoding = 'latin-1')

"""## Preparing the training set and the test set"""
training_set  = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set  = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')


"""## Getting the number of users and movies"""
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))


"""## Converting the data into an array with users in lines and movies in columns"""
def convert(data):
    new_data = []
    for id_users in range(1,nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        #index np start from 0 and id_movies start from 1
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)
        


"""## Converting the data into Torch tensors"""
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

"""## Creating the architecture of the Neural Network"""
#create a child class of torch.nn caled Module
class SAE(nn.Module):
    #self is the autoencoder, self,(vide) beacuse it will take the variable of Module
    #function 1: archtecture of the neuron network
    def __init__(self, ):
        #using super method to get the inheratance methods from Module
        super(SAE,self).__init__()
        #the full connection the input vector (ratings for all specific user) features
        #and the first hidden layer 
        #vector hiritten class : full conxion between input feature and the first encoder vector
         #20 nbr of the neurons in the first hidden layer
         #fc1 is an object of the linear class
        self.fc1 = nn.Linear(nb_movies,20)
        #second full connexion
        self.fc2 = nn.Linear(20,10)
        #start to decode , constract the input vector
        self.fc3 = nn.Linear(10,20)
        #last decoding
        self.fc4 = nn.Linear(20,nb_movies)
        self.activation  = nn.Sigmoid()
    # function 2 : encoding and decoding, apply activation function
    # self is the object, self to apply function on the object
    def forward(self,x):
        #encoding input to the first full connxtion layer
        x = self.activation(self.fc1(x)) #return the encoding vector
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE() 
criterian = nn.MSELoss()
criterion = criterian
#lr : learning rate, decay : reduce the learnong rate after few epochsto to regulate the model
#otimizer to update the weights
optimizer = optim.RMSprop(sae.parameters(),lr = 0.01, weight_decay=0.5)




nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = input.clone()
    if torch.sum(target.data > 0) > 0:
      output = sae(input)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step()
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))


"""## Training the SAE"""
nb_epoch = 200 
for epoch in range(1, nb_epoch+  1):
    train_loss = 0
    # nbr of users rated to optimze the memory
    s = 0.
    for id_users in range(nb_users):
        #the fucntion Variable convert to vector a matrix
         # 0 is the index of the new dimision
        input = Variable(training_set[id_users]).unsqueeze(0)
        # clone copy the variable
        target = input.clone()
        if torch.sum(target.data>0)>0:
            output = sae(input)
            #Making shoore to not applicate gradient on the target
            target.require_grad = False
            # not counting the unrated movies in the optimizer
            output[target == 0] = 0
            loss = criterian(output, target)
            # nb_movies /nbr movies that have positive ratings
            #average of the error by only consedering rated movies
            mean_corrector = nb_movies / float(torch.sum(target.data >0)+1e-10)
            #backward decide the direction of the weight (increase or decrease)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1
            #step will applicate the optimizer
            #optimize decide the intensity of the weights
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))
            
    
    


"""## Testing the SAE"""
#data in test set are the data unrated in the training set
#predict by the training set then compare this prediction to the data in  the test set
test_loss = 0
s = 0.
for id_users in range(nb_users):
    input = Variable(training_set[id_users]).unsqueeze(0)
    target = Variable(test_set[id_users]).unsqueeze(0)
    if torch.sum(target.data>0)>0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterian(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data >0)+1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss/s))
            

