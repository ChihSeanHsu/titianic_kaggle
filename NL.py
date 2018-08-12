#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:56:51 2017

@author: Vincent
"""

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from subprocess import check_output
'''print(check_output(["ls", ""]).decode("utf8"))'''


def batch_generator(x_train, y_train, batch_size=256):
    """
    Gives equal number of positive and negative samples, and rotates them randomly in time
    """
    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1]), dtype='float32')
    y_batch = np.empty((batch_size), dtype='float32')
    
    yes_idx = np.where(y_train == 1.)[0]
    non_idx = np.where(y_train == 0.)[0]
    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)
        
        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]
    
        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
        yield x_batch, y_batch
        
def train_model(model, lr, nb_epochs):
    model.compile(optimizer=Adam(lr), loss = 'binary_crossentropy', metrics=['accuracy'])
    hist = model.fit_generator(batch_generator(train_X, train_y, batch_size), 
                           validation_data=(valid_X, valid_y), 
                           verbose=0, epochs=nb_epochs,
                           steps_per_epoch=train_X.shape[0]//batch_size)
    return model, hist

def plot_hist(hist, N):
    train_loss ,= plt.plot(np.convolve(hist.history['loss'], np.ones((N,))/N, mode='valid'), color='b', label='training loss')
    val_loss ,= plt.plot(hist.history['val_loss'], color='r', label='validation loss')
    plt.ylabel('Loss')
    plt.legend(handles=[train_loss, val_loss])
    plt.show()
    train_acc ,= plt.plot(np.convolve(hist.history['acc'], np.ones((N,))/N, mode='valid'), color='b', label='training accuracy')
    val_acc ,= plt.plot(np.convolve(hist.history['val_acc'], np.ones((N,))/N, mode='valid'), color='r', label='validation accuracy')
    plt.ylabel('Accuracy')
    plt.legend(handles=[train_acc, val_acc])
    plt.show()
    
    
    
    
    
    
#test_df = pd.read_csv('test.csv')

# Create all datasets that are necessary to train, validate and test models
'''train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
'''
#test_X = test_df[ 891: ].as_matrix()

seeds = np.arange(3)
from sklearn.cross_validation import train_test_split
train_X , valid_X , train_y , valid_y = train_test_split( X, Y, train_size = .8 )
'''
train_X = train_X.as_matrix()
valid_X = valid_X.as_matrix()
train_y = train_y.as_matrix()
valid_y = valid_y.as_matrix()
'''
print (train_X.shape , valid_X.shape , train_y.shape , valid_y.shape )






hists = []
models = []
promising_seeds = []

# rapidly test seeds and keep promising ones
for a_seed in seeds:
    # create model
    model = Sequential()
    model.add(Dense(train_X.shape[1]+10, activation='relu',input_shape=train_X.shape[1:]))
    #model.add(Dropout(0.1))
    model.add(Dense(8, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    batch_size = 256
    # random seed
    np.random.seed(a_seed)
    #Start with a slightly lower learning rate, to ensure convergence
    model, hist = train_model(model, 6e-4, 200)
    # faster convergence
    model, hist = train_model(model, 9e-4, 300)
    hists.append(hist)
    models.append(model)
    plot_hist(hist, 100)
    print("seed", a_seed, "crossval accuracy", np.mean(hist.history['val_acc'][-100:-1]))
    # if the model has reach at least 55% cross validation accuracy
    if np.mean(hist.history['val_acc'][-100:-1]) > 0.55:
        promising_seeds.append(a_seed)
        
        
        
        
        
        
        
# train promising models
best_seed = 0
best_acc = 0.0
# not using good seeds at the moment
good_seeds = []
for a_seed in promising_seeds:
    # add some convergence time
    models[a_seed], hists[a_seed] = train_model(models[a_seed], 9e-4, 200)
    acc = np.mean(hists[a_seed].history['val_acc'][-100:-1])
    print("seed", a_seed, "crossval accuracy", acc)
    if acc >0.6:
        good_seeds.append(a_seed)
    if acc > best_acc:
        best_seed = a_seed
        best_acc = acc
        
        

# draw all models to check if we kept the best  
for a_seed in seeds:
    hist = hists[a_seed]
    plot_hist(hist, 100)
    if a_seed == best_seed:
        print("best seed")
    print(np.mean(hist.history['val_acc'][-100:-1]))
    
    
    


prediction = models[best_seed].predict(test_X)[:,0]
test_y = (prediction + 0.5).astype("int")
ids = full[891:]['PassengerId']
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': test_y })
output.to_csv('submission.csv', index = False)
output.head()