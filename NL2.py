#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:30:34 2017

@author: Vincent
"""

from keras.models import Sequential
from keras.layers import Dense

X = train_df.iloc[:, :].values
Y = train_df.iloc[:, 1].values
x = test_df.iloc[:, :].values


model = Sequential()
model.add(Dense(units=22, input_dim=11, activation='tanh')) 
model.add(Dense(units=10, activation='tanh')) 
model.add(Dense(units=1, activation='sigmoid')) 

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(50001):
    cost = model.train_on_batch(X, Y)
    if step % 100 == 0:
        print('step', step, 'train cost:', cost)

# predict
test_predict = model.predict(x)

for a in range(0,418):
    if test_predict[a] >= 0.5:
        test_predict[a] = 1
    else:
        test_predict[a] = 0


print(test_predict)

test_predict= test_predict.reshape(418,)
test_predict = test_predict.astype(np.int64)


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_predict
    })
submission.to_csv('submissionNL.csv', index=False)
