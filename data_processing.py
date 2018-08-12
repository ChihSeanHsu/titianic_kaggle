#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 00:13:20 2017

@author: Vincent
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('train.csv')



test = pd.read_csv('test.csv')

#check null
dataset.isnull().sum()
test.isnull().sum()
dataset['Embarked'] = dataset['Embarked'].fillna('Unknown')


#class resort
class_map = {
        3 : 1, 
        2 : 2,
        1 : 3}
dataset['Pclass'] = dataset['Pclass'].map(class_map)
test['Pclass'] = test['Pclass'].map(class_map)


#into array
X = dataset.iloc[:, [2, 4, 5, 6, 7, 9]].values
y = dataset.iloc[:, 1].values
t_x = test.iloc[:, [1, 3, 4, 5, 6, 8].values


#encode
from sklearn.preprocessing import LabelEncoder
'''X[:,0] = LabelEncoder().fit_transform(X[:,0])'''
X[:,1] = LabelEncoder().fit_transform(X[:,1])
'''X[:,6] = LabelEncoder().fit_transform(X[:,6])'''

t_x[:,1] = LabelEncoder().fit_transform(t_x[:,1])
'''t_x[:,6] = LabelEncoder().fit_transform(t_x[:,6])'''

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

imputer_t = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_t = imputer_t.fit(t_x)
t_x = imputer_t.transform(t_x)

#onehotencoder
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1: ] #onehotencoder trap

onehotencoder_t = OneHotEncoder(categorical_features = [6])
t_x = onehotencoder_t.fit_transform(t_x).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





#output

result_lables = forest.predict(t_xc)

t_x = pd.DataFrame(t_x)

results = pd.DataFrame({
    'PassengerId' : test['PassengerId'],
    'Survived' : result_lables
})
    
    
results.to_csv("submission forest2.csv", index=False)

from sklearn.cross_validation import cross_val_score
cv_scores = np.mean(cross_val_score(forest, X, y, scoring='roc_auc', cv=5))
print (cv_scores)



