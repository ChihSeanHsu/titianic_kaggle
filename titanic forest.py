#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:54:46 2017

@author: Vincent
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Importing the dataset
dataset = pd.read_csv('train.csv')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',]
training_data = dataset[features]
training_label = dataset['Survived']



training_data = training_data.fillna(0)
#training_data['Name'] = LabelEncoder().fit_transform(training_data['Name'])

training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'])
#training_data['Ticket'] = LabelEncoder().fit_transform(training_data['Ticket'])

model = RandomForestClassifier(n_estimators = 15)
model.fit(training_data, training_label)

y_pos = np.arange(len(features))
plt.barh(y_pos, model.feature_importances_, align='center', alpha=0.4)
plt.yticks(y_pos, features)
plt.xlabel('features')
plt.title('feature_importances')
plt.show()


features = ['Age','Fare']
training_data = dataset[features]
training_label = dataset['Survived']


training_data = training_data.fillna(0)
#training_data['Name'] = LabelEncoder().fit_transform(training_data['Name'])

training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'])
#training_data['Ticket'] = LabelEncoder().fit_transform(training_data['Ticket'])

model2 = RandomForestClassifier()
model2.fit(training_data, training_label)

y_pos = np.arange(len(features))
plt.barh(y_pos, model2.feature_importances_, align='center', alpha=0.4)
plt.yticks(y_pos, features)
plt.xlabel('features')
plt.title('feature_importances')
plt.show()


test_data = pd.read_csv("test.csv")
preidction_data = test_data[features]
preidction_data = preidction_data.fillna(0)
result_lables = model2.predict(preidction_data)
results = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : result_lables
})

results.to_csv("submission2.csv", index=False)


from sklearn.cross_validation import cross_val_score
cv_scores = np.mean(cross_val_score(model2, training_data, training_label, scoring='roc_auc', cv=5))
print (cv_scores)

X = dataset.iloc[:,[2, 4, 5, 6, 7, 9]].values
y = dataset.iloc[:, 1].values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()



# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)




X.dtype = 'float32'

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""