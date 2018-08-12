#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:35:46 2017

@author: Vincent
"""

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#import data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
#print(test_df.head())
print(train_df.head())
print(train_df.columns.values)



train_df.info()
print('_'*40)
test_df.info()



train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`


train_df.describe(include=['O'])


train_df[['Pclass', 'Survived']].groupby(['Pclass'], \
       as_index=False).mean().sort_values(by='Survived', ascending=False)


train_df[["Sex", "Survived"]].groupby(['Sex'], \
        as_index=False).mean().sort_values(by='Survived', ascending=False)


train_df[["SibSp", "Survived"]].groupby(['SibSp'],\
         as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["Parch", "Survived"]].groupby(['Parch'],
        as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["Cabin", "Survived"]].groupby(['Cabin'],
        as_index=False).mean().sort_values(by='Survived', ascending=False)




g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)



# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=5)
grid.add_legend();

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()



for dataset in combine:
     dataset['CabinChar'] = dataset['Cabin'].str[:1]
     dataset['CabinChar'] = dataset.CabinChar.apply(lambda x: x if not pd.isnull(x) else 0)

for dataset in combine:
    dataset['CabinChar'] = dataset['CabinChar'].map( {0 : 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8} ).astype(int)


train_df.head()

#fill null if there are many different values and many nulls
train_df['Cabin'] = train_df.Cabin.apply(lambda x: x if pd.isnull(x) else 1)
train_df['Cabin'] = train_df.Cabin.apply(lambda x: x if not pd.isnull(x) else 0)

test_df['Cabin'] = test_df.Cabin.apply(lambda x: x if pd.isnull(x) else 1)
test_df['Cabin'] = test_df.Cabin.apply(lambda x: x if not pd.isnull(x) else 0)
combine = [train_df, test_df]


train_df[['CabinChar', 'Survived']].groupby(['CabinChar'], as_index=False).mean()




#clean data
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape




#name title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])



for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()





title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()



train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape




for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()




# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


for dataset in combine:
    dataset['Age is null'] = np.where(dataset['Age'].isnull(), 1, 0)

train_df[['Age is null', 'Survived']].groupby(['Age is null'], as_index=False).mean()


guess_ages = np.zeros((2,3))
guess_ages




for dataset in combine:
    for i in range(0, 2):
        for j in range(3 , -1):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


for dataset in combine:
    dataset['Pclass'] = dataset['Pclass'].map( {3 : 0, 2 : 1, 1 : 2} ).astype(int)



train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], 
        as_index=False).mean().sort_values(by='AgeBand', ascending=True)


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()



train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()



for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], 
        as_index=False).mean().sort_values(by='Survived', ascending=False)



for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()



train_df = train_df.drop(['IsAlone', 'FamilySize'], axis=1)
test_df = test_df.drop(['IsAlone', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()





for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], 
        as_index=False).mean().sort_values(by='Survived', ascending=False)


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()



test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], 
        as_index=False).mean().sort_values(by='FareBand', ascending=True)




for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)

train_df = train_df.drop(['Age*Class'], axis=1)
test_df = test_df.drop(['Age*Class'], axis=1)
#train_df = train_df.drop(['IsAlone'], axis=1)
#test_df = test_df.drop(['IsAlone'], axis=1)


    



X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
X = X_train.iloc[: , :].values

x = X_test.iloc[:,:].values




from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
x = onehotencoder.transform(x).toarray()
X = X[:, 1:]
x = x[:,1:]


onehotencoder2 = OneHotEncoder(categorical_features = [8])
X = onehotencoder2.fit_transform(X).toarray()
x = onehotencoder2.transform(x).toarray()
X = X[:, 1:]
x = x[:,1:]

onehotencoder3 = OneHotEncoder(categorical_features = [16])
X = onehotencoder3.fit_transform(X).toarray()
x = onehotencoder3.transform(x).toarray()
X = X[:, 1:]
x = x[:,1:]



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
x = sc.transform(x)


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 40, activation = 'relu', input_dim = 10))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 20, activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, Y_train, batch_size = 5, nb_epoch = 200)

test_predict = classifier.predict(x)

y_pred = (test_predict + 0.5).astype("int")

#Y_pred = y_pred.reshape(418)

y_pred = pd.Series(y_pred)

y_pred = y_pred.flatten()


print(test_predict)





# Logistic Regression

logreg = LogisticRegression(penalty = 'l2', C = 0.1)
logreg.fit(X, Y_train)
Y_pred = logreg.predict(x)
acc_log = round(logreg.score(X, Y_train) * 100, 2)
acc_log


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# Support Vector Machines

svc = SVC()
svc.fit(X, Y_train)
Y_pred = svc.predict(x)
acc_svc = round(svc.score(X, Y_train) * 100, 2)
acc_svc


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, Y_train)
Y_pred = knn.predict(x)
acc_knn = round(knn.score(X, Y_train) * 100, 2)
acc_knn


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X, Y_train)
Y_pred = gaussian.predict(x)
acc_gaussian = round(gaussian.score(X, Y_train) * 100, 2)
acc_gaussian


# Perceptron

perceptron = Perceptron()
perceptron.fit(X, Y_train)
Y_pred = perceptron.predict(x)
acc_perceptron = round(perceptron.score(X, Y_train) * 100, 2)
acc_perceptron


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X, Y_train)
Y_pred = linear_svc.predict(x)
acc_linear_svc = round(linear_svc.score(X, Y_train) * 100, 2)
acc_linear_svc


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X, Y_train)
Y_pred = sgd.predict(x)
acc_sgd = round(sgd.score(X, Y_train) * 100, 2)
acc_sgd


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, Y_train)
Y_pred = decision_tree.predict(x)
acc_decision_tree = round(decision_tree.score(X, Y_train) * 100, 2)
acc_decision_tree



# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, criterion="entropy", n_jobs = -1)
random_forest.fit(X, Y_train)
Y_pred = random_forest.predict(x)
random_forest.score(X, Y_train)
acc_random_forest = round(random_forest.score(X, Y_train) * 100, 2)
acc_random_forest


#compare
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'], 
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)





submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })  
submission.to_csv('new.csv', index=False)
    
