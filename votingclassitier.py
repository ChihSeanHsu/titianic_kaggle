#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:59:40 2017

@author: Vincent
"""


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier





knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

pipe_knn = Pipeline([('scl', StandardScaler()),
                    ('clf', knn)])



forest = RandomForestClassifier(criterion= 'entropy', n_estimators = 1000, n_jobs = -1)


vcl = VotingClassifier(estimators = [
        ('lsvc',linear_svc),('b', gaussian),('lr', logreg), ('rf', random_forest), ('knn', knn)], voting='hard')

vcl.fit(X, Y_train)


from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator = vcl,
                         X = X,
                         y = Y_train,
                         cv = 10,
                         n_jobs = 1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (
        np.mean(scores), np.std(scores)))


Y_pred = vcl.predict(x)
