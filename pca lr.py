#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 00:37:51 2017

@author: Vincent
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca= pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


lr2 = LogisticRegression(penalty = 'l1', C = 10**-3, random_state=0)
lr2.fit(X_train_pca, Y_train)

from sklearn.cross_validation import cross_val_score
cv_scores = np.mean(cross_val_score(lr2, X_train_pca, Y_train, scoring='roc_auc', cv=10))
print (cv_scores)




y_pred = lr2.predict(X_test_pca)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
