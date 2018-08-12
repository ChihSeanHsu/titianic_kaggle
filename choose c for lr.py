#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:09:37 2017

@author: Vincent
"""

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

fig = plt.figure()

ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow','black', 'pink', 'lightgreen', 'lightblur']
weights, params = [], []


a_arange = np.arange(-4, 6, dtype=np.float)
coef_ = pd.DataFrame()
label_a = np.arange(0,9)

for c in a_arange:
    lr = LogisticRegression(penalty = 'l1', C = 10**c, random_state=0)
    lr.fit(X_train, Y_train)

    weights.append(lr.coef_[0])
    params.append(10 ** c)
    

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label = label_a[column], color = color)
    
    
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc = 'upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()