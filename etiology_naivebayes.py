# -*- coding: utf-8 -*-
"""
naivebayes_etiology(X,y,test_pc)
Returns Naive Bayes output for classification with scatter plot and confusion matrix.
Tweaked for use with etiologyclassifier (removed plotting ability in current function)
X = matrix of predictor data
y = array of class data
test_pc = % of data for test set
@author: BCM
"""
######################################################################################################
# Import packages
import numpy as np 
import random
from sklearn.naive_bayes import GaussianNB

def naivebayes_etiology(X,y,test_pc):
    ######################################################################################################
    # Split data into training/test sets
    # Set % of data for test set
    ntest=np.round_(X.shape[0]*(test_pc/100))
    # convert float to int
    ntest=int(ntest)
    
    # Randomly select indices to split rows into training/testing sets 
    trainidx=np.arange(0,X.shape[0])
    testidx=np.array(random.sample(range(X.shape[0]), ntest))
    trainidx = np.delete(trainidx, testidx)
    
    # Split predictor data into training/testing sets
    X_train = X[trainidx,:]
    X_test = X[testidx,:]
    
    # Split target data into training/testing sets
    y_train = y[trainidx]
    y_test = y[testidx]
    
    ######################################################################################################
    # Train a Gaussian Naive Bayes classifier
    # Setup model
    mdl = GaussianNB()
    
    # Fit model
    mdl.fit(X_train, y_train)
    print(mdl)
    
    # Predict y
    y_pred = mdl.predict(X_test)
    
    ######################################################################################################
    # Metrics, evaluation
    print('Training set score = {:.3f}'.format(mdl.score(X_train, y_train)))
    print('Test set score = {:.3f}'.format(mdl.score(X_test, y_test)))
    print(sum(y_test==y_pred),'out of ',len(y_test),'classifications correct')
        
    return mdl, y_test, y_pred
