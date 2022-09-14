# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:01:35 2022

Audiological etiology classifier script

Uses Naive Bayes classifier, with feature importance, to 
classify unknown etiologies in dataset below

Using Audiology (Standardized) Data Set from below:
https://archive.ics.uci.edu/ml/datasets/Audiology+%28Standardized%29

@author: BCM
"""
############################################################################################################################################################################################################
# Import packages, pre-processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from naivebayes_etiology import naivebayes_etiology
from sklearn.inspection import permutation_importance
from sklearn import preprocessing

# Import dataset
filename='audiology.standardized.data'
folder=<ENTER FOLDER HERE>
data = pd.read_csv(folder+filename,header=None)

# Setup data
# Audiological data
X=data.iloc[:,0:-2]
# Etiologies (classes)
y=data.iloc[:,-1]

# Represent categorical variables numerically
le = preprocessing.LabelEncoder()

# Transform etiology data
yt=le.fit_transform(y)

############################################################################################################################################################################################################
# Bar plot etiology counts
et_string=np.unique(y)
et_counts=np.arange(0,np.max(yt)+1)
for i in et_counts:
    et_counts[i]=(sum(yt==i))   

# Return indices of sorted array by size
et_sort=np.argsort(et_counts)    
# Plot
plt.grid(color = 'black', linestyle = '--', linewidth = 0.2)
plt.bar(np.arange(0,len(et_counts)),et_counts[et_sort])
plt.xlabel("Etiology", fontsize=14)   
plt.ylabel("Counts", fontsize=14) 
plt.xticks(range(len(et_string)),et_string[et_sort],rotation='vertical', fontsize=11)
plt.show()

############################################################################################################################################################################################################
# Transform entire audiological predictor data numerically using for loop
Xt=X
for i in np.arange(0,Xt.shape[1]):
    Xt[i] = le.fit_transform(Xt[i])
# Convert dataframe to array
Xt=np.array(Xt)
# Make X original data again (unsure why it gets transformed)    
X=data.iloc[:,0:-2]

# Xt=Xt.rename(columns={0:"age60",1:"air",2:"ABG",3:"ar_c",4:"ar_u",5:"bone",6:"boneab",7:"bser",8:"history_buzzing"})

############################################################################################################################################################################################################
# Setup training and test data using data with known etiolgies
known_idx=np.array(np.where(yt!=7))
Xt_known=np.squeeze(Xt[known_idx,:])
y2=np.squeeze(yt[known_idx])

############################################################################################################################################################################################################
# Create, run model
mdl, y_test, y_pred=naivebayes_etiology(Xt_known,y2,20)

# Print prediction strings
print(np.column_stack((et_string[y_test]+" predicted to be "+et_string[y_pred])))

############################################################################################################################################################################################################
# Calculate feature importance using permutation importance technique
imps = permutation_importance(mdl, Xt_known, y2)
importance = imps.importances_mean
importance_idx = np.argsort(importance)[::-1]
features=np.arange(0,Xt.shape[1])

# # Print feature importance
# print("Feature importance")
# for i in range(Xt_known.shape[1]):
#     print("(%s) (%f)" % (features[importance_idx[i]], importance[importance_idx[i]]))

rankimportance=importance[importance_idx]
featureimportance=features[importance_idx]
featureimportance_std=imps.importances_std[importance_idx]  
# Plot nfplot most important features
nfplot=15
plt.bar(np.arange(0,nfplot),rankimportance[0:nfplot],yerr=featureimportance_std[0:nfplot])
plt.xticks(np.arange(0,nfplot),featureimportance[0:nfplot],fontsize=11)
plt.xlim([-1,nfplot])
plt.xlabel("Feature #", fontsize=14)   
plt.ylabel("Importance", fontsize=14) 
plt.show()

############################################################################################################################################################################################################
# Now use model on data with unknown etiologies
# Setup unknown etiology data
unknown_idx=np.array(np.where(yt==7))
X_unknown=X.iloc[np.squeeze(np.array(unknown_idx, dtype=np.int32))]
Xt_unknown=np.squeeze(Xt[unknown_idx,:])

# Predict etiologies of unknown etiology data
unknown_predict=mdl.predict(Xt_unknown)
et_string[unknown_predict]

# Put data and predictions of unknown into dataframe
predictionDf=pd.DataFrame(X_unknown)
#predictionDf['Predictions']=unknown_predict
predictionDf['StrPredictions']=et_string[unknown_predict]
print(predictionDf)
