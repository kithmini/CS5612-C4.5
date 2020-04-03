#!/usr/bin/env python
# coding: utf-8

# In[19]:


print('##### Logistic Regression on Wisconsin Breast Cancer Data #####')

print('----- Importing required libraries & modules-----')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split


# In[20]:


print('----- Importing dataset -----')
data = pd.read_csv('bcwd.csv', header=None)

data.columns = ['sample_code_number','clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion',              'single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
feature_columns = ['clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion',              'single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses']



print ('Imported Rows, Columns - ', data.shape)
print ('Data Head :')
data.head()


# In[21]:


missingRemovedData =  data[data['bare_nuclei'] != '?'] # remove rows with missing data

X = missingRemovedData[feature_columns]
y = missingRemovedData['class']
y = (y-2)/2 #simplified

# split X and y into training and teting sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

#Implemnt C4.5 model
cdtModel = tree.DecisionTreeClassifier()
cdtModel = cdtModel.fit(X_train, y_train)

get_ipython().run_line_magic('matplotlib', 'inline')
tree.plot_tree(cdtModel)


# 

# ##### 

# In[22]:


y_pred = cdtModel.predict(X_test)


# In[23]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn import metrics

print("Accuracy: ",  metrics.accuracy_score(y_test, y_pred)*100.0)

