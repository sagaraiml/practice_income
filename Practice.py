# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 6 19:05:04 2019

@author: sagar_paithankar
"""
#############################importing libraries and data set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import os
path = os.getcwd()
print(path)
os.chdir(r'G:\Anaconda_CC\spyder')
del path

plt.rc('font', size = 14)
sns.set(style='white')

raw = pd.read_csv('income.csv')
shape = raw.shape

############################preprocessing
raw.isna().any()
raw.dropna()
#dependent columns deletion
raw.drop(columns=['education', 'relationship'], inplace=True)
#renaming y variable to Income
raw = raw.rename(columns = {'Other' : 'Income'})
#dtypes of columns
dty =raw.dtypes
#uniques values in each columns
uni = [raw[col].unique() for col in raw]
#removing rows containig ' ?' value in columns
for col in ['workclass', 'occupation', 'native-country']:
    raw = raw[raw[col] != ' ?']
#uniques values in each columns again all ? are gone now
uni = [raw[col].unique() for col in raw]
del(col)
#######################finding relation between varibles now
#changin >50 to 1 and <=50 to 0 this is coz
#we can directly see the relation with Income now 
rcol = {'Income' : {' <=50K':0, ' >50K': 1 }}
data = raw.replace(rcol, regex=True)
del rcol
y = data['Income']
y.value_counts()    
sns.countplot(x=y, data=data, palette='hls')
#calculating relationshp
rich = len(data[y == 1])
poor = len(data[y == 0])
print('% of rich and poor is : '+str(rich*100/len(y))+' , '+str(poor*100/len(y)))

################Label encoding
cat_vars = ['workclass', 'marital-status', 'occupation', 'race', 'sex','native-country']

from sklearn.preprocessing import LabelEncoder
LEX = LabelEncoder()
for col in ['workclass', 'marital-status', 'occupation', 'race', 'sex','native-country']:
    data.loc[:, col] = LEX.fit_transform(data.loc[:, col])
dty =data.dtypes
del(col)

#uniques values in each columns
uniLE = [data[col].unique() for col in data]

preprocessed_data = open('preprocessed_data.pkl', 'wb')
pickle.dump(data, preprocessed_data)
preprocessed_data.close()

################One hot encoding
for var in cat_vars:
    #cat_list='var'+'_'+var #i dont know what this line is making  ?
    cat_list = pd.get_dummies(data[var], prefix=var)
    df=data.join(cat_list)
    data = df
del(var)
#droping original cat columns
data.drop(columns=cat_vars, inplace=True)

###################### splitting the data and feature scaling
X = data.loc[:, data.columns != 'Income']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 
at second case we just transformed coz its already splitted
"""
"""
####################### model building LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#################### predecting result
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
acc = metrics.classification_report(y_test, y_pred)
print(cm)
#[[5343  388]
# [ 748 1062]]
print(accuracy) #0.8493568492242408
print(acc)
print(classifier.coef_)

################## ROC Curve
Logit_roc = metrics.roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1])
fpr, tpr, thresholds = metrics.roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Log Regression (area = %0.2f)' % Logit_roc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False + rate')
plt.ylabel('True + rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
"""
########################----------------------- model building xgboost
#The data is stored in a DMatrix object 
#label is used to define our outcome variable
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

#parameters = {'booster':'gbtree', 'max_depth':7, 'eta':1 ,'gamma':1, 'min_child_weight':1, \
#         'objective':'binary:logistic', 'subsample':0.5, 'colsample_by*':0.6, \
#         'eval_metric':'auc', 'learning_rate':0.5}

#parameters={'max_depth':7, 'silent':1,'objective':'binary:logistic',\
#            'eval_metric':'auc','learning_rate':.05}

#parameters = {'n_estimators':200,'objective':'reg:linear','booster':'gbtree','max_depth':10,\
#           'learning_rate':0.04,'colsample_bytree':0.6,'colsample_bylevel':0.6,\
#           'subsample':0.8,'min_child_weight':1,'gamma':1}

#parameters={'max_depth':10, 'silent':1,'objective':'binary:logistic',\
#            'eval_metric':'auc','learning_rate':.05}

parameters={'max_depth':2, 'silent':1, 'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
#validation set is eval set
evallist = [(dval, 'eval')]

#training our model 
num_round=50
from datetime import datetime
start = datetime.now()

xg=xgb.train(parameters,dtrain,num_round, evallist, early_stopping_rounds=10)

stop = datetime.now()

#Execution time of the model
#datetime.timedelta( , , ) representation => (days , seconds , microseconds)  
time = stop-start
print(time)

#################### predecting result
y_pred = xg.predict(dtest)

#converting probs in to 1, 0 
for i in range(len(y_pred)):
    if y_pred[i] >=0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

#calculating accurancy
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
acc = metrics.classification_report(y_test, y_pred)
print(cm)
print(accuracy) 
print(acc)
#---------------------------------------------------------------------
########################----------------------- model building lightgbm
import lightgbm as lgb
#important : label.values.ravel() is important
dtrain = lgb.Dataset(data=X_train, label=y_train.values.ravel())
dval = lgb.Dataset(data=X_val, label=y_val.values.ravel(), reference=dtrain)

paras = {'max_depth':7,'objective':'Binary', 'Learning_rate':0.05,'metric':'auc','num_iterations':100}

model = lgb.train(paras, dtrain, valid_sets=dval, early_stopping_rounds=15)









