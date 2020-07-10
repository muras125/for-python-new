# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:44:54 2019
Note : Using LinearRegression
@author: Murali
"""


#import packages
import pandas as pd
import numpy as np
 
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.liner_model import LinerRegression
from sklearn.dataset import make_regression
 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os

root_dir_path = str(os.curdir)
 
#load data
train_df = pd.read_excel(root_dir_path+"\Data\Data_Train_trimed.xlsx")
test_df = pd.read_excel(root_dir_path+"\Data\Data_Test_trimed.xlsx")

train_df['Type'] = 'Train'
test_df['Type'] = 'Test'
 
fulldata = pd.concat([train_df,test_df],axis=0)
 
#read the data set and handle missing values
 
print(fulldata.columns)
 
print(fulldata.head(10))
 
print(fulldata.describe())
 
#Identify the a) ID variables b)  Target variables c) Categorical Variables d) Numerical Variables e) Other Variables
 
ID_col = ['Restaurant']
target_col = ['Delivery_Time']
#cat_cols = ['Location','Rating','Votes','Reviews','Average_Cost','Cuisines','Minimum_Order']
cat_cols = ['Location','Rating','Votes','Reviews']
other_col = ['Type']
num_cols = list(set(list(fulldata.columns))-set(cat_cols)-set(ID_col)-set(target_col)-set(other_col))

 
#Will return the feature with True or False,True means have missing value else False
print(fulldata.isnull().any())
 
num_cat_cols = num_cols+cat_cols
 
#Create a new variable for each variable having missing value with VariableName_NA 
# and flag missing value with 1 and other with 0
 
for var in num_cat_cols:
    if fulldata[var].isnull().any()==True:
        fulldata[var+'_NA']=fulldata[var].isnull()*1
 
#Impute numerical missing values with mean                     
fulldata[num_cols] = fulldata[num_cols].fillna(fulldata[num_cols].mean(),inplace=True)
 
#Impute categorical missing values with -9999
fulldata[cat_cols] = fulldata[cat_cols].fillna(value = -9999)
 
#create label encoders for categorical features

number = LabelEncoder()

for var in cat_cols:
    #number = LabelEncoder()
    fulldata[var] = number.fit_transform(fulldata[var].astype('str'))
 
 
#Traget values is also a categorical, so convert it
fulldata['Delivery_Time'] = number.fit_transform(fulldata['Delivery_Time'].astype('str'))
 
train = fulldata[fulldata['Type']=='Train']
test = fulldata[fulldata['Type']=='Test']
 
train['is_train']= np.random.uniform(0, 1, len(train)) <= .75
 
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]
 
 
 
features = list(set(list(fulldata.columns))-set(ID_col)-set(target_col)-set(other_col))
 
x_train = Train[list(features)].values
 
y_train = Train['Delivery_Time'].values
 
x_validate = Validate[list(features)].values
 
y_validate = Validate['Delivery_Time'].values
 
x_test = test[list(features)].values

model = LinerRegression()
model.fit(x_train, y_train)

"""
 
random.seed(100)
rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train, y_train)
 
#Check performance and make predictions
 
#status = rf.predict_proba(x_validate)
#fpr, tpr, _ = roc_curve(y_validate, status[:,1])
 
#roc_auc = auc(fpr, tpr)

#l = list(number.inverse_transform(test["Delivery_Time"]))

final_status = rf.predict_proba(x_test)[:, 1]
#score = rf.score(x_train, y_train)
test['Delivery_Time']=final_status#[:, 1]

#print(test.head(10))

for i in range(len(x_test)-1):
    print("X=%s, Predict=%s" % (x_test[i], final_status[i]))
    
#results_ordered_by_probability = map(lambda x: x[0], sorted(zip(rf.classes_, final_status), key=lambda x: x[1], reverse=True))
#print(results_ordered_by_probability)


test.to_csv(root_dir_path+'\model_output.csv', columns=['Restaurant','Delivery_Time'])
 