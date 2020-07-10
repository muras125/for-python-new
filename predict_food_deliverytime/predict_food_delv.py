# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:42:41 2019

@author: Murali
"""

import pandas as pd
import matplotlib as mp
import os
import pandas_profiling

root_dir_path = str(os.curdir)
df_train = pd.read_excel(root_dir_path+"\Data\Data_Train.xlsx")

print(df_train.describe())
print(df_train.head(3))

#profile = df_train.profile_report(title='Food Delivery Profiling')
#profile.to_file(output_file="Food_Delivery_Profiling.html")

"""
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight=’balanced’)
boruta_selector = BorutaPy(rfc, n_estimators=’auto’, verbose=2)
x=df.iloc[:,:].values
y=dflabel.iloc[:,0].values
boruta_selector.fit(x,y)
print(“==============BORUTA==============”)
print (boruta_selector.n_features_)
"""

import scipy

def ChiSquare(df,featureList,label,alpha=0.05):
    for category in featureList:
        ct=pd.crosstab(df[category],df[label])
        #print(ct)
        chi_square_value,p_value,_,_=scipy.stats.chi2_contingency(ct)
        if p_value <=alpha:
            print(category,'is Important with p Value:',p_value)
        else:
            print(category,'is Not Important with p Value:',p_value)


ChiSquare(df_train, ["Location","Cuisines","Average_Cost","Minimum_Order","Rating","Votes","Reviews","Delivery_Time"],"Delivery_Time")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

labels= df_train['Delivery_Time']
train1 = df_train.drop(['Restaurant','Delivery_Time','Cuisines','Votes','Reviews'], axis=1)
#train1 = df_train.drop(['id','Restaurant'], axis=1)

df_cat = pd.DataFrame({'A': train1['Location']})
print(df_cat)

"""
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size = 0.10, random_state=2)

print (reg.fit(x_train, y_train))
print (reg.score(x_test, y_test))


from sklearn import ensemble

# Fit regression model
params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}


clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')
print (clf.fit(x_train, y_train))

test_score = clf.score(x_test, y_test)
"""