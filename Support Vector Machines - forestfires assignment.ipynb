# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:05:28 2023

@author: sksha
"""
#Import the data
import numpy as np
import pandas as pd
df = pd.read_csv("forestfires.csv")
df.info()
df.shape  #(517, 31)
df.corr()
#================================================================================================
#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #
import seaborn as sns
import matplotlib.pyplot as plt
data = df.iloc[:,2:11]
data
for column in data:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
#we have seen the ouliers , so we will try to remove them

"""removing the ouliers"""

import seaborn as sns
import matplotlib.pyplot as plt
# List of column names with continuous variables
continuous_columns = df.iloc[:,2:11]
continuous_columns.shape #(138, 9)
# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df.copy()
for column in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker) & (data_without_outliers[column] <= upper_whisker)]

# Print the cleaned data without outliers
print(data_without_outliers)
data_without_outliers.shape   #(302, 31)
df = data_without_outliers
df.shape
# Check the shape and info of the cleaned DataFrame
print(df.shape)   #(302, 31)
print(df.info())

#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df.hist()
df.skew()
df.kurt()
df.describe()
#================================================================================================
#Continous Variables 
df_cont = df.drop(df.columns[[0,1,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]],axis=1)
df_cont.shape   #(302, 9)
df_cont.head()
#Standardisation 
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(df_cont)
SS_X
X1 =pd.DataFrame(SS_X)
X1.columns = list(df_cont)
X1.head()
X1.shape #(302, 9)
X1.dropna()
#================================================================================================
#Categorical Variables
df_cat = df[df.columns[[0,1,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]]
df_cat.shape  #(302, 22)
df_cat.head()
#Label Encoding
from  sklearn.preprocessing import LabelEncoder 
LE = LabelEncoder()
# Loop through each column and label encode its values
for col in df_cat:
    df_cat[col] = LE.fit_transform(df_cat[col])

df_cat.head()
df_cat.shape

df_cat.reset_index(drop=True, inplace=True)

X1.reset_index(drop=True, inplace=True)

#Concatenation
df_final = pd.concat([X1,df_cat],axis=1)
df_final
df_final.info()

#Split the variables
Y = df["size_category"]
Y
X = df_final.drop(df_final.columns[[30]],axis=1)
X
#Data Partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,random_state=123)
X_train.shape
X_test.shape

#####################################################################
#Linear function
#SVM 
from sklearn.svm import SVC 
svc = SVC(C=1.0,kernel='linear')   #linear Classifier
svc.fit(X_train,Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_test = svc.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("training accuracy score:",(ac1*100).round(3))    #training accuracy score: 100.0
ac2 = accuracy_score(Y_test,Y_pred_test)
print("testing accuracy score:",(ac2*100).round(3))     #testing accuracy score: 96.364

#Validation set approach
training_accuracy = []
test_accuracy = []
for i in range (1,101):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,random_state=i)
    svc.fit(X_train,Y_train)
    Y_pred_train = svc.predict(X_train)
    Y_pred_test = svc.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

print("Average training accuracy score:",(np.mean(training_accuracy)*100).round(3))    #Average training accuracy score: 99.982
print("Average test accuracy score:",(np.mean(test_accuracy)*100).round(3))            #Average test accuracy score: 96.418

from sklearn.svm import SVC 
svc = SVC(C=1.0,kernel='linear')   #linear Classifier
svc.fit(X,Y)
X =df_final.iloc[:,:2]
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X.values,
                      y=Y.values,
                      clf=svc,
                      legend=4)
    
#############################################################
##Polynomial function
#SVM
from sklearn.svm import SVC
svc =SVC(degree=3,kernel='poly')  #Polynomial function
svc.fit(X_train,Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_test = svc.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("training accuracy score:",(ac1*100).round(3))    #training accuracy score: 97.576
ac2 = accuracy_score(Y_test,Y_pred_test)
print("testing accuracy score:",(ac2*100).round(3))     #testing accuracy score: 98.182

#Validation set approach
training_accuracy = []
test_accuracy = []
for i in range (1,101):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,random_state=i)
    svc.fit(X_train,Y_train)
    Y_pred_train = svc.predict(X_train)
    Y_pred_test = svc.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

print("Average training accuracy score:",(np.mean(training_accuracy)*100).round(3))    #Average training accuracy score: 95.533
print("Average test accuracy score:",(np.mean(test_accuracy)*100).round(3))            #Average test accuracy score: 95.218

from sklearn.svm import SVC
svc =SVC(degree=3,kernel='poly')  #Polynomial function
svc.fit(X,Y)
X =df.iloc[:,:2]
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X.values,
                      y=Y.values,
                      clf=svc,
                      legend=4)

#######################################################
#Radial Basis Function
#SVM
from sklearn.svm import SVC
svc =SVC(degree=3,kernel='rbf')  #Polynomial function
svc.fit(X_train,Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_test = svc.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("training accuracy score:",ac1.round(3))    #training accuracy score: 0.945
ac2 = accuracy_score(Y_test,Y_pred_test)
print("testing accuracy score:",ac2.round(3))     #testing accuracy score: 0.982

#Validation set approach
training_accuracy = []
test_accuracy = []
for i in range (1,101):
    X_train,X_test,Y_train,Y_test=train_test_split(df_final,Y,train_size=0.75,random_state=i)
    svc.fit(X_train,Y_train)
    Y_pred_train = svc.predict(X_train)
    Y_pred_test = svc.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

print("Average training accuracy score:",(np.mean(training_accuracy)*100).round(3))    #Average training accuracy score: 96.976
print("Average test accuracy score:",(np.mean(test_accuracy)*100).round(3))            #Average test accuracy score: 95.236

#==================================================================================
from sklearn.svm import SVC
svc =SVC(degree=3,kernel='rbf')  #Polynomial function
svc.fit(X,Y)

X =df_final.iloc[:,:2]
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X.values,
                      y=Y.values,
                      clf=svc,
                      legend=4)

#===========================================================================================
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
X_visual = df_final.iloc[:, :2]
plot_decision_regions(X=X_visual.values, y=Y.values, clf=svc, legend=4)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundaries (RBF Kernel)')
plt.show()

#============================================================================================