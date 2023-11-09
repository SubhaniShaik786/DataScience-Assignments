# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:11:59 2023

"""

## Load the dataset
import numpy as np
import pandas as pd
df = pd.read_csv("glass.csv")
df.head()
df.shape   #(214, 10)
df.info()

#EDA
#BOXPLOT AND OUTLIERS CALCULATION #
df1 = df.iloc[:,0:10]
from scipy import stats
# Define a threshold for Z-score (e.g., Z-score greater than 3 or less than -3 indicates an outlier)
z_threshold = 3
# Calculate the Z-scores for each column in the DataFrame
z_scores = np.abs(stats.zscore(df1))

# Create a mask to identify rows with outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)

# Remove rows with outliers from the DataFrame
df = df[~outlier_mask]
df.shape  #(194, 10))

# Now, df contains the data with outliers removed

# Data Preprocessing
# Check for missing values (if any)
df.isnull().sum()  #there are no missing values

#continous variables
X = df.iloc[:,0:9]
X

Y = df["Type"]
Y

#DATA PARTITION
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75)

#Standardisation 
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
SS_X = SS.fit_transform(X_train,Y_train)



#MODEL FITTING KNN 
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(X_train,Y_train)
Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

#METRICS
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#ACCURACY SCORE
ac1 = accuracy_score(Y_train,Y_pred_train)
print("KNN-Training accuracy score:",(ac1).round(3))     #KNN-Training accuracy score: 0.786
ac2 = accuracy_score(Y_test,Y_pred_test)
print("KNN-Test accuracy score:",(ac2).round(3))         #KNN-Test accuracy score: 0.612

#CONFUSION MATRIX
conf_matrix_1 = confusion_matrix(Y_train, Y_pred_train)
print("Confusion Matrix for training:\n", conf_matrix_1)
conf_matrix_2 = confusion_matrix(Y_test, Y_pred_test)
print("Confusion Matrix for testing:\n", conf_matrix_2)

#CLASSIFICATION REPORT
class_report_1 = classification_report(Y_train, Y_pred_train)
print("Classification Report for training:\n", class_report_1)
class_report_1 = classification_report(Y_test, Y_pred_test)
print("Classification Report for testing:\n", class_report_1)

#VALIDTAION APPROACH FOR KNN
l1 = [] 
l2 = []
training_accuracy=[]
test_accuracy=[]
for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=i)
    KNN = KNeighborsClassifier(n_neighbors=15,p=2,) #best k value = 
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test = KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
print("Average Trianing accuracy :",np.mean(training_accuracy).round(3)) #Average Trianing accuracy : 0.677
print("Average Test accuarcy :",np.mean(test_accuracy).round(3))        #Average Test accuarcy : 0.646

#Average accuracies are getting stored 
l1.append(np.mean(training_accuracy).round(3))
l2.append(np.mean(test_accuracy).round(3))
print(l1)
print(l2)

#subtracting two list by converting into arrays
l1 
l2  

array1= np.array(l1)
array1
array2= np.array(l2)
array2
deviation = np.subtract(array1,array2)
deviation
list(deviation.round(3))

#Graph between accuracy score and k-value
import matplotlib.pyplot as plt
plt.scatter(range(5,27,2),l1)
plt.plot(range(5,27,2),l1,color='black')
plt.scatter(range(5,27,2),l2,color='red')
plt.plot(range(5,27,2),l2,color='black')
plt.show()