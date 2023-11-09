# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:32:01 2023

@author: sksha
"""

import numpy as np
import pandas as pd
df = pd.read_csv("crime_data.csv")
df
df.info()
df.isnull().sum()
pd.set_option('display.max_columns', None)
df
df.shape #(50, 5)

#=================================================================================
# EDA #
#EDA
#BOXPLOT AND OUTLIERS CALCULATION #
df1 = df.iloc[:,1:5]
from scipy import stats
# Define a threshold for Z-score (e.g., Z-score greater than 3 or less than -3 indicates an outlier)
z_threshold = 3
# Calculate the Z-scores for each column in the DataFrame
z_scores = np.abs(stats.zscore(df1))

# Create a mask to identify rows with outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)

# Remove rows with outliers from the DataFrame
df = df[~outlier_mask]
df.shape  #(3630, 12)

# Now, df contains the data with outliers removed

#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
import seaborn as sns
df["Murder"].hist()
sns.distplot(df["Murder"])
df["Murder"].skew()
df["Murder"].kurt()
df["Murder"].describe()

df["Assault"].hist()
sns.distplot(df["Assault"])
df["Assault"].skew()
df["Assault"].kurt()
df["Assault"].describe()

df["UrbanPop"].hist()
sns.distplot(df["UrbanPop"])
df["UrbanPop"].skew()
df["UrbanPop"].kurt()
df["UrbanPop"].describe()

df["Rape"].hist()
sns.distplot(df["Rape"])
df["Rape"].kurt()
df["Rape"].skew()
df["Rape"].describe()


# understanding the relationships between all the four variables#

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df, vars=['Murder', 'Assault', 'UrbanPop', 'Rape']) #to check relationship between more than 1 variables
plt.show()

"""# we can find Positive or Negative Relationships
#correlation
#outliers
#Histograms
#Outliers from the above code between all the four variables instead of doing scatter plot"""

correlation_matrix = df.corr()
#Heat Map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

"""Values close to 1 indicate a strong positive correlation.
Values close to -1 indicate a strong negative correlation.
Values close to 0 indicate a weak or no correlation"""

X = df.iloc[:,1:5].values
X.shape

# transformation on the data #
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)

#construction of dendogram
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(X,method = 'complete'))

#====================================================================================================
#AgglomerativeClustering
#forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',linkage = 'complete')
Y = cluster.fit_predict(X)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

#Scatter plot
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], cmap='rainbow')
plt.xlabel('X-axis Label for Feature 1')
plt.ylabel('Y-axis Label for Feature 2')
plt.title('Scatter Plot with Rainbow Colormap')
plt.colorbar(label='Feature 3 Value')  # Add a colorbar to indicate Feature 3 values
plt.show()

#====================================================================================================
#performing k means on the same data
#KMeans (Elbow method)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
Y = kmeans.fit_predict(X)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

kmeans.inertia_

kresults = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_predict(X)
    kresults.append(kmeans.inertia_)
    
kresults

#Scatterplot
import matplotlib.pyplot as plt
plt.scatter(x=range(1,11),y=kresults)
plt.plot(range(1,11),kresults,color="red")
plt.show()

"""according to the elbow method we can get clarity on upto  which k value should be choosen"""

#====================================================================================================
#DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN(eps = 1.0,min_samples = 2)
X = pd.DataFrame(SS_X)
db.fit(X)

Y = db.labels_
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

df

df_new = pd.concat([df,Y_new],axis=1)

df_final = df_new[df_new[0] != -1]
df_final.shape


df_noise = df_new[df_new[0] == -1]
df_noise

df_1 = df_new[df_new[0] == 1].value_counts()
df_1
