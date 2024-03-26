# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:17:47 2024

@author: Priyanka
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
forest=pd.read_csv("C:/Data Set/forestfires.csv")
forest.head()
forest.dtypes

#EDA 

forest.shape
plt.figure(1,figsize=(16,10))
sns.countplot(x=forest['month']) 
#Aug and Sept has highest value
sns.countplot(x=forest['day'])
#Friday And Saturday has highest value
sns.displot(forest.FFMC)
#data is normal and slight left skewed
sns.boxplot(forest.FFMC)
#There are several Outliers

sns.displot(forest.RH)
#data is normal and slighly left skewed
sns.boxplot(forest.RH)
#There are outliers

sns.displot(forest.wind)
#Data is normal and slightly right
sns.boxplot(forest.wind)

sns.displot(forest.rain)
#Data is normal
sns.boxplot(forest.rain)
#There are outliers

###########################################################
#Now let us check highest fire in KM?
forest.sort_values(by='area',ascending=False).head(5)

highest_fire_area=forest.sort_values(by='area',ascending=True)

plt.figure(figsize=(8,6))

plt.title('Temperature vs Area of fire')
plt.bar(highest_fire_area['temp'],
        highest_fire_area['area'])

plt.xlabel('Temperature')
plt.ylabel('Area as per km-sq')
plt.show()

#Once the fire starts ,almost 1000+ sq area's
#temperature goes beyound 25 and 
#around 750 km area facing temp 30+
#Now let us check the highest rain i the forest

highest_rain=forest.sort_values(by='rain',ascending=False)[['month','day','rain']].head(5)
highest_rain

#Highest rain is observed in the month of aug
#Let us check highest and lowest temperature in the month 

highest_temp=forest.sort_values(by='temp',ascending=False)[['month','day','rain']].head(5)
highest_rain

lowest_temp=forest.sort_values(by='temp',ascending=True)[['month','day','rain']].head(5)
lowest_temp

print('Highest temperature',highest_temp)
#Highest temp observed in aug
print('Lowest Temperature',lowest_temp)
#Lowest temperature observed mostly in dec and feb

forest.isna().sum()
#There are no null values for the columns

##############################################################
#sal1.dtypes

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
forest.month=labelencoder.fit_transform(forest.month)
forest.day=labelencoder.fit_transform(forest.day)
forest.size_category=labelencoder.fit_transform(forest.size)

forest.dtypes

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['month'])
df_t=winsor.fit_transform(forest[['month']])
sns.boxplot(df_t)
#Do this for every column like area,temp,rain
#################################################################
tc=forest.corr()
tc
fig,ax=plt.subplots()
fig.set_size_inches(200,10)
sns.heatmap(tc,annot=True,cmap='YlGnBu')
#all variables are moderately correlated with the size with size_correlation

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(forest,test_size=0.3)
train_X=train.iloc[::,30]
train_y=train.iloc[::,30]
test_X=test.iloc[::,30]
test_y=test.iloc[::,30]
 
'''#kernel linear
model_linear=SVC(kernel="linear")
model_linear.fit(train_X,train_y) 
pred_test_linear=model_linear.predict(test_X)
np.mean(pred_test_linear=test_y)
'''
# kernel linear 
model_linear = SVC(kernel="linear") 
model_linear.fit(train_X, train_y) 
pred_test_linear = model_linear.predict(test_X) 
accuracy = np.mean(pred_test_linear == test_y)
print("Accuracy:", accuracy)
 
#RBF
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train_X,train_y)
pred_test_rbf=model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y)


