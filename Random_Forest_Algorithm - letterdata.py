# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:18:37 2024

@author: Priyanka
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

letter=pd.read_csv("C:/Data Set/letterdata.csv")
letter.dtypes
letter.head
letter.columns
letter.shape
plt.figure(1,figsize=(16,10))
sns.countplot(x=letter['letter']) 
#Letter U is definedmultiple times
sns.countplot(x=letter['onpix'])
#Data is right skewed and 2 has the highst value
sns.displot(letter.height)
#data is normal and slight right skewed
sns.boxplot(letter.width)
#There are several Outliers
###########################################################
#Now let us check highest fire in KM?
letter.sort_values(by='height',ascending=False).head(5)

highest_fire_area=letter.sort_values(by='width',ascending=True)

plt.figure(figsize=(8,6))
plt.title('Temperature vs Area of fire')
plt.bar(highest_fire_area['hight'],
         highest_fire_area['width'])

plt.xlabel('Height')
plt.ylabel('Width')
plt.show()



