# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:03:40 2024

@author: Priyanka
"""
"""
2.	 Divide the diabetes data into train and test datasets
 and build a Random Forest and Decision Tree model with 
 Outcome as the output variable. 
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes_data =pd.read_csv("C:/Data Set/Diabetes.csv")
X = diabetes_data[' Age (years)']
y = diabetes_data[' Diastolic blood pressure']
diabetes_data
# Convert target variable to binary classification task
y_binary = [1 if val > 140 else 0 for val in y]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Decision Tree model
X_array = X.values

# Reshape the array appropriately
X_reshaped = X_array.reshape(-1, 1)

# Now you can use X_reshaped to fit your model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_reshaped, y)
X_reshaped = X_array.reshape(-1, 1)
X_reshaped = X.to_numpy().reshape(-1, 1)
# Now you can use X_reshaped to fit your model
dt_model.fit(X_reshaped, y)    
dt_model.fit(X_train, y_train)
#dt_pred = dt_model.fit(X_test,y_test)
#dt_pred = dt_model.fit(X_test.reshape(-1, 1), y_test)
X_test_reshaped = X_test.to_numpy().reshape(-1, 1)

# Now you can use X_test_reshaped to fit your model
dt_pred = dt_model.fit(X_test_reshaped, y_test) 
y_pred = dt_model.predict(X_test_reshaped)  # Replace X_test_reshaped with your reshaped test feature data

# Now you can calculate accuracy using the predicted values
print("Accuracy:", accuracy_score(y_test, y_pred)) 
# Evaluate Decision Tree model
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, dt_pred))  
print("Classification Report:")
#print(classification_report(y_test, dt_pred))
from sklearn.metrics import classification_report

# Assuming dt_model is your fitted DecisionTreeClassifier model
# Use the predict method to obtain predicted values
y_pred = dt_model.predict(X_test_reshaped)  # Replace X_test_reshaped with your reshaped test feature data

# Now you can generate the classification report using the predicted values
print(classification_report(y_test, y_pred)) 

# Build Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest model
print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:")
print(classification_report(y_test, rf_pred))

