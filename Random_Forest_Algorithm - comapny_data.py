# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:34:05 2024

@author: Priyanka
"""

"""
3.1.A cloth manufacturing company is interested to know about the 
different attributes contributing to high sales. Build a decision 
tree & random forest model with Sales as target variable (first 
convert it into categorical variable).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("C:/Data Set/Company_Data.csv")

# Convert Sales into categorical variable
data['Sales'] = pd.cut(data['Sales'], bins=[0, 100, 200, 300, 400, float('inf')], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Define features and target variable
X = data.drop(['Sales', 'Income'], axis=1)
y = data['Income']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder

# Convert categorical labels to numerical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Build Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train_encoded)
dt_pred = dt_model.predict(X_test)

# Build Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train_encoded)
rf_pred = rf_model.predict(X_test)


# Build Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train) 
dt_pred = dt_model.predict(X_test)

# Evaluate Decision Tree model
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Classification Report:")
print(classification_report(y_test, dt_pred))

# Build Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest model
print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:")
print(classification_report(y_test, rf_pred))


