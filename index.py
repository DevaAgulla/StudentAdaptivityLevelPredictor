#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
student_df = pd.read_csv("students_adaptability_level_online_education.csv")

# Check the data
print(student_df.describe())
print(student_df.isnull().sum())

# Encode categorical features
le = LabelEncoder()
student_df.Gender = le.fit_transform(student_df.Gender)
student_df["Age"] = le.fit_transform(student_df["Age"])
student_df["Education Level"] = le.fit_transform(student_df["Education Level"])
student_df["Institution Type"] = le.fit_transform(student_df["Institution Type"])
student_df["IT Student"] = le.fit_transform(student_df["IT Student"])
student_df["Location"] = le.fit_transform(student_df["Location"])
student_df["Load-shedding"] = le.fit_transform(student_df["Load-shedding"])
student_df["Financial Condition"] = le.fit_transform(student_df["Financial Condition"])
student_df["Internet Type"] = le.fit_transform(student_df["Internet Type"])
student_df["Network Type"] = le.fit_transform(student_df["Network Type"])
student_df["Class Duration"] = le.fit_transform(student_df["Class Duration"])
student_df["Self Lms"] = le.fit_transform(student_df["Self Lms"])
student_df["Device"] = le.fit_transform(student_df["Device"])
student_df["Adaptivity Level"] = le.fit_transform(student_df["Adaptivity Level"])

# Splitting the data
X = student_df.drop('Adaptivity Level', axis=1)
y = student_df['Adaptivity Level']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, Y_train)

# Predictions
predictions = rfc.predict(X_test)
pred_train = rfc.predict(X_train)

# Print metrics
print(confusion_matrix(predictions, Y_test))
print(classification_report(Y_test, predictions))
print("Accuracy_test:", accuracy_score(predictions, Y_test))
print("Accuracy_train:", accuracy_score(pred_train, Y_train))

# Save the model to a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(rfc, file)

print("Model saved to model.pkl")

