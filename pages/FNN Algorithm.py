import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# import tensorflow as tf
# from tensorflow import keras
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

st.title("FeedForward Neural Network")
df = pd.read_csv("./dataset/wine_quality_unclean.csv")
st.write("This data is from a ChatGPT generated dataset")
st.write("The data is uncleaned and contains missing values")
st.write(df)

st.write("Next, we will clean the data and build a KNN model")
df["quality"] = pd.to_numeric(df["quality"], errors="coerce")
df.drop_duplicates(inplace=True)
df = df[df["quality"] > 1]
df.dropna(inplace=True)
st.write(df)

min_max_scaler = MinMaxScaler()
quality = df["quality"]
df = df.drop("quality", axis=1)
df = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
# st.write(df)

X = df
y = quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter tuning using GridSearchCV
# param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['adam', 'sgd'],
#     'max_iter': [500, 1000]
# }

# mlp = MLPClassifier(random_state=42)
# grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# best_mlp = grid_search.best_estimator_
# best_params = grid_search.best_params_
# st.write("Best Parameters:", best_params)

# Train
best_mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(100,50), max_iter=500, solver='sgd', random_state=42)
best_mlp.fit(X_train, y_train)

# Predict
y_train_pred = best_mlp.predict(X_train)
y_test_pred = best_mlp.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)



train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)
st.header("Train Report:")
st.write("Train Accuracy:", train_acc)
st.text(train_report)
st.header("Test Report:")
st.write("Test Accuracy:", test_acc)
st.text(test_report)

import matplotlib.pyplot as plt
import numpy as np
import random

# input
st.write("## Predict Quality")
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=15.0, value=random.uniform(0.0, 15.0))
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=random.uniform(0.0, 2.0))
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=random.uniform(0.0, 1.0))
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=15.0, value=random.uniform(0.0, 15.0))
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=random.uniform(0.0, 1.0))
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=random.uniform(0.0, 100.0))
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=random.uniform(0.0, 300.0))
density = st.number_input("Density", min_value=0.0, max_value=1.0, value=random.uniform(0.0, 1.0))
pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=random.uniform(0.0, 14.0))
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=random.uniform(0.0, 2.0))
alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, value=random.uniform(0.0, 20.0))

quality_input = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
quality_input = pd.DataFrame(quality_input, columns=X.columns)
quality_input_scaled = min_max_scaler.transform(quality_input)

if st.button("Predict"):
    prediction = best_mlp.predict(quality_input_scaled)
    st.write("Predict (Data): ", quality_input)
    st.write("Prediction (Quality): ", prediction)