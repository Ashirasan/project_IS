import streamlit as st
import pandas as pd

st.title("Machine Learning")
st.write("Use two algorithm to predict the default of credit scoring")
st.write("1. KNN Algorithm")
st.write("2. SVM Algorithm")
st.header("Process to train the model")
st.write("1. Data Preprocessing \n - Import the data \n - Check the missing value \n - Check the data type")
st.write("2. Data Splitting \n - Scaling data \n - Split the data into training and testing data")
st.write("3. Model Training \n - Train the model using the training data")
st.write("4. Model Evaluation \n - Evaluate the model using the testing data")
st.write("5. Calculate the accuracy of the model")

st.header("Why use KNN and SVM Algorithm?")
st.write("1. To predict the default of credit scoring, we can use classification model, and KNN and SVM are powerful model for classification problem.")
st.write("2. Data need to preprocess, because the data contains missing values and uncleaned data.")
st.write("3. Dataset size is small, so KNN and SVM are suitable for this dataset.")

st.header("Dataset for Machine Learning")
st.write("The dataset is from ChatGPT generated dataset")
st.write("The data is uncleaned and contains missing values")
df = pd.read_csv("./dataset/unclean_credit_scoring.csv")
st.write(df)
