import streamlit as st
import pandas as pd

st.title("Neural Network")
st.write("FeedForward Neural Network model to predict the quality of wine")

st.header("Process to train the model")
st.write("1. Data Preprocessing \n - Import the data \n - Check the missing value \n - Check the data type")
st.write("2. Data Splitting \n - Scaling data \n - Split the data into training and testing data")
st.write("3. Model Design\n - Define the layers \n - Define the activation function \n - Define the optimizer")
st.write("4. Model Training \n - Train the model using the training data")
st.write("5. Model Evaluation \n - Evaluate the model using the testing data")
st.write("6. Calculate the accuracy of the model")

st.header("Why use FeedForward Neural Network Model?")
st.write("1. Quality of wine is a continuous variable, so we can use regression model to predict the quality of wine, and FeedForward Neural Network is a powerful model for regression problem.")
st.write("2. Data need to preprocess, because the data contains missing values and uncleaned data.")

st.header("Dataset for Neural Network")
st.write("The dataset is from ChatGPT generated dataset")
st.write("The data is uncleaned and contains missing values")
df = pd.read_csv("./dataset/wine_quality_unclean.csv")
st.write(df)
