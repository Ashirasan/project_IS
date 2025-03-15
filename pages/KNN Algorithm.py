import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
st.set_page_config(page_title="KNN Model", page_icon=":shark:", layout="wide")

st.title("KNN Model")
df = pd.read_csv("./dataset/unclean_credit_scoring.csv")
st.write("This data is from a ChatGPT generated dataset")
st.write("The data is uncleaned and contains missing values")
st.write(df)   

st.write("Next, we will clean the data and build a KNN model")
# clean data
df["Credit_Score"] = pd.to_numeric(df["Credit_Score"], errors="coerce")
df.drop_duplicates(inplace=True)
imputer = SimpleImputer(strategy="median")
df[["Age", "Credit_Score"]] = imputer.fit_transform(df[["Age", "Credit_Score"]])
df = df[df["Income"] < 500000]
# show clean data
st.write("Data cleaned")
st.write(df)


# scale
scaler = MinMaxScaler()
df[["Age", "Income", "Credit_Score", "Loan_Amount"]] = scaler.fit_transform(df[["Age", "Income", "Credit_Score", "Loan_Amount"]])
X = df[["Age", "Income", "Credit_Score", "Loan_Amount"]]
y = df["Default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# predict
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# acc f1
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

# plot cmatrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots()
cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)
for (i, j), val in np.ndenumerate(conf_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)


# input
import random
st.write("## Predict Default")
age = st.number_input("Age", min_value=0, max_value=80, value=random.randint(0, 80))
income = st.number_input("Income", min_value=0, max_value=500000, value=random.randint(0, 500000))
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=random.randint(0, 1000))
loan_amount = st.number_input("Loan Amount", min_value=0, max_value=1000000, value=random.randint(0, 1000000))

# scale
input_data = np.array([[age, income, credit_score, loan_amount]])
input_data = pd.DataFrame(input_data, columns=["Age", "Income", "Credit_Score", "Loan_Amount"])
input_data_scaled = scaler.transform(input_data)

# st.write(X_test[:6])
# st.write(input_data_scaled)
# predict input
if st.button("Predict"):
    st.write("Predict Data:", input_data)
    prediction = knn.predict(input_data_scaled)
    st.write("Prediction (0 = No Default, 1 = Default):", prediction)