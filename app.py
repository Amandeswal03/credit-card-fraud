import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = pickle.load(open("random_forest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Credit Card Fraud Detection")

# Input features
st.write("Enter the transaction details:")
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
# Add inputs for all required features

if st.button("Predict"):
    # Preprocess input
    input_data = pd.DataFrame([[feature1, feature2, ...]], columns=["Feature1", "Feature2", ...])
    input_data["NormalizedAmount"] = scaler.transform(input_data[["Amount"]])
    input_data.drop(["Amount", "Time"], axis=1, inplace=True)

    # Prediction
    prediction = model.predict(input_data)[0]
    result = "Fraudulent Transaction" if prediction == 1 else "Genuine Transaction"
    st.success(result)
