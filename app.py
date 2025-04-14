import streamlit as st
import joblib
import numpy as np

# Load scaler (fit on 4 features) and model (trained on all 9)
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# App title
st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

# Input fields for numeric features
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
subscription_length = st.number_input("Enter Subscription Length (Months)", min_value=0, max_value=130, value=12)
monthly_bill = st.number_input("Enter Monthly Bill", min_value=30, max_value=300, value=100)
total_usage = st.number_input("Enter Total Usage (in GB)", min_value=0, max_value=1000, value=50)

# Input fields for categorical features
gender = st.selectbox("Select Gender", ["Male", "Female"])
location = st.selectbox("Select Location", ["Houston", "Los Angeles", "Miami", "New York"])

st.divider()

# Predict button
predictbutton = st.button("Predict!")

if predictbutton:
    # Convert gender to binary
    gender_male = 1 if gender == "Male" else 0

    # One-hot encode location
    location_houston = 1 if location == "Houston" else 0
    location_los_angeles = 1 if location == "Los Angeles" else 0
    location_miami = 1 if location == "Miami" else 0
    location_new_york = 1 if location == "New York" else 0

    # Scale the 4 numerical features
    numerical = np.array([[age, subscription_length, monthly_bill, total_usage]])
    scaled_numerical = scaler.transform(numerical)

    # Combine with binary features
    final_input = np.concatenate([
        scaled_numerical[0],  # Flatten the 2D scaled array
        [gender_male, location_houston, location_los_angeles, location_miami, location_new_york]
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_input)[0]
    predicted = "Yes" if prediction == 1 else "No"

    st.success(f"Churn Prediction: **{predicted}**")

else:
    st.info("Please enter the values and press the predict button.")
