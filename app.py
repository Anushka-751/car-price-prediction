import streamlit as st
import joblib
import pandas as pd

model = joblib.load('models/car_price_model.pkl')

st.title(" Car Price Prediction App")

# User input
year = st.slider("Year", 1990, 2024, 2015)
present_price = st.number_input("Present Price (in Lakhs)", value=5.0)
driven_kms = st.number_input("Kilometers Driven", value=50000)

fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

# Encode input
fuel_map = {'CNG': 0, 'Diesel': 1, 'Petrol': 2}
seller_map = {'Dealer': 0, 'Individual': 1}
trans_map = {'Automatic': 0, 'Manual': 1}

input_data = pd.DataFrame([[year, present_price, driven_kms,
                            fuel_map[fuel_type],
                            seller_map[seller_type],
                            trans_map[transmission],
                            owner]],
                          columns=['Year', 'Present_Price', 'Driven_Kms',
                                   'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])

if st.button("Predict Selling Price"):
    price = model.predict(input_data)
    st.success(f"Estimated Selling Price: ₹{price[0]:,.2f} Lakhs")
