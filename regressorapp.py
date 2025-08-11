import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained regression model
model = tf.keras.models.load_model('regression_model.h5')

# Load encoders and scaler
with open('regressoe_label_encoder_gender.pkl', 'rb') as file:
    regressoe_label_encoder_gender = pickle.load(file)

with open('regressor_onehot_encoder_geo.pkl', 'rb') as file:
    regressor_onehot_encoder_geo = pickle.load(file)

with open('regressor_scaler.pkl', 'rb') as file:
    regressor_scaler = pickle.load(file)

# Dark theme UI setup
st.set_page_config(page_title="Estimated Salary Predictor", page_icon="💵", layout="centered")

st.markdown("""
    <style>
        .stApp {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main {
            max-width: 720px;
            margin: 2rem auto;
            background-color: #1e1e1e;
            border-radius: 14px;
            padding: 2.5rem 3rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.9);
        }
        div.stButton > button {
            background-color: #00b894;
            color: white;
            font-weight: 700;
            padding: 0.7rem 2rem;
            font-size: 1.15rem;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #019875;
        }
        label, .css-1aumxhk, .css-k1vhr4, .css-14xtw13 {
            color: #e0e0e0 !important;
        }
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .css-14xtw13 {
            background-color: #333 !important;
            color: #eee !important;
            border-radius: 8px;
            padding: 0.3rem 0.6rem;
        }
        .css-18e3th9 {
            background-color: #1e1e1e !important;
        }
        .stSlider > div[data-baseweb] > div {
            background-color: #00b894 !important;
            border-radius: 8px !important;
        }
        .stSlider > div[data-baseweb] > div > div > div {
            background-color: #019875 !important;
        }
        html, body, .stApp, .main {
            height: auto !important;
            overflow-y: auto !important;
            max-height: 100% !important;
        }
        .main {
            padding-bottom: 4rem !important;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.title('💵 Estimated Salary Predictor')
    st.write('Fill out the customer details and click Predict to estimate the salary.')

    with st.form(key="Salary_Form"):
        col1, col2 = st.columns(2)
        with col1:
            geography_categories = regressor_onehot_encoder_geo.categories_[0]
            geography = st.selectbox('🌍 Geography', geography_categories)
            gender = st.selectbox('👤 Gender', regressoe_label_encoder_gender.classes_)
            age = st.slider('🎂 Age', 18, 92)
            tenure = st.slider('⏳ Tenure (years)', 0, 10)
            num_of_products = st.slider('📦 Number Of Products', 1, 4)
            exited = st.selectbox('🚪 Exited', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        with col2:
            credit_score = st.number_input('💳 Credit Score', min_value=0, max_value=1000, step=1, format="%d")
            balance = st.number_input('🏦 Balance', min_value=0.0, format="%.2f")
            has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            is_active_member = st.selectbox('🟢 Is Active Member', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

        submit_button = st.form_submit_button(label="Predict Salary")

    if submit_button:
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [regressoe_label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'Exited': [exited]
        })

        geo_encode = regressor_onehot_encoder_geo.transform([[geography]])

        # Fix: if encoder outputs fewer columns than categories, slice categories accordingly
        n_cols = geo_encode.shape[1]
        safe_geo_categories = geography_categories[:n_cols]

        geo_encode_df = pd.DataFrame(geo_encode, columns=[f"Geography_{cat}" for cat in safe_geo_categories])
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encode_df], axis=1)

        input_data_scaled = regressor_scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        prediction_salary = prediction

        st.success(f"💰 Predicted Estimated Salary: ${prediction_salary:.2f}")
    else:
        st.info("Please enter details and click **Predict Salary** to get an estimate.")
