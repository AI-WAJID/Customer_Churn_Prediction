import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_gender.pkl', 'rb') as file:
    onehot_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set page configuration and dark theme background with scroll fix
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.markdown("""
    <style>
        /* Dark background for the whole app */
        .stApp {
            background-color: #121212;
            color: #e0e0e0;
        }

        /* Container for the form and outputs with dark card style */
        .main {
            max-width: 700px;
            margin: auto;
            background-color: #1e1e1e;
            border-radius: 12px;
            padding: 2rem 2.5rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.9);
            /* Enable smooth scrolling and proper height */
            height: auto !important;
            overflow-y: auto !important;
            max-height: 100% !important;
            padding-bottom: 4rem !important;
        }

        /* Streamlit buttons with bright accent */
        div.stButton > button {
            background-color: #2978b5;
            color: white;
            font-weight: 700;
            padding: 0.7rem 1.5rem;
            font-size: 1.1rem;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #5390d9;
        }

        /* Inputs and sliders styling for dark mode */
        label, .css-1aumxhk, .css-k1vhr4, .css-14xtw13 {
            color: #e0e0e0 !important;
        }
        .css-18e3th9 {
            background-color: #1e1e1e !important;
        }
        .stTextInput>div>div>input {
            background-color: #333 !important;
            color: #eee !important;
        }
        .stNumberInput>div>div>input {
            background-color: #333 !important;
            color: #eee !important;
        }
        .css-14xtw13 {
            background-color: #333 !important;
        }
        /* Slider track and handle */
        .stSlider > div[data-baseweb] > div {
            background-color: #2978b5 !important;
            border-radius: 8px !important;
        }
        .stSlider > div[data-baseweb] > div > div > div {
            background-color: #5390d9 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main content container
with st.container():
    st.title('📉 Customer Churn Prediction')
    st.write('Fill in the details below and click **Predict** to check churn likelihood.')

    with st.form(key="churn_form"):
        col1, col2 = st.columns(2)
        with col1:
            geo_categories = onehot_encoder_gender.categories_[0]
            geography = st.selectbox('🌍 Geography', geo_categories)
            gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
            age = st.slider('🎂 Age', 18, 92)
            tenure = st.slider('⏳ Tenure (years)', 0, 10)
            num_of_products = st.slider('📦 Number Of Products', 1, 4)
        with col2:
            credit_score = st.number_input('💳 Credit Score', min_value=0, max_value=1000, step=1, format="%d")
            balance = st.number_input('🏦 Balance', min_value=0.0, format="%.2f")
            estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, format="%.2f")
            has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            is_active_member = st.selectbox('🟢 Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        geo_encode = onehot_encoder_gender.transform([[geography]])
        geo_encode_df = pd.DataFrame(geo_encode, columns=[f"Geography_{cat}" for cat in geo_categories])
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encode_df], axis=1)

        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        if prediction_proba > 0.5:
            st.error(f"⚠️ The customer is likely to churn (Probability: {prediction_proba:.2f}).")
        else:
            st.success(f"✅ The customer is not likely to churn (Probability: {prediction_proba:.2f}).")
    else:
        st.info("Please fill in the details and click **Predict** to see the result.")
