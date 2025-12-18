import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

with open('Term deposit subscription Prediction.pkl', 'rb') as ft:
    model = pickle.load(ft)
   
expected_cols = model.feature_names_in_

st.title(':red[Term Deposit Subscription Predictor]')  

st.header("Enter customer details: ")

st.header("Enter Customer Details:")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                           'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
marital = st.selectbox("Marital Status", ['divorced', 'married', 'single'])
education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox("Has credit in default?", ['no', 'yes'])
balance = st.number_input("Account Balance", value=0)
housing = st.selectbox("Has housing loan?", ['no', 'yes'])
loan = st.selectbox("Has personal loan?", ['no', 'yes'])
contact_day = st.number_input("Contact Day of Month", min_value=1, max_value=31, value=15)
contact_month = st.selectbox("Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
duration = st.number_input("Last Contact Duration (seconds)", value=100)
campaign = st.number_input("Number of Contacts in this Campaign", value=1)
pdays = st.number_input("Days Passed After Previous Campaign (-1 means never contacted)", value=-1)
previous = st.number_input("Number of Contacts Before this Campaign", value=0)


if st.button("Predict"):
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact_day': contact_day,
        'contact_month': contact_month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous
    }

    input_df = pd.DataFrame([input_dict])

binary_map = {'yes': 1,'no': 0}
for col in ['default', 'housing', 'loan']:
    input_df[col] = input_df[col].map(binary_map)

input_df = pd.get_dummies(input_df,drop_first=True)

for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[expected_cols]

result = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

st.success(f"Subscribed: {'Yes ✅' if result == 1 else 'No ❌'}")
st.info(f"Prediction Confidence: {prob*100:.2f}%")