# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

# ---------------------------
# Load and preprocess the data
# ---------------------------

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('bank-additional.csv', sep=";")
    
    encoder = LabelEncoder()
    categorical_columns = ['job', 'day_of_week', 'month', 'contact', 'loan', 'housing',
                           'marital', 'default', 'poutcome', 'education', 'y']
    
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])
    
    return data

data = load_data()

# ---------------------------
# Train the model
# ---------------------------

X = data.drop(columns='y')
Y = data['y']

# Build pipeline with Lasso (best params from your grid search)
model = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.01, fit_intercept=True, max_iter=1000, selection='random', tol=0.01)
)

model.fit(X, Y)

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("Bank Term Deposit Subscription Prediction")
st.write("Enter the client's information below:")

# Input fields for each feature
age = st.number_input('Age', min_value=18, max_value=100, value=30)
job = st.selectbox('Job (encoded)', list(data['job'].unique()))
marital = st.selectbox('Marital status (encoded)', list(data['marital'].unique()))
education = st.selectbox('Education (encoded)', list(data['education'].unique()))
default = st.selectbox('Has credit in default? (encoded)', list(data['default'].unique()))
housing = st.selectbox('Has housing loan? (encoded)', list(data['housing'].unique()))
loan = st.selectbox('Has personal loan? (encoded)', list(data['loan'].unique()))
contact = st.selectbox('Contact communication type (encoded)', list(data['contact'].unique()))
month = st.selectbox('Last contact month of year (encoded)', list(data['month'].unique()))
day_of_week = st.selectbox('Last contact day of the week (encoded)', list(data['day_of_week'].unique()))
duration = st.number_input('Last contact duration (seconds)', min_value=0, value=100)
campaign = st.number_input('Number of contacts performed during this campaign', min_value=0, value=1)
pdays = st.number_input('Days passed after last contact from a previous campaign', min_value=-1, value=999)
previous = st.number_input('Number of contacts performed before this campaign', min_value=0, value=0)
poutcome = st.selectbox('Outcome of the previous marketing campaign (encoded)', list(data['poutcome'].unique()))
emp_var_rate = st.number_input('Employment variation rate', value=-1.8)
cons_price_idx = st.number_input('Consumer price index', value=92.893)
cons_conf_idx = st.number_input('Consumer confidence index', value=-46.2)
euribor3m = st.number_input('Euribor 3 month rate', value=1.313)
nr_employed = st.number_input('Number of employees', value=5000.0)

# Arrange input as numpy array
input_data = np.array([job, age, marital, education, default, housing, loan, contact,
                       month, day_of_week, duration, campaign, pdays, previous, poutcome,
                       emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed])

input_data_reshaped = input_data.reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data_reshaped)
    predicted_class = int(prediction >= 0.5)
    
    if predicted_class == 1:
        st.success("✅ The client **WILL SUBSCRIBE** to a term deposit.")
    else:
        st.warning("❌ The client **WILL NOT SUBSCRIBE** to a term deposit.")
