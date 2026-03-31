import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses Info, Warning, and Error logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Stops the oneDNN floating-point message
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# load the trained model
model=tf.keras.models.load_model('model.h5')

# load the encoder and scaler
with open('label_encoder.pkl','rb') as file:
    label_encoder_=pickle.load(file)

with open('one_hot_encoder.pkl','rb') as file:
    onehot_encoder_=pickle.load(file)    

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)    

# steamlit app
st.title('Custommer Churn prediction')

# user input
geography = st.selectbox('Geography',onehot_encoder_.categories_[0] )
gender = st.selectbox('Gender',label_encoder_.classes_)
age = st.slider('Age', 18, 92, 40)
balance = st.number_input('Balance', 0.0, 250000.0, 60000.0)
credit_score = st.slider('Credit Score', 350, 850, 600)
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Is Active Member?', [0, 1])


# prepare input data
input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
    })

 # Apply One-Hot Encoding to 'Geography'
geo_input_df = pd.DataFrame([[geography]], columns=['Geography'])
geo_encoded = onehot_encoder_.transform(geo_input_df).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_.get_feature_names_out())

# combine onehot_encoded column with input data
input_data =pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# input_data = input_data[scaler.feature_names_in_]
# scale the input data
input_data_scaled=scaler.transform(input_data)

# prediction churn
prediction=model.predict(input_data_scaled)
prediction_proba =prediction[0][0]

st.write(f'Churn Probability:{prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('the customer is like to churn')
else :
    st.write('the customer is not likely to churn')