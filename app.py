import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

st.title('The Boston Housing')

boston_df = pd.read_csv("BostonHousing.csv")


tax_filter_min = st.slider('min tax falue to filtr: ', boston_df['tax'].min(), boston_df['tax'].max(), 296)
tax_filter_max = st.slider('max tax falue to filtr: ', boston_df['tax'].min(), boston_df['tax'].max(), 396)

boston_df = boston_df[(boston_df['tax'] >= tax_filter_min) & (boston_df['tax'] <= tax_filter_max)]

if st.checkbox('Show only hous near the river'):
    boston_df[boston_df['chas']==1]
else:
    boston_df


_str = '## Plot of medv feture distribution'
st.markdown(_str)

fig, ax = plt.subplots()
ax.hist(boston_df['medv'], bins=30, edgecolor='black')
ax.set_xlabel('Price of houses')
ax.set_ylabel('Number of houses')
ax.set_title('medv distrybution')
st.pyplot(fig)


with st.form(key='my_form'):
    crim_input = st.number_input('crim:')
    zn_input = st.number_input('zn:')
    indus_input = st.number_input('indus:')
    submit_btn = st.form_submit_button('Predict!')
    
if submit_btn:
    loaded_lr = joblib.load('linear_regression_model.joblib')
    input = np.array([crim_input, zn_input, indus_input])
    pred = loaded_lr.predict(input.reshape(1, -1))
    st.write(f'MEDV prediction for inputet example is: {pred[0]:.2f}')
