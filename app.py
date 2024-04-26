import streamlit as st
import pandas as pd
import pickle

st.write("""
# Diabetes Prediction Model

This app predicts the **Diabetis probability** by selected parameter!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pregnancies_week = st.sidebar.slider('Pregnancies Week', 0, 17, 6)
    Glucose_level = st.sidebar.slider('Glucose Level', 0, 199, 148)
    Blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    Skin_thickness = st.sidebar.slider('Skin Thinkness', 0, 99, 35)
    Insulin = st.sidebar.slider('Insulin', 0, 846, 0)
    BMI = st.number_input(label='BMI',step=1.,format="%.2f")
    DiabetesPedigreeFunction = st.number_input(label='DiabetesPedigreeFunction',step=1.,format="%.2f")
    Age = st.sidebar.slider('Age', 0, 100, 50)
    data = {'Pregnancies': Pregnancies_week,
            'Glucose': Glucose_level,
            'BloodPressure': Blood_pressure,
            'SkinThickness' : Skin_thickness,
            'Insulin' : Insulin,
            'BMI' : BMI,
            'DiabetesPedigreeFunction' : DiabetesPedigreeFunction,
            'Age ' : Age,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("diabetes.h5", "rb"))

prediction = loaded_model.predict(df)


st.subheader(' FuelConsumption Prediction')
st.write(prediction)
