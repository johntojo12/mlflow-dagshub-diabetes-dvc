import streamlit as st
import pandas as pd
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    return input_df

# Streamlit app
def main():
    st.title("Model Deployment with Streamlit")
    
    # Input fields for user to enter data
    st.header("Enter Data for Prediction")
    input_data = {
        'Pregnancies': st.number_input('Pregnancies', value=0),
        'Glucose': st.number_input('Glucose', value=0),
        'BloodPressure': st.number_input('Blood Pressure', value=0),
        'SkinThickness': st.number_input('Skin Thickness', value=0),
        'Insulin': st.number_input('Insulin', value=0),
        'BMI': st.number_input('BMI', value=0.0),
        'DiabetesPedigreeFunction': st.number_input('Diabetes Pedigree Function', value=0.0),
        'Age': st.number_input('Age', value=0)
    }

    # Preprocess input data
    input_df = preprocess_input(input_data)

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.success("The prediction is: Positive")
        else:
            st.success("The prediction is: Negative")

if __name__ == '__main__':
    main()