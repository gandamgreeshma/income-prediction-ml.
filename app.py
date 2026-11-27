
import streamlit as st
import pandas as pd
import pickle
import zipfile
import pickle
import zipfile
import pickle

with zipfile.ZipFile("income_prediction.zip", "r") as z:
    with z.open("income_prediction.pkl") as f:
        model = pickle.load(file)


# Page configuration
st.set_page_config(
    page_title="Income Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Income Prediction App")
st.write("Predict whether a person's annual income exceeds $50K.")

st.divider()

st.subheader("Enter Personal Details")

# User inputs
age = st.number_input("Age", 17, 90, 30)

workclass = st.selectbox(
    "Workclass",
    ["Private", "Self-emp-not-inc", "Self-emp-inc",
     "Federal-gov", "Local-gov", "State-gov",
     "Without-pay", "Never-worked"]
)

education = st.selectbox(
    "Education",
    ["Bachelors", "HS-grad", "11th", "Masters",
     "Some-college", "Assoc-acdm", "Assoc-voc",
     "Doctorate", "Prof-school", "10th", "9th",
     "12th", "7th-8th", "5th-6th", "1st-4th",
     "Preschool"]
)

education_num = st.number_input(
    "Education Number", 1, 16, 10
)

marital_status = st.selectbox(
    "Marital Status",
    ["Never-married", "Married-civ-spouse",
     "Divorced", "Separated", "Widowed",
     "Married-spouse-absent", "Married-AF-spouse"]
)

occupation = st.selectbox(
    "Occupation",
    ["Tech-support", "Craft-repair", "Other-service",
     "Sales", "Exec-managerial", "Prof-specialty",
     "Handlers-cleaners", "Machine-op-inspct",
     "Adm-clerical", "Farming-fishing",
     "Transport-moving", "Priv-house-serv",
     "Protective-serv", "Armed-Forces"]
)

relationship = st.selectbox(
    "Relationship",
    ["Wife", "Own-child", "Husband",
     "Not-in-family", "Other-relative", "Unmarried"]
)

race = st.selectbox(
    "Race",
    ["White", "Black", "Asian-Pac-Islander",
     "Amer-Indian-Eskimo", "Other"]
)

sex = st.selectbox("Sex", ["Male", "Female"])

hours = st.slider("Hours per week", 1, 99, 40)

capital_gain = st.number_input(
    "Capital Gain", min_value=0, value=0
)

capital_loss = st.number_input(
    "Capital Loss", min_value=0, value=0
)

fnlwgt = st.number_input(
    "Final Weight", min_value=1, value=100000
)

native_country = st.selectbox(
    "Native Country",
    ["United-States", "India", "Mexico", "Canada",
     "Germany", "Philippines", "Other"]
)

st.divider()

# Prediction button
if st.button("Predict Income", use_container_width=True):

    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours,
        "native-country": native_country
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == ">50K":
        st.success("Predicted Income: Above $50K")
    else:
        st.info("Predicted Income: $50K or Below")

    st.caption(
        "This is a machine learning prediction, "
        "not a guarantee of actual income."
    )
